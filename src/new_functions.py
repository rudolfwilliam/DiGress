import torch
from torch.functional import F
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import SanitizeFlags
from hydra import compose, initialize
from torch_geometric.data import Data
from analysis.visualization import MolecularVisualization
import utils
import re

ATOM_DECODER = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H']
TYPES = {atom: i for i, atom in enumerate(ATOM_DECODER)}
BONDS = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


def extract_svg_content(svg_string):
    """
    Extracts the content inside the <svg>...</svg> tags from an SVG string.
    """
    svg_content = re.search(r'<svg[^>]*>(.*?)</svg>', svg_string, re.DOTALL).group(1)
    return svg_content


def mol_to_data(mol):
    N = mol.GetNumAtoms()

    type_idx = []
    for atom in mol.GetAtoms():
        type_idx.append(TYPES[atom.GetSymbol()])

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BONDS[bond.GetBondType()] + 1]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(BONDS) + 1).to(torch.float)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    x = F.one_hot(torch.tensor(type_idx), num_classes=len(TYPES)).float()
    y = torch.zeros(size=(1, 0), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=0)

    # Try to build the molecule again from the graph. If it fails, do not add it to the training set
    dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
    dense_data = dense_data.mask(node_mask, collapse=True)
    X, E = dense_data.X, dense_data.E

    return X, E


def clean_and_convert_samples(samples):
    from datasets.moses_dataset import MOSESinfos, MOSESDataModule
    """Sanitize and remove duplicates from a list of mol samples. Then convert to mol objects"""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
    datamodule = MOSESDataModule(cfg=cfg)
    #datamodule.prepare_data()
    infos = MOSESinfos(datamodule=datamodule)
    vis = MolecularVisualization(remove_h=True, dataset_infos=infos)
    samples_mol = [vis.mol_from_graphs(sample[0], sample[1]) for sample in samples]
    # sanitize molecules
    # Sanitize molecules and collect valid ones
    valid_samples_mol = []
    for sample in samples_mol:
        try:
            Chem.SanitizeMol(sample)
            valid_samples_mol.append(sample)
        except Exception as e:
            continue
    # remove molecules that consist of multiple disconnected components
    valid_samples_mol = [mol for mol in valid_samples_mol if len(Chem.GetMolFrags(mol, asMols=True)) == 1]
    # remove duplicates
    unique_mol_dict = {}
    for mol in valid_samples_mol:
        smiles = Chem.MolToSmiles(mol, canonical=True)  # Generate canonical SMILES
        if smiles not in unique_mol_dict:
            unique_mol_dict[smiles] = mol

    # Get the list of unique molecules
    unique_samples_mol = list(unique_mol_dict.values())
    return unique_samples_mol


def is_valid_mol(mol):
    """Check if a given RDKit molecule object is valid."""
    if mol is None:
        return False

    try:
        # Attempt to sanitize the molecule without kekulization
        Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_KEKULIZE)
        
        # Optionally, try to kekulize separately to identify issues
        try:
            Chem.Kekulize(mol)
        except Chem.KekulizeException:
            return False

        # Additional checks
        if mol.GetNumAtoms() == 0:
            return False
        
        # Check for disconnected components
        if len(Chem.GetMolFrags(mol, asMols=True)) > 1:
            return False

        return True
    except (ValueError, Chem.rdchem.KekulizeException) as e:
        return False
    
def generate_non_substructure_vis(mol, sub, height=500, width=500):
    """Higlight non-substructure atoms and bonds in a molecule as a svg."""
    from rdkit.Chem.Draw import rdMolDraw2D
    
    # Get substructure match
    hit_ats = set(mol.GetSubstructMatch(sub))
    all_ats = set(range(mol.GetNumAtoms()))
    non_hit_ats = list(all_ats - hit_ats)
    
    # Get bonds not in substructure match
    non_hit_bonds = []
    for bond in mol.GetBonds():
        aid1 = bond.GetBeginAtomIdx()
        aid2 = bond.GetEndAtomIdx()
        if aid1 in non_hit_ats or aid2 in non_hit_ats:
            non_hit_bonds.append(bond.GetIdx())
    
    # Create MolDraw2DSVG object
    d = rdMolDraw2D.MolDraw2DSVG(height, width)
    
    # Draw molecule with highlighted non-substructure
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=non_hit_ats, highlightBonds=non_hit_bonds)
    
    # Finish drawing
    d.FinishDrawing()
    
    # Get SVG string
    svg = d.GetDrawingText()
    
    return svg

def extract_svg_content(svg_string):
    """
    Extracts the content inside the <svg>...</svg> tags from an SVG string.
    """
    svg_content = re.search(r'<svg[^>]*>(.*?)</svg>', svg_string, re.DOTALL).group(1)
    return svg_content

def combine_svg_strings(svg_strings, output_file, height=500, width=500):
    # Read SVG strings
    svg_contents = []
    for svg_string in svg_strings:
        svg_contents.append(extract_svg_content(svg_string))
    
    # Create combined SVG content
    combined_svg = '<svg height="' + str(height) + '" width="' + str(len(svg_strings)*width) + '">'
    x_offset = 0
    for i, svg in enumerate(svg_contents):
        # Adjust x_offset to position each SVG side by side
        combined_svg += f'<g transform="translate({x_offset}, 0)">{svg}</g>'
        x_offset += width
    
    combined_svg += '</svg>'
    
    # Save combined SVG to output file
    with open(output_file, 'w') as f:
        f.write(combined_svg)
    
    print(f"Combined SVG saved to {output_file}.")
