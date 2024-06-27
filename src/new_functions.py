import torch
from torch.functional import F
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import SanitizeFlags
from torch_geometric.data import Data
import utils

ATOM_DECODER = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H']
TYPES = {atom: i for i, atom in enumerate(ATOM_DECODER)}
BONDS = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

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

def is_valid_molecule(mol):
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
            print("Warning: Molecule could not be kekulized, but will be considered valid.")

        # Additional checks
        if mol.GetNumAtoms() == 0:
            return False

        return True
    except (ValueError, Chem.rdchem.KekulizeException) as e:
        print(f"Sanitization error: {e}")
        return False