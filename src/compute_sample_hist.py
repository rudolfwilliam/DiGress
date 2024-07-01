import moses
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from new_functions import mol_to_data, is_valid_mol, clean_and_convert_samples
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import TanimotoSimilarity
import rdkit.Chem as Chem
import numpy as np
import torch


def main(n=100, epsilon=0.15):
    #test = moses.get_dataset("test")
    test = np.load("moses_data.npy", allow_pickle=True)
    for i in range(n):
        model = DiscreteDenoisingDiffusion.load_from_checkpoint("./outputs/checkpoint_moses.ckpt", map_location=torch.device('cuda'))
        model.eval()
        ns = []
        mol = Chem.MolFromSmiles(test[i])
        mol_data = mol_to_data(mol)
        atom_count = mol.GetNumAtoms()
        # remove one arbitrary atom from the molecule that results in a valid molecule
        valid = False
        while not valid:
            idx = torch.randint(atom_count, (1,))[0].item()
            # Remove the ith row
            X_mod = torch.cat((mol_data[0][:, :idx], mol_data[0][:, idx+1:]), dim=1)
            E_mod = torch.cat((mol_data[1][:, :idx, :], mol_data[1][:, idx+1:, :]), dim=1)
            # Remove the ith column
            E_mod = torch.cat((E_mod[:, :, :idx], E_mod[:, :, idx+1:]), dim=2)
            mol_data_cropped = (X_mod[0, ...], E_mod[0, ...])
            mol_cropped = clean_and_convert_samples([mol_data_cropped])
            if len(mol_cropped) == 0:
                continue
            valid = is_valid_mol(mol_cropped[0])

        samples = model.sample_batch(batch_id=1, batch_size=128, keep_chain=1, number_chain_steps=1, save_final=1, num_nodes=atom_count, scaffold_mask=mol_data_cropped)
        samples = clean_and_convert_samples(samples)
        # check if the original molecule is in the epsilon neighborhood of the sampless
        fp_mol = MACCSkeys.GenMACCSKeys(mol)
        n = 1
        for sample in samples:
            # compute the tanimoto similarity between the original molecule and the sample macskeys
            fp_sample = MACCSkeys.GenMACCSKeys(sample)
            dist = 1. - TanimotoSimilarity(fp_mol, fp_sample)
            if dist <= epsilon:
                break
            n += 1
        if n == (len(samples) + 1):
            print("Original molecule not in the epsilon neighborhood of the samples")
            ns.append(float("inf"))
        else:
            ns.append(n)
        
        print(f"Average number of samples to find a molecule in the epsilon neighborhood: {sum(ns)/len(ns)}")


if __name__ == "__main__":
    main(n=100, epsilon=0.15)
