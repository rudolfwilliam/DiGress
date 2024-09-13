import moses
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from new_functions import mol_to_data, is_valid_mol, clean_and_convert_samples
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import TanimotoSimilarity
import rdkit.Chem as Chem
import numpy as np
from random import choice
import torch


def main(n=100, epsilon=0.15):
    #test = moses.get_dataset("test")
    model = DiscreteDenoisingDiffusion.load_from_checkpoint("./outputs/checkpoint_moses.ckpt", map_location=torch.device('cuda'))
    model.eval()
    ns = []
    dists = []
    test = np.load("moses_data.npy", allow_pickle=True)
    for i in range(n):
        mol = Chem.MolFromSmiles(test[i])
        mol_data = mol_to_data(mol)
        
        if len(idxs) == 0:
            print("Could not find a valid molecule")
            continue
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
            dists.append(float("inf"))
        else:
            ns.append(n)
            dists.append(dist)
        if i % 10 == 0:
            print(f"Sample {i}")
            np.save("ns.npy", ns)
            np.save("dists.npy", dists)
    
    # save the results
    np.save("ns.npy", ns)
    np.save("dists.npy", dists)
    print(f"Average number of samples to find a molecule in the epsilon neighborhood: {sum(ns)/len(ns)}")


if __name__ == "__main__":
    main(n=100, epsilon=0.15)
