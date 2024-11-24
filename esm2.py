import torch
from Bio import SeqIO
from tqdm import tqdm
import numpy as np

# Load ESM-2 model
model, alphabet = torch.hub.load("facebookresearch/esm:main", 'esm2_t36_3B_UR50D')
model.eval().cuda()  # disables dropout for deterministic results
batch_converter = alphabet.get_batch_converter()


def esm2(data):
    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    batch_tokens = batch_tokens.to(next(model.parameters()).device)

    # Extract per-residue representations
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[36], return_contacts=False)
    token_representations = results["representations"][36].cpu().numpy()

    return token_representations[0, 1: -1]


def main():
    file_path = 'data/DNAPred_Dataset/PDNA-46_sequence.fasta'
    # file_path = 'autodl-fs/data/DNAPred_Dataset/PDNA-52_sequence.fasta'

    for record in tqdm(SeqIO.parse(file_path, 'fasta')):
        data = [
            (record.id, str(record.seq)),
        ]
        matrix = esm2(data)
        np.savetxt('data/646_46/' + record.id + '.csv', matrix, delimiter=',')


if __name__ == '__main__':
    main()
