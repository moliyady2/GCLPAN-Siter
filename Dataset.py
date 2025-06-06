from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp


def create_dataset(root_x, root_y, dis_threshold):
    with open(root_y) as f:
        lines = f.readlines()
    # Remove newline characters
    lines = [line.strip() for line in lines]
    label_ptr = {}
    for i in range(0, len(lines), 2):
        key = lines[i][1:]
        value = list(map(int, lines[i + 1].split()))
        label_ptr[key] = value
    data_list = []
    for record in tqdm(SeqIO.parse(root_x, 'fasta')):
        x = np.loadtxt('./data/573_129/' + record.id + '.csv', delimiter=",")
        x = torch.FloatTensor(x)
        adj_matrix = np.loadtxt('./data/distance_matrix_573_129/' + record.id + '.csv',
                                delimiter=",")
        adj_matrix[adj_matrix < dis_threshold] = 1
        adj_matrix[adj_matrix >= dis_threshold] = 0
        np.fill_diagonal(adj_matrix, 0)
        # coo_matrix is a sparse matrix format, also known as coordinate format or triplet format,
        # which uses three numbers to store the row index, column index, and value of nonzero elements
        # Convert the adjacency matrix to the coo_matrix sparse matrix format
        adj_sparse_matrix = sp.coo_matrix(adj_matrix)
        edge_index, edge_attr = from_scipy_sparse_matrix(adj_sparse_matrix)
        y = torch.zeros(len(record.seq))
        for i in label_ptr[record.id]:
            y[i - 1] = 1
        data_list.append((Data(x=x, y=y, edge_index=edge_index)))
    return data_list


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, root_x=None, root_y=None,
                 out_filename=None,
                 dis_threshold=15):
        self.dis_threshold = dis_threshold
        self.root_x = root_x
        self.root_y = root_y
        self.out_filename = out_filename
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.root_x, self.root_y]

    @property
    def processed_file_names(self):
        return [self.out_filename]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = create_dataset(self.raw_file_names[0], self.raw_file_names[1], self.dis_threshold)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])