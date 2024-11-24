import random
import torch
from sklearn.preprocessing import OneHotEncoder
from evaluate import evaluate
from Focal_Loss import FocalLoss
from graphsage import SAGE
from Dataset import MyOwnDataset
from torch_geometric.loader import DataLoader
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus


def main():
    test_x = 'autodl-fs/data/DNAPred_Dataset/PDNA-52_sequence.fasta'
    test_y = 'autodl-fs/data/DNAPred_Dataset/PDNA-52_label.fasta'
    train_x = 'autodl-fs/data/DNAPred_Dataset/PDNA-335_sequence.fasta'
    train_y = 'autodl-fs/data/DNAPred_Dataset/PDNA-335_label.fasta'
    train_dataset = MyOwnDataset(root='train', root_x=train_x, root_y=train_y,
                                 out_filename='335-18.pt', dis_threshold=18)
    test_dataset = MyOwnDataset(root='evaluate', root_x=test_x, root_y=test_y,
                                out_filename='52-18.pt', dis_threshold=18)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=10)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=4)

    my_net = SAGE()
    device = torch.device("cuda")
    my_net = my_net.to(device)
    enc = OneHotEncoder(sparse=False)
    optimizer = torch.optim.Adam(my_net.parameters(), lr=0.0001)
    crit = FocalLoss(alpha=.5, gamma=3)
    for epoch in range(100):
        loss_all = 0
        my_net.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = my_net(data)
            label = data.y.unsqueeze(1)
            label = torch.tensor(enc.fit_transform(label.cpu()), dtype=torch.float32).cuda()
            loss = crit(output, label)
            loss_all += loss.item()
            loss.backward()
            optimizer.step()
        print(f'loss:{loss_all / len(train_loader)}',end='\t')
        mcc, sen, spe, acc, pre = evaluate(my_net, test_loader)
        print(f'mcc:{mcc},sen:{sen},spe:{spe},acc:{acc},pre:{pre}')


if __name__ == "__main__":
    set_seed(12345)
    main()
