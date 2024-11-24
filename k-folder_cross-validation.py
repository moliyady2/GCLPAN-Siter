import torch
from Focal_Loss import FocalLoss
from graphsage import SAGE
from Dataset import MyOwnDataset
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def main():
    val_x = 'autodl-fs/data/DNAPred_Dataset/PDNA-335_sequence.fasta'
    val_y = 'autodl-fs/data/DNAPred_Dataset/PDNA-335_label.fasta'
    val_dataset = MyOwnDataset(root='train', root_x=val_x, root_y=val_y,
                               out_filename='335-18.pt', dis_threshold=18)

    my_net = SAGE()

    device = torch.device("cuda")
    my_net = my_net.to(device)

    optimizer = torch.optim.Adam(my_net.parameters(), lr=0.0001)
    crit = FocalLoss(alpha=1, gamma=2)
    kf = KFold(n_splits=10)

    enc = OneHotEncoder(sparse=False)
    predictions = []
    labels = []
    for (train_indices, val_indices) in tqdm(kf.split(val_dataset), total=10):
        train_dataset = torch.utils.data.Subset(val_dataset, train_indices)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=4)
        test_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=2)

        my_net.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = my_net(data)
            label = data.y.unsqueeze(1)
            label = torch.tensor(enc.fit_transform(label.cpu()), dtype=torch.float32).cuda()
            loss = crit(output, label)
            loss.backward()
            optimizer.step()
        my_net.eval()
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                pred = my_net(data)
                pred = (pred).argmax(-1)
                predictions.append(pred)
                labels.append(data.y)

    predictions = torch.hstack(predictions).cpu().numpy()
    labels = torch.hstack(labels).cpu().numpy()
    TN, FP, FN, TP = confusion_matrix(labels, predictions).ravel()
    avg_mcc = matthews_corrcoef(labels, predictions)
    avg_sen = TP * 100 / (TP + FN)
    avg_spe = TN * 100 / (TN + FP)
    avg_acc = (TP + TN) * 100 / (TP + TN + FP + FN)
    avg_pre = TP / (TP + FP)
    print(f'mcc:{avg_mcc},sen:{avg_sen},spe:{avg_spe},acc:{avg_acc},pre:{avg_pre}')


if __name__ == "__main__":
    main()
