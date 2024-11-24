import random
import torch
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.optim import Adam
from GCL.models import DualBranchContrast
from torch_geometric.nn.conv import SAGEConv as Conv
from Focal_Loss import FocalLoss
from Dataset import MyOwnDataset
import warnings
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix, matthews_corrcoef
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda')
test_x = 'data/DNAPred_Dataset/PDNA-41_sequence.fasta'
test_y = 'data/DNAPred_Dataset/PDNA-41_label.fasta'
train_x = 'data/DNAPred_Dataset/PDNA-543_sequence.fasta'
train_y = 'data/DNAPred_Dataset/PDNA-543_label.fasta'
train_dataset = MyOwnDataset(root='train', root_x=train_x, root_y=train_y,
                             out_filename='543-15.pt', dis_threshold=8)
test_dataset = MyOwnDataset(root='evaluate', root_x=test_x, root_y=test_y,
                            out_filename='41-15.pt', dis_threshold=8)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=12)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=4)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus


class SAGE(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(SAGE, self).__init__()
        self.conv1 = Conv(in_channels=in_c, out_channels=hid_c, aggr='mean')
        self.conv2 = Conv(in_channels=hid_c, out_channels=out_c, aggr='mean')

    def forward(self, x, edge_index):
        hid = self.conv1(x=x, edge_index=edge_index)
        hid = F.relu(hid)

        out = self.conv2(x=hid, edge_index=edge_index)
        out = F.relu(out)
        return out


class KAN(nn.Module):
    def __init__(self, in_c, hid_c1, hid_c2, out_c):
        super(KAN, self).__init__()
        self.lin1 = torch.nn.Linear(in_c, hid_c1)
        self.poly1 = torch.nn.Linear(hid_c1, hid_c1, bias=False)  # Polynomial term 1
        self.poly2 = torch.nn.Linear(hid_c1, hid_c1, bias=False)  # Polynomial term 2
        self.lin2 = torch.nn.Linear(hid_c1, hid_c2)
        self.poly3 = torch.nn.Linear(hid_c2, hid_c2, bias=False)  # Polynomial term 3
        self.lin3 = torch.nn.Linear(hid_c2, out_c)

    def forward(self, x):
        # First hidden layer with polynomial terms
        hid = self.lin1(x)
        poly_term1 = self.poly1(hid) * hid
        poly_term2 = self.poly2(hid) * hid ** 2
        hid = F.relu(hid + poly_term1 + poly_term2)

        # Second hidden layer with polynomial terms
        hid = self.lin2(hid)
        poly_term3 = self.poly3(hid) * hid
        hid = F.relu(hid + poly_term3)

        # Output layer
        out = self.lin3(hid)
        return out


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index)
        z1 = self.encoder(x1, edge_index1)
        z2 = self.encoder(x2, edge_index2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]

    # compute extra pos and neg masks for semi-supervised learning
    extra_pos_mask = torch.eq(data.y, data.y.unsqueeze(dim=1)).to('cuda')
    extra_pos_mask.fill_diagonal_(False)
    # pos_mask: [N, 2N] for both inter-view and intra-view samples
    extra_pos_mask = torch.cat([extra_pos_mask, extra_pos_mask], dim=1).to('cuda')
    # fill interview positives only; pos_mask for intraview samples should have zeros in diagonal
    extra_pos_mask.fill_diagonal_(True)

    extra_neg_mask = torch.ne(data.y, data.y.unsqueeze(dim=1)).to('cuda')
    extra_neg_mask.fill_diagonal_(False)
    extra_neg_mask = torch.cat([extra_neg_mask, extra_neg_mask], dim=1).to('cuda')

    loss = contrast_model(h1=h1, h2=h2, extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(encoder, classifier, optimizer, criterion):
    encoder.eval()
    train_hid = []
    test_hid = []
    train_labels = []
    test_labels = []
    enc = OneHotEncoder(sparse=False)

    with torch.no_grad():
        for data in train_loader:
            data = data.cuda()
            z, _, _ = encoder(data.x, data.edge_index)
            train_hid.append(z)
            label = torch.FloatTensor(enc.fit_transform(data.y.unsqueeze(1).cpu())).cuda()
            train_labels.append(label)

        for data in test_loader:
            data = data.cuda()
            z, _, _ = encoder(data.x, data.edge_index)
            test_hid.append(z)
            test_labels.append(data.y)

    train_labels = torch.vstack(train_labels)
    test_labels = torch.hstack(test_labels).cpu().numpy()
    train_hid = torch.vstack(train_hid)
    test_hid = torch.vstack(test_hid)

    best_mcc = 0
    best_epoch = 0
    best_metrics = None

    for epoch in range(100):
        classifier.train()
        optimizer.zero_grad()
        output = classifier(train_hid)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()

        classifier.eval()
        test_pred = classifier(test_hid)
        test_pred_probs = torch.softmax(test_pred, dim=1)[:, 1].detach().cpu().numpy()  # 获取预测概率  # 获取预测概率
        # test_pred_probs = torch.max(test_pred, dim=1).values.detach().cpu().numpy()
        # 预测类别标签
        test_pred_classes = test_pred.argmax(1).cpu().numpy()

        # 计算混淆矩阵并提取 TN, FP, FN, TP
        TN, FP, FN, TP = confusion_matrix(test_labels, test_pred_classes).ravel()

        # 计算各个指标
        mcc = matthews_corrcoef(test_labels, test_pred_classes)
        sen = TP / (TP + FN)  # Recall / Sensitivity
        spe = TN / (TN + FP)  # Specificity
        acc = (TP + TN) / (TP + TN + FP + FN)  # Accuracy
        pre = TP / (TP + FP)  # Precision
        f1 = f1_score(test_labels, test_pred_classes)  # F1 score
        auc = roc_auc_score(test_labels, test_pred_probs)  # AUC
        aupr = average_precision_score(test_labels, test_pred_probs)  # AUPR

        # 更新最佳指标
        if mcc > best_mcc:
            best_mcc = mcc
            best_epoch = epoch
            best_metrics = {
                'sen': sen,
                'spe': spe,
                'acc': acc,
                'pre': pre,
                'f1': f1,
                'auc': auc,
                'aupr': aupr
            }

    # 输出最佳结果
    print(f'best_mcc: {best_mcc}, sen: {best_metrics["sen"]}, spe: {best_metrics["spe"]}, '
          f'acc: {best_metrics["acc"]}, pre: {best_metrics["pre"]}, f1: {best_metrics["f1"]}, '
          f'auc: {best_metrics["auc"]}, aupr: {best_metrics["aupr"]}, best_epoch: {best_epoch}')


# noinspection PyTypeChecker
def main():
    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.4)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.4)])

    gconv = SAGE(in_c=2560, hid_c=1680, out_c=640).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=640, proj_dim=640).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)
    optimizer_enc = Adam(encoder_model.parameters(), lr=0.01)

    kan = KAN(640, 320, 80, 2).to(device)
    optimizer_kan = Adam(kan.parameters(), lr=0.001)
    crit = FocalLoss(alpha=0.25, gamma=3)

    for epoch in range(9):
        print(f'epoch:{epoch + 1}', end=' ')
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            loss = train(encoder_model, contrast_model, data, optimizer_enc)
            loss_all += loss
        print(f'loss:{loss_all / len(train_loader)}')

        # 使用学习率调度器

    evaluate(encoder_model, kan, optimizer_kan, crit)

    # 使用学习率调度器


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    set_seed(12345)
    main()