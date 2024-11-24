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
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast, GradScaler

torch.cuda.set_per_process_memory_fraction(0.5, device=0)
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = GradScaler()

test_x = 'autodl-tmp/data/DNAPred_Dataset/PDNA-41_sequence.fasta'
test_y = 'autodl-tmp/data/DNAPred_Dataset/PDNA-41_label.fasta'
train_x = 'autodl-tmp/data/DNAPred_Dataset/PDNA-543_sequence.fasta'
train_y = 'autodl-tmp/data/DNAPred_Dataset/PDNA-543_label.fasta'
train_dataset = MyOwnDataset(root='train', root_x=train_x, root_y=train_y,
                             out_filename='543-15.pt', dis_threshold=8)
test_dataset = MyOwnDataset(root='evaluate', root_x=test_x, root_y=test_y,
                            out_filename='41-15.pt', dis_threshold=8)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=4)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class TransformerModel(nn.Module):
    def __init__(self, in_c, hid_c1, out_c, num_layers=1, nhead=1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(in_c, hid_c1)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hid_c1, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hid_c1, out_c)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        out = self.fc_out(x)
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
    with autocast():
        z, z1, z2 = encoder_model(data.x, data.edge_index)
        h1, h2 = [encoder_model.project(x) for x in [z1, z2]]

        extra_pos_mask = torch.eq(data.y, data.y.unsqueeze(dim=1)).to(device)
        extra_pos_mask.fill_diagonal_(False)
        extra_pos_mask = torch.cat([extra_pos_mask, extra_pos_mask], dim=1).to(device)
        extra_pos_mask.fill_diagonal_(True)

        extra_neg_mask = torch.ne(data.y, data.y.unsqueeze(dim=1)).to(device)
        extra_neg_mask.fill_diagonal_(False)
        extra_neg_mask = torch.cat([extra_neg_mask, extra_neg_mask], dim=1).to(device)

        loss = contrast_model(h1=h1, h2=h2, extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)

    # Check for NaNs in loss
    if torch.isnan(loss).any():
        print("Loss has NaN values")

    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), max_norm=1.0)  # Gradient clipping
    scaler.step(optimizer)
    scaler.update()
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
            data = data.to(device)
            z, _, _ = encoder(data.x, data.edge_index)
            train_hid.append(z)
            label = torch.FloatTensor(enc.fit_transform(data.y.unsqueeze(1).cpu())).to(device)
            train_labels.append(label)

        for data in test_loader:
            data = data.to(device)
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

    for epoch in range(1000):
        classifier.train()
        optimizer.zero_grad()
        with autocast():
            output = classifier(train_hid)
            loss = criterion(output, train_labels)

        # Check for NaNs in loss
        if torch.isnan(loss).any():
            print(f"Loss has NaN values at epoch {epoch}")

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)  # Gradient clipping
        scaler.step(optimizer)
        scaler.update()

        classifier.eval()
        with torch.no_grad():
            test_pred = classifier(test_hid)

        TN, FP, FN, TP = confusion_matrix(test_labels, test_pred.argmax(1).cpu().numpy()).ravel()
        mcc = matthews_corrcoef(test_labels, test_pred.argmax(1).cpu().numpy())

        sen = TP * 100 / (TP + FN)
        spe = TN * 100 / (TN + FP)
        acc = (TP + TN) * 100 / (TP + TN + FP + FN)
        pre = TP / (TP + FP)

        if mcc > best_mcc:
            best_mcc = mcc
            best_epoch = epoch
            best_metrics = {'sen': sen, 'spe': spe, 'acc': acc, 'pre': pre}

    if best_metrics is not None:
        print(
            f'best_mcc: {best_mcc}, sen: {best_metrics["sen"]}, spe: {best_metrics["spe"]}, acc: {best_metrics["acc"]}, pre:{best_metrics["pre"]}, best_epoch: {best_epoch}')
    else:
        print("Evaluation metrics are None")


def main():
    aug1 = A.Compose([A.EdgeRemoving(pe=0.4), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.4), A.FeatureMasking(pf=0.3)])

    gconv = SAGE(in_c=2560, hid_c=1280, out_c=640).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=640, proj_dim=640).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)
    optimizer_enc = Adam(encoder_model.parameters(), lr=0.0001)  # Lower learning rate

    transformer_model = TransformerModel(640, 80, 2, num_layers=1, nhead=1).to(device)  # 使用更小的Transformer
    optimizer_transformer = Adam(transformer_model.parameters(), lr=0.0001)
    crit = FocalLoss(alpha=0.25, gamma=3)

    for epoch in range(8):
        print(f'epoch:{epoch + 1}', end=' ')
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            loss = train(encoder_model, contrast_model, data, optimizer_enc)
            loss_all += loss

        print(f'loss:{loss_all / len(train_loader)}')
        torch.cuda.empty_cache()

    evaluate(encoder_model, transformer_model, optimizer_transformer, crit)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    set_seed(12345)
    main()