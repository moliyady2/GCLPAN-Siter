import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef


def evaluate(model, dataset):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for i, data in enumerate(dataset):
            device = torch.device("cuda")
            data = data.to(device)
            pred = model(data)
            pred = (pred).argmax(-1)
            predictions.append(pred)
            labels.append(data.y)

    predictions = torch.hstack(predictions).cpu().numpy()
    labels = torch.hstack(labels).cpu().numpy()
    TN, FP, FN, TP = confusion_matrix(labels, predictions).ravel()
    return matthews_corrcoef(labels, predictions), TP * 100 / (TP + FN), \
           TN * 100 / (TN + FP), (TP + TN) * 100 / (TP + TN + FP + FN), \
           TP / (TP + FP)
