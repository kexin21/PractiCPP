import pandas as pd
from torch.utils.data import Dataset
import math
import numpy as np
import torch
import pickle as pkl
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def sample_negatives_Trans(device, net, k, N, num_pos, neg_train):
    pool_size = k * num_pos
    indices = random.sample(range(neg_train[0].shape[0]), pool_size)

    pool_pep_data = neg_train[0][indices].to(device)
    pool_esm = neg_train[1][indices].to(device)
    pool_targets = neg_train[2][indices].to(device)
    pool_fp = neg_train[3][indices].to(device)

    mask = (pool_pep_data == 0)
    mask = mask.to(device)

    logits = net(pool_pep_data, pool_esm, pool_fp, mask=mask)
    probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1].detach()
    _, hard_indices = torch.topk(probs, N * num_pos)
    return [pool_pep_data[hard_indices], pool_targets[hard_indices], pool_esm[hard_indices], pool_fp[hard_indices]]



def mix_pos_neg_Trans(device, pos_data, neg_data):
    total_length = pos_data[0].shape[0] + neg_data[0].shape[0]
    indices = torch.randperm(total_length)
    data = [torch.cat((pos_data[i].to(device), neg_data[i]), dim=0) for i in range(len(pos_data))]
    return data[0][indices], data[1][indices], data[2][indices], data[3][indices]


def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp + 1e-06)
    npv = float(tn) / (tn + fn + 1e-06)
    sensitivity = float(tp) / (tp + fn + 1e-06)
    specificity = float(tn) / (tn + fp + 1e-06)
    mcc = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-06)
    f1 = float(tp * 2) / (tp * 2 + fp + fn + 1e-06)
    return acc, precision, npv, sensitivity, specificity, mcc, f1


def Trans_evaluate(device, model, testloader):
    y_score, pred_y, label = [], [], []
    model.eval()
    with torch.no_grad():
        for pep_data, targets, esm, fp in testloader:
            pep_data = pep_data.to(device)
            esm = esm.to(device)
            fp = fp.to(device)
            targets = targets.to(device)
            mask = (pep_data == 0)
            mask = mask.to(device)
            outputs = model(pep_data, esm, fp, mask=mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            y_score.append(probabilities[:, 1].detach().cpu().numpy())
            pred_y.append(outputs.argmax(dim=1).cpu().numpy())
            label.append(targets.cpu().numpy())
    y_score = np.array(np.concatenate(y_score)).reshape(-1, 1)
    pred_y = np.array(np.concatenate(pred_y)).reshape(-1, 1)
    label = np.array(np.concatenate(label)).reshape(-1, 1)
    precision_1, recall_1, _ = precision_recall_curve(label, y_score)
    aupr = auc(recall_1, precision_1)
    acc, precision, npv, sensitivity, specificity, mcc, f1 = calculate_performace(len(pred_y), pred_y, label)
    return aupr, acc, precision, npv, sensitivity, specificity, mcc, f1



class PeptideDataset_Trans(Dataset):
    def __init__(self, pep_data, targets, ids, esm, morgan_fp):
        self.pep_data = pep_data
        self.targets = targets
        self.ids = ids
        self.esm = esm
        self.morgan_fp = morgan_fp

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.pep_data[idx], self.targets[idx], self.esm[idx], self.morgan_fp[idx]


def generate_features_target_Trans(pretrained_file, id_target_file, fingerprint_file):
    data = pd.read_csv(id_target_file)
    try:
        esm = torch.FloatTensor(torch.load(pretrained_file))
    except FileNotFoundError:
        esm = torch.zeros((len(data), 1))
        print('fail to load esm pretrained embedding!')
    targets = torch.LongTensor(data['label'])
    with open(fingerprint_file, 'rb') as f:
        fps_nonzero = pkl.load(f)
    fps = []
    for i in range(targets.shape[0]):
        fp = np.zeros(1024)
        fp[fps_nonzero[i]] = 1
        fps.append(fp)
    fps = np.stack(fps)
    morgan_fp = torch.FloatTensor(fps)
    ids = data['id']
    return ids, esm, targets, morgan_fp


