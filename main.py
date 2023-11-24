import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
import numpy as np
from utils import Trans_evaluate,PeptideDataset_Trans, generate_features_target_Trans, mix_pos_neg_Trans, sample_negatives_Trans
from model import TransModel
from parse import parse_args
import random
import pickle as pkl
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F



args = parse_args()
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def genData(file, max_len):
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23}
    with open(file, 'r') as inf:
        lines = inf.read().strip().split('\n')

    long_pep_counter = 0
    ids = []
    pep_codes = []
    labels = []
    pep_seq = []
    max_seq_len = 70
    for i in range(0, len(lines), 2):
        header = lines[i]
        pep = lines[i+1]
        label_str, id = header[1:].split('_')
        ids.append(int(id))
        if label_str == 'CPP':
            labels.append(1)
        else:
            labels.append(0)
        if not len(pep) > max_len:
            current_pep = []
            for aa in pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(torch.tensor(current_pep))
        else:
            long_pep_counter += 1
    print("length > 63:", long_pep_counter)
    data = rnn_utils.pad_sequence(pep_codes, batch_first=True)

    desired_length = 61
    if data.size(1) < desired_length:
        pad_size = desired_length - data.size(1)
        data = F.pad(data, (0,pad_size))

    return data, torch.tensor(labels), np.array(ids)


def read_length_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    sequences = [lines[i] for i in range(1, len(lines), 2)]  # Extract only sequence lines
    return [len(seq) for seq in sequences]

def pos_dataset():
    pretrained_file1 = '../data/pos_esm_pretrained'
    id_target_file1 = '../data/pos_id_target.csv'
    fingerprint_file1 = '../data/pos_fingerprint.pkl'
    pos_id, pos_esm, pos_targets, pos_morgan_fp = generate_features_target_Trans(pretrained_file1, id_target_file1,
                                                                                 fingerprint_file1)
    pep_data, labels, ids = genData("../data/merged-positive.txt", 61)
    # ids = range(pos_targets.shape[0])
    pos_dataset = PeptideDataset_Trans(pep_data, pos_targets, ids, pos_esm, pos_morgan_fp)


    pos_total_size = len(pos_dataset)
    pos_test_size = int(pos_total_size * 0.15)
    pos_val_size = int(pos_total_size * 0.15)
    pos_train_size = pos_total_size - pos_test_size - pos_val_size

    pos_train_dataset, pos_val_dataset, pos_test_dataset = random_split(pos_dataset,
                                                                        [pos_train_size, pos_val_size, pos_test_size])
    return pos_train_dataset, pos_val_dataset, pos_test_dataset

def neg_dataset(neg_ratio):
    pretrained_file2 = f'../unlabeled/sample_{neg_ratio}/final_mean_esm_{args.neg_ratio}'
    id_target_file2 = f'../unlabeled/sample_{neg_ratio}/sample_{neg_ratio}_id_target.csv'
    fingerprint_file2 = f'../unlabeled/sample_{neg_ratio}/sample_{neg_ratio}_fingerprint.pkl'
    neg_id, neg_esm, neg_targets, neg_morgan_fp = generate_features_target_Trans(pretrained_file2, id_target_file2,
                                                                                 fingerprint_file2)

    # neg_ids = range(neg_targets.shape[0])
    pep_data, labels, ids = genData(f'../unlabeled/sample_{neg_ratio}/sample_ratio_{neg_ratio}.fasta', 61)
    ids = ids + 650
    neg_dataset = PeptideDataset_Trans(pep_data, neg_targets, ids, neg_esm, neg_morgan_fp)

    neg_total_size = len(neg_dataset)
    neg_test_size = int(neg_total_size * 0.15)
    neg_val_size = int(neg_total_size * 0.15)
    neg_train_size = neg_total_size - neg_test_size - neg_val_size

    indices = torch.randperm(neg_total_size)

    neg_train_indices = indices[:neg_train_size]
    neg_val_indices = indices[neg_train_size:neg_train_size+neg_val_size]
    neg_test_indices = indices[neg_train_size+neg_val_size:]

    neg_train = [pep_data[neg_train_indices], neg_esm[neg_train_indices], neg_targets[neg_train_indices], neg_morgan_fp[neg_train_indices]]

    neg_val_dataset = Subset(neg_dataset, neg_val_indices)
    neg_test_dataset = Subset(neg_dataset, neg_test_indices)

    return neg_train, neg_val_dataset, neg_test_dataset


pos_train_dataset, pos_val_dataset, pos_test_dataset = pos_dataset()
neg_train, neg_val_dataset, neg_test_dataset = neg_dataset(args.neg_ratio)


# train_dataset = ConcatDataset([pos_train_dataset, neg_train_dataset])
val_dataset = ConcatDataset([pos_val_dataset, neg_val_dataset])
test_dataset = ConcatDataset([pos_test_dataset, neg_test_dataset])

# Now we have our best hyperparameters, we train our model with these hyperparameters on the complete training set
print('----------------------------------------')
print('Testing model')

net = TransModel()
print('using Transformer!')

net = net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

Pos_TrainLoader = DataLoader(pos_train_dataset, batch_size=args.train_batch, shuffle=True)
ValLoader = DataLoader(val_dataset, batch_size=args.test_batch, shuffle=True)
TestLoader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=True)



best_val_mcc = 0.0
for epoch in range(0,args.epoch):
    print(f'Starting epoch {epoch}')
    net.train()
    for i, pos_data in enumerate(Pos_TrainLoader, 0):
        num_pos = pos_data[0].shape[0]
        neg_data = sample_negatives_Trans(device, net, args.K, args.N, num_pos, neg_train)
        pep_data, targets, esm_emb, morgan_fp = mix_pos_neg_Trans(device, pos_data, neg_data)

        mask = (pep_data == 0)
        mask = mask.to(device)
        outputs = net(pep_data, esm_emb, morgan_fp, mask = mask)
        loss = torch.nn.CrossEntropyLoss(reduction='sum')(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        val_aupr, val_acc, val_precision, val_npv, val_sensitivity, val_specificity, val_mcc, val_f1 = Trans_evaluate(device, net, ValLoader)
        print(
            f'val loss : aupr={val_aupr}, precision={val_precision},sensitivity={val_sensitivity},specificity={val_specificity},f1={val_f1}')

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_epoch = epoch
            torch.save(net.state_dict(), f'./model_trained/x_fpseed{args.seed}_{args.neg_ratio}best_model_N{args.N}_K{args.K}')
print('Training process has finished.')
print(f'best epoch {best_epoch}')

best_net = TransModel()
best_net.load_state_dict(torch.load(f'./model_trained/x_fpseed{args.seed}_{args.neg_ratio}best_model_N{args.N}_K{args.K}',map_location=torch.device('cpu')))
best_net = best_net.to(device)

test_aupr, test_acc, test_precision, test_npv, test_sensitivity, test_specificity, test_mcc, test_f1 = Trans_evaluate(device, best_net, TestLoader)
print(
    f'test loss : aupr={test_aupr}, precision={test_precision},sensitivity={test_sensitivity},specificity={test_specificity},f1={test_f1}')


