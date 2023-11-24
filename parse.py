import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go CPP Prediction")
    parser.add_argument('--seed', type=int,default=2)
    parser.add_argument('--neg_ratio', type=int,default=1000,
                        help="ratio of negatives to positives")
    parser.add_argument('--epoch', type=int,default=301)
    parser.add_argument('--K', type=int,default=12, help='randomly sample negatives as K times |positives|')
    parser.add_argument('--N', type=int,default=3, help='|positives|:|negatives| in a batch is 1:N')
    parser.add_argument('--lr', type=float,default=0.00005)
    parser.add_argument('--weight_decay', type=float,default=1e-4)
    parser.add_argument('--train_batch', type=int,default=32)
    parser.add_argument('--test_batch', type=int,default=32)
    parser.add_argument("--cuda", type=bool, default=False, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")

    return parser.parse_args()