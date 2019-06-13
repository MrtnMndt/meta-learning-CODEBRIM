"""
Command line argument options parser.
Adopted and modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py

Usage with two minuses "- -". Options are written with a minus "-" in command line, but
appear with an underscore "_" in the attributes' list.

Attributes:
    task (int): available task options [architecture search | train fixed net] (default: 1)
    replay_buffer_csv_path (str): path to replay buffer csv (default: None)
    fixed_net_index_no (str): index of fixed net in replay buffer (default: -1)
    dataset (str): name of dataset (default: CODEBRIM)
    dataset_path (str): path to dataset (default: )
    workers (int): number of data loading workers (default: 4)
    patch_size (int): patch size for image crops (default: 224)
    normalize_data (bool): if True normalize data with the per channel mean and std
    continue_search (bool): if True continue an incomplete search
    q_values_csv_path (str): path to stored Q-values to continue search from (default: None)
    continue_epsilon (float): epsilon to continue search from, value should be from epsion schedule (default: 1.0)
    continue_ite (int): iteration to continue search from, value should be less than limit for epsilon (default: 1)
    weight_init (str): weight initialization method (default: kaiming-normal)
    early_stopping_threshold (float): threshold for early stopping (default: 0.15)
    early_stopping_epoch (int): epoch at which the check for early stopping is done (default: 1)
    epochs (int): number of epochs to train (default: 150)
    batch_size (int): mini-batch size (default: 16)
    momentum (float): momentum value (default: 0.9)
    weight_decay (float): l2-norm/weight-decay value (default: 0.0)
    batch_norm (float): batch normalization value (default: 1e-4)
    drop_out_drop (float): drop rate for drop out (default: 0.5)
    print_freq (int): frequency of printing performance measures in an epoch while training (default: 200)
    learning_rate (float): initial learning rate (default: 0.01)
    lr_wr_epochs (int): epochs defining one warm restart cycle (default: 10)
    lr_wr_mul (int): factor to grow warm restart cycle length after each cycle (default: 2)
    lr_wr_min (float): minimum learning rate to use in warm restarts (default: 1e-5)
    q_learning_rate (float): learning rate for the update rule (default: 0.1)
    q_discount_factor (float): discount factor for the update rule (default: 1.0)
    conv_layer_min_limit (int): minimum no. of conv layers [wrn block is counted as 2 conv layers] (default: 3)
    conv_layer_max_limit (int): maximum no. of conv layers [wrn block is counted as 2 conv layers] (default: 10)
    max_fc (int): maximum no. of hidden FC layers [classifier layer not counted] (default: 1)
"""

########################
# importing libraries
########################
# system libraries
import argparse

parser = argparse.ArgumentParser(description='Q-learning for MLP search')

# architecture search | train fixed net
parser.add_argument('-t', '--task', type=int, default=1, help='1: architecture search, 2: train fixed net (default: 1)')

# replay dictionary from a search and the net index no. for training search net (for continuing search or training
# fixed net)
parser.add_argument('--replay-buffer-csv-path', default=None, help='path to replay buffer (default: None)')
parser.add_argument('--fixed-net-index-no', type=int, default=-1, help='index of fixed net in replay buffer '
                                                                       '(default: -1)')

# dataset and loading
parser.add_argument('--dataset', default='CODEBRIM', help='name of dataset (default: CODEBRIM)')
parser.add_argument('--dataset-path', default='./datasets/CODEBRIM', help="path to dataset"
                                                                          "(default: './datasets/CODEBRIM')")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers '
                                                                              '(default: 4)')
parser.add_argument('-p', '--patch-size', default=224, type=int, metavar='P', help='patch size for crops'
                                                                                   '(default: 224)')
parser.add_argument('--normalize-data', type=bool, default=False, help='set to true to normalize dataset '
                                                                       '(default: False)')

# path to replay dictionary, Q-values, epsilon value and iteration no. to continue search from
parser.add_argument('--continue-search', type=bool, default=False, help='set the flag to continue search '
                                                                        '(default: False)')
parser.add_argument('--q-values-csv-path', default=None, help='path to stored Q-values to continue search'
                                                              '(default: None)')
parser.add_argument('--continue-epsilon', default=1.0, type=float, help='epsilon to continue search from '
                                                                        '(default: 1.0)')
parser.add_argument('--continue-ite', default=1, type=int, help='iteration to continue search from (default: 1)')

# weight initialization for model of net
parser.add_argument('--weight-init', default='kaiming-normal', metavar='W', help='weight-initialization scheme '
                                                                                 '(default: kaiming-normal)')

# precision for early stopping of a net while training
parser.add_argument('--early-stopping-thresh', default=0.15, type=float, help='threshold for early stopping '
                                                                              '(default: 0.15)')
parser.add_argument('--early-stopping-epoch', default=10, type=int, help='epoch for comparing with early '
                                                                         'stopping threshold (default: 10)')

# training and validation hyper-parameters
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run (default: 150)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='B', help='mini-batch size (default: 16)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default 0.9)')
parser.add_argument('-wd', '--weight-decay', default=0.0, type=float, metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('-bn', '--batch-norm', default=1e-4, type=float, metavar='BN', help='batch normalization '
                                                                                        '(default 1e-4)')
parser.add_argument('--drop-out-drop', default=0.5, type=float, help='drop out drop probability (default 0.5)')
parser.add_argument('-pf', '--print-freq', default=200, type=int, metavar='PF', help='print frequency (default: 200)')

# learning rate schedule
parser.add_argument('-lr', '--learning-rate', default=1e-2, type=float, metavar='LR', help='initial learning rate '
                                                                                           '(default: 0.01)')
parser.add_argument('--lr-wr-epochs', default=10, type=int, help='length of first warm restart cycle (default: 10)')
parser.add_argument('--lr-wr-mul', default=2, type=int, help='scaling factor for warm restarts (default: 2)')
parser.add_argument('--lr-wr-min', default=1e-5, type=float, help='minimum learning rate (default: 1e-5)')

# MetaQNN hyperparameters
parser.add_argument('--q-learning-rate', default=0.1, type=float, help='Q-learning rate (default: 0.1)')
parser.add_argument('--q-discount-factor', default=1.0, type=float, help='Q-learning discount factor (default: 1.0)')
parser.add_argument('--conv-layer-min-limit', default=3, type=int, help='Minimum amount of conv layers in model'
                                                                        '(default: 3)')
parser.add_argument('--conv-layer-max-limit', default=10, type=int, help='Maximum amount of conv layers in model'
                                                                         '(default: 10)')
parser.add_argument('--max-fc', default=1, type=int, help='maximum number of FC layers (default: 1)')
