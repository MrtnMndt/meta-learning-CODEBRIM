########################
# importing libraries
########################
# system libraries
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn

# custom libraries
from lib.Models.network import Net
from lib.Training.train import train
from lib.Training.val import val
from lib.Training.learning_rate_scheduling import LearningRateScheduler
from lib.Utility.utils import GPUMem
from lib.Utility.utils import save_checkpoint


def train_val_net(state_list, dataset, weight_initializer, device, args, save_path):
    """
    builds a net given a state list, and trains and validates it
    
    Parameters:
        state_list (list): list of states to build the net
        dataset (lib.Datasets.datasets.CODEBRIM): dataset to train and validate the net on
        weight_initializer (lib.Models.initialization.WeightInit): weight initializer for initializing the weights of
                                                                   the network
        device (torch.device): type of computational device available (cpu / gpu)
        args (argparse.ArgumentParser): parsed command line arguments
        save_path (string): path for saving results to
    
    Returns:
        memfit (bool): True if the network fits the memory after batch splitting, False otherwise
        val_acc_all_epochs (list): list of validation accuracies in all epochs
        train_flag (bool): False if net's been early-stopped, False otherwise
    """
    # reset the data loaders
    dataset.train_loader, dataset.val_loader, dataset.test_loader = dataset.get_dataset_loader(args.batch_size,
                                                                                               args.workers,
                                                                                               torch.cuda.is_available()
                                                                                               )
    net_input, _ = next(iter(dataset.train_loader))

    num_classes = dataset.num_classes
    batch_size = net_input.size(0)

    # gets number of available gpus and total gpu memory
    num_gpu = float(torch.cuda.device_count())
    gpu_mem = GPUMem(torch.device('cuda') == device)

    # builds the net from the state list
    model = Net(state_list, num_classes, net_input, args.batch_norm, args.drop_out_drop)

    print(model)
    print('*' * 80)
    print('no. of spp scales: {}'.format(model.spp_size))
    print('*' * 80)

    # sets cudnn benchmark flag
    cudnn.benchmark = True

    # initializes weights
    weight_initializer.init_model(model)

    # puts model on gpu/cpu
    model = model.to(device)

    # gets available gpu memory
    gpu_avail = (gpu_mem.total_mem - gpu_mem.total_mem * gpu_mem.get_mem_util()) / 1024.
    print('gpu memory available:{gpu_avail:.4f}'.format(gpu_avail=gpu_avail))

    # prints estimated gpu requirement of model but actual memory requirement is higher than what's estimated (from
    # experiments)
    print("model's estimated gpu memory requirement: {gpu_mem_req:.4f} GB".format(gpu_mem_req=model.gpu_mem_req))

    # scaling factor and buffer for matching expected memory requirement with empirically observed memory requirement
    scale_factor = 4.0
    scale_buffer = 1.0
    scaled_gpu_mem_req = (scale_factor / num_gpu) * model.gpu_mem_req + scale_buffer
    print("model's empirically scaled gpu memory requirement: {scaled_gpu_mem_req:.4f}".format(scaled_gpu_mem_req=
                                                                                               scaled_gpu_mem_req))
    split_batch_size = batch_size
    # splits batch into smaller batches
    if gpu_avail < scaled_gpu_mem_req:
        # estimates split batch size as per available gpu mem. (may not be a factor of original batch size)
        approx_split_batch_size = int(((gpu_avail - scale_buffer) * num_gpu / scale_factor) //
                                      (model.gpu_mem_req / float(batch_size)))

        diff = float('inf')
        temp_split_batch_size = 1
        # sets split batch size such that it's close to the estimated split batch size, is also a factor of original
        # batch size & should give a terminal batch size of more than 1
        for j in range(2, approx_split_batch_size + 1):
            if batch_size % j == 0 and abs(j - approx_split_batch_size) < diff and (len(dataset.train_set) % j > 1):
                diff = abs(j - approx_split_batch_size)
                temp_split_batch_size = j
        split_batch_size = temp_split_batch_size

    print('split batch size:{}'.format(split_batch_size))
    print('*' * 80)

    # returns memfit = False if model doesn't fit in memory even after splitting the batch size to as small as 1
    if split_batch_size < 2:
        return False, None, None, None, None, None, False, None, None, None, None, None, None

    # set the data loaders using the split batch size
    dataset.train_loader, dataset.val_loader, dataset.test_loader = dataset.get_dataset_loader(split_batch_size,
                                                                                               args.workers,
                                                                                               torch.cuda.is_available()
                                                                                               )

    # use data parallelism for multi-gpu machine
    model = torch.nn.DataParallel(model)

    # cross entropy loss criterion (LogSoftmax and NLLoss together)
    criterion = nn.BCELoss(reduction='mean').to(device)

    # SGD optimizer with warm restarts
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    # quarter cosine learning rate schedule for SGD with warm restarts
    lr_scheduler = LearningRateScheduler(args.lr_wr_epochs, len(dataset.train_loader.dataset), args.batch_size,
                                         args.learning_rate, args.lr_wr_mul, args.lr_wr_min)

    train_flag = True
    epoch = 0
    loss_val_all_epochs = []
    hard_val_all_epochs = []
    soft_val_all_epochs = []
    hard_best_background = 0.0
    hard_best_crack = 0.0
    hard_best_spallation = 0.0
    hard_best_exposed_bars = 0.0
    hard_best_efflorescence = 0.0
    hard_best_corrosion_stain = 0.0

    while epoch < args.epochs:
        # train and validate the model
        train(dataset.train_loader, model, criterion, epoch, optimizer, lr_scheduler, device, args, split_batch_size)
        loss_val, hard_val, soft_val, hard_background, hard_crack, hard_spallation, hard_exposed_bars,\
            hard_efflorescence, hard_corrosion_stain = val(dataset.val_loader, model, criterion, device)
        if int(args.task) == 2:
            _ = val(dataset.test_loader, model, criterion, device, is_val=False)

        if len(hard_val_all_epochs) == 0 or hard_val == max(hard_val_all_epochs):
            hard_best_background = hard_background
            hard_best_crack = hard_crack
            hard_best_spallation = hard_spallation
            hard_best_exposed_bars = hard_exposed_bars
            hard_best_efflorescence = hard_efflorescence
            hard_best_corrosion_stain = hard_corrosion_stain
        loss_val_all_epochs.append(loss_val)
        hard_val_all_epochs.append(hard_val)
        soft_val_all_epochs.append(soft_val)

        if int(args.task) == 2:
            # saves model dict while training fixed net
            state = {'epoch': epoch,
                     'arch': 'Fixed net: replay buffer - {}, index no - {}'.format(args.replay_buffer_csv_path,
                                                                                   args.fixed_net_index_no),
                     'state_dict': model.state_dict(),
                     'hard_val': hard_val,
                     'optimizer': optimizer.state_dict()
                     }
            save_checkpoint(state, max(hard_val_all_epochs) == hard_val, save_path)

        # checks for early stopping; early-stops if the mean of the validation accuracy from the last 3 epochs before
        # the early stopping epoch isn't at least as high as the early stopping threshold
        if epoch == (args.early_stopping_epoch - 1) and float(np.mean(hard_val_all_epochs[-5:])) <\
                (args.early_stopping_thresh * 100.):
            train_flag = False
            break

        epoch += 1
    hard_best_val = max(hard_val_all_epochs)
    soft_best_val = max(soft_val_all_epochs)

    # free up memory by deleting objects
    spp_size = model.module.spp_size
    del model, criterion, optimizer, lr_scheduler

    return True, spp_size, hard_best_val, hard_val_all_epochs, soft_best_val, soft_val_all_epochs, train_flag,\
           hard_best_background, hard_best_crack, hard_best_spallation, hard_best_exposed_bars,\
           hard_best_efflorescence, hard_best_corrosion_stain
