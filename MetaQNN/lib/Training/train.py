########################
# importing libraries
########################
# system libraries
import time
import math
import torch

# custom libraries
from lib.Utility.metrics import AverageMeter


def train(train_loader, model, criterion, epoch, optimizer, lr_scheduler, device, args, split_batch_size):
    """
    trains the model of a net for one epoch on the train set
    
    Parameters:
        train_loader (torch.utils.data.DataLoader): data loader for the train set
        model (lib.Models.network.Net): model of the net to be trained
        criterion (torch.nn.BCELoss): loss criterion to be optimized
        epoch (int): continuous epoch counter
        optimizer (torch.optim.SGD): optimizer instance like SGD or Adam
        lr_scheduler (lib.Training.learning_rate_scheduling.LearningRateScheduler): class implementing learning rate
                                                                                    schedules
        device (torch.device): computational device (cpu or gpu)
        args (argparse.ArgumentParser): parsed command line arguments
        split_batch_size (int):  smaller batch size after splitting the original batch size for fitting the device
                                 memory
    """
    # performance and computational overhead metrics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    hard_prec = AverageMeter()
    soft_prec = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    factor = args.batch_size//split_batch_size
    last_batch = int(math.ceil(len(train_loader.dataset)/float(split_batch_size)))

    optimizer.zero_grad()

    print('training')

    for i, (input_, target) in enumerate(train_loader):
        # hacky way to deal with terminal batch-size of 1
        if input_.size(0) == 1:
            print('skip last training batch of size 1')
            continue

        input_, target = input_.to(device), target.to(device)

        data_time.update(time.time() - end)

        # adjust learning rate after every 'factor' times 'batch count' (after every batch had the batch size not been
        # split)
        if i % factor == 0:
            lr_scheduler.adjust_learning_rate(optimizer, i//factor + 1)

        output = model(input_)

        # scale the loss by the ratio of the split batch size and the original
        loss = criterion(output, target) * input_.size(0) / float(args.batch_size)

        # update the 'losses' meter with the actual measure of the loss
        losses.update(loss.item() * args.batch_size / float(input_.size(0)), input_.size(0))

        # compute performance measures
        output = output >= 0.5  # binarizing sigmoid output by thresholding with 0.5
        equality_matrix = (output.float() == target).float()
        hard = torch.mean(torch.prod(equality_matrix, dim=1)) * 100.
        soft = torch.mean(equality_matrix) * 100.

        # update peformance meters
        hard_prec.update(hard.item(), input_.size(0))
        soft_prec.update(soft.item(), input_.size(0))

        loss.backward()

        # update the weights after every 'factor' times 'batch count' (after every batch had the batch size not been
        # split)
        if (i+1) % factor == 0 or i == (last_batch - 1):
            optimizer.step()
            optimizer.zero_grad()

        del output, input_, target
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print performance and computational overhead measures after every 'factor' times 'batch count' (after every
        # batch had the batch size not been split)
        if i % (args.print_freq * factor) == 0:
            print('epoch: [{0}][{1}/{2}]\t'
                  'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  'hard prec {hard_prec.val:.3f} ({hard_prec.avg:.3f})\t'
                  'soft prec {soft_prec.val:.3f} ({soft_prec.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, losses=losses, hard_prec=hard_prec, soft_prec=soft_prec))

    lr_scheduler.scheduler_epoch += 1

    print(' * train: loss {losses.avg:.3f} hard prec {hard_prec.avg:.3f} soft prec {soft_prec.avg:.3f}'
          .format(losses=losses, hard_prec=hard_prec, soft_prec=soft_prec))
    print('*' * 80)
