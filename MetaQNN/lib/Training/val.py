########################
# importing libraries
########################
# system libraries
import torch

# custom libraries
from lib.Utility.metrics import AverageMeter


def val(val_loader, model, criterion, device, is_val=True):
    """
    validates the model of a net for one epoch on the validation set

    Parameters:
        val_loader (torch.utils.data.DataLoader): data loader for the validation set
        model (lib.Models.network.Net): model of a net that has to be validated
        criterion (torch.nn.BCELoss): loss criterion
        device (torch.device): device name where data is transferred to
        is_val (bool): validation or testing mode

    Returns:
        losses.avg (float): average of the validation losses over the batches
        hard_prec.avg (float): average of the validation hard precision over the batches for all the defects
        soft_prec.avg (float): average of the validation soft precision over the batches for all the defects
        hard_prec_background.avg (float): average of the validation hard/soft precision over the batches for background
        hard_prec_crack.avg (float): average of the validation hard/soft precision over the batches for crack
        hard_prec_spallation.avg (float): average of the validation hard/soft precision over the batches for spallation
        hard_prec_exposed_bars.avg (float): average of the validation hard/soft precision over the batches for exposed
                                            bars
        hard_prec_efflorescence.avg (float): average of the validation hard/soft precision over the batches for
                                             efflorescence
        hard_prec_corrosion_stain.avg (float): average of the validation hard/soft precision over the batches for
                                               corrosion stain
    """
    # performance metrics
    losses = AverageMeter()
    hard_prec = AverageMeter()
    soft_prec = AverageMeter()
    hard_prec_background = AverageMeter()
    hard_prec_crack = AverageMeter()
    hard_prec_spallation = AverageMeter()
    hard_prec_exposed_bars = AverageMeter()
    hard_prec_efflorescence = AverageMeter()
    hard_prec_corrosion_stain = AverageMeter()

    # for computing correct prediction percentage for multi-label samples and single-label samples
    number_multi_label_samples = number_correct_multi_label_predictions = number_single_label_samples =\
        number_correct_single_label_predictions = 0

    # switch to evaluate mode
    model.eval()

    if is_val:
        print('validating')
    else:
        print('testing')

    #  to ensure no buffering for gradient updates
    with torch.no_grad():
        for i, (input_, target) in enumerate(val_loader):
            if input_.size(0) == 1:
                # hacky way to deal with terminal batch-size of 1
                print('skip last val/test batch of size 1')
                continue
            input_, target = input_.to(device), target.to(device)

            output = model(input_)

            loss = criterion(output, target)

            # update the 'losses' meter
            losses.update(loss.item(), input_.size(0))

            # compute performance measures
            output = output >= 0.5  # binarizing sigmoid output by thresholding with 0.5
            temp_output = output
            temp_output = temp_output.float() + (((torch.sum(temp_output.float(), dim=1, keepdim=True) == 0).float())
                                                 * 0.5)

            # compute correct prediction percentage for multi-label/single-label samples
            sum_target_along_label_dimension = torch.sum(target, dim=1, keepdim=True)

            multi_label_samples = (sum_target_along_label_dimension > 1).float()
            target_multi_label_samples = (target.float()) * multi_label_samples
            number_multi_label_samples += torch.sum(multi_label_samples).item()
            number_correct_multi_label_predictions += \
                torch.sum(torch.prod((temp_output.float() == target_multi_label_samples).float(), dim=1)).item()

            single_label_samples = (sum_target_along_label_dimension == 1).float()
            target_single_label_samples = (target.float()) * single_label_samples
            number_single_label_samples += torch.sum(single_label_samples).item()
            number_correct_single_label_predictions += \
                torch.sum(torch.prod((temp_output.float() == target_single_label_samples).float(), dim=1)).item()

            equality_matrix = (output.float() == target).float()
            hard = torch.mean(torch.prod(equality_matrix, dim=1)) * 100.
            soft = torch.mean(equality_matrix) * 100.
            hard_per_defect = torch.mean(equality_matrix, dim=0) * 100.

            # update peformance meters
            hard_prec.update(hard.item(), input_.size(0))
            soft_prec.update(soft.item(), input_.size(0))
            hard_prec_background.update(hard_per_defect[0].item(), input_.size(0))
            hard_prec_crack.update(hard_per_defect[1].item(), input_.size(0))
            hard_prec_spallation.update(hard_per_defect[2].item(), input_.size(0))
            hard_prec_exposed_bars.update(hard_per_defect[3].item(), input_.size(0))
            hard_prec_efflorescence.update(hard_per_defect[4].item(), input_.size(0))
            hard_prec_corrosion_stain.update(hard_per_defect[5].item(), input_.size(0))

    percentage_single_labels = (100. * number_correct_single_label_predictions) / number_single_label_samples
    percentage_multi_labels = (100. * number_correct_multi_label_predictions) / number_multi_label_samples

    if is_val:
        print(' * val: loss {losses.avg:.3f}, hard prec {hard_prec.avg:.3f}, soft prec {soft_prec.avg:.3f},\t'
              '% correct single-label predictions {percentage_single_labels:.3f}, % correct multi-label predictions'
              '{percentage_multi_labels:.3f},\t'
              'hard prec background {hard_prec_background.avg:.3f}, hard prec crack {hard_prec_crack.avg:.3f},\t'
              'hard prec spallation {hard_prec_spallation.avg:.3f}, '
              'hard prec exposed bars {hard_prec_exposed_bars.avg:.3f},\t'
              'hard prec efflorescence {hard_prec_efflorescence.avg:.3f}, hard prec corrosion stain'
              ' {hard_prec_corrosion_stain.avg:.3f}\t'
              .format(losses=losses, hard_prec=hard_prec, soft_prec=soft_prec,
                      percentage_single_labels=percentage_single_labels,
                      percentage_multi_labels=percentage_multi_labels, hard_prec_background=hard_prec_background,
                      hard_prec_crack=hard_prec_crack, hard_prec_spallation=hard_prec_spallation,
                      hard_prec_exposed_bars=hard_prec_exposed_bars, hard_prec_efflorescence=hard_prec_efflorescence,
                      hard_prec_corrosion_stain=hard_prec_corrosion_stain))
        print('*' * 80)
    else:
        print(' * test: loss {losses.avg:.3f}, hard prec {hard_prec.avg:.3f}, soft prec {soft_prec.avg:.3f},\t'
              '% correct single-label predictions {percentage_single_labels:.3f}, % correct multi-label predictions'
              '{percentage_multi_labels:.3f},\t'
              'hard prec background {hard_prec_background.avg:.3f}, hard prec crack {hard_prec_crack.avg:.3f},\t'
              'hard prec spallation {hard_prec_spallation.avg:.3f}, '
              'hard prec exposed bars {hard_prec_exposed_bars.avg:.3f},\t'
              'hard prec efflorescence {hard_prec_efflorescence.avg:.3f}, hard prec corrosion stain'
              '{hard_prec_corrosion_stain.avg:.3f}\t'
              .format(losses=losses, hard_prec=hard_prec, soft_prec=soft_prec,
                      percentage_single_labels=percentage_single_labels,
                      percentage_multi_labels=percentage_multi_labels,
                      hard_prec_background=hard_prec_background, hard_prec_crack=hard_prec_crack,
                      hard_prec_spallation=hard_prec_spallation, hard_prec_exposed_bars=hard_prec_exposed_bars,
                      hard_prec_efflorescence=hard_prec_efflorescence,
                      hard_prec_corrosion_stain=hard_prec_corrosion_stain))
        print('*' * 80)

    if is_val:
        return losses.avg, hard_prec.avg, soft_prec.avg, hard_prec_background.avg, hard_prec_crack.avg,\
               hard_prec_spallation.avg, hard_prec_exposed_bars.avg, hard_prec_exposed_bars.avg,\
               hard_prec_corrosion_stain.avg
    else:
        return None
