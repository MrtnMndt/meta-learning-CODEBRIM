########################
# importing libraries
########################
# system libraries
import sys
import os
import shutil
import subprocess
import torch
import torch.utils.data


def save_checkpoint(state, is_best, save_path=None, filename='checkpoint.pth.tar'):
    """
    saves the checkpoint for the model at the end of a certain epoch

    Parameters:
        state (dictionary): dictionary containing model state-dict, optimizer state-dict etc
        is_best (bool): if True then this is the best model seen as of now 
        save_path (string): path to save the models to
        file_name (string): file-name to be used for saving the model
    """
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename), os.path.join(save_path, 'model_best.pth.tar'))


class GPUMem:
    def __init__(self, is_gpu):
        """
        reads the total gpu memory that the program can access and computes the amount of gpu memory available 

        Parameters:
            is_gpu (bool): says if the computation device is cpu or gpu

        Attributes:
            total_mem (float): total gpu memory that the program can access 
        """
        self.is_gpu = is_gpu
        if self.is_gpu:
            self.total_mem = self._get_total_gpu_memory()

    def _get_total_gpu_memory(self):
        """
        gets the total gpu memory that the program can access

        Returns:
            total gpu memory (float) that the program can access
        """
        total_mem = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.total",
                                             "--format=csv,noheader,nounits"])

        return float(total_mem[0:-1])  # gets rid of "\n" and converts string to float

    def get_mem_util(self):
        """
        gets the amount of gpu memory currently being used

        Returns:
            mem_util (float): amount of gpu memory currently being used
        """
        if self.is_gpu:
            # Check for memory of GPU ID 0 as this usually is the one with the heaviest use
            free_mem = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free",
                                                "--format=csv,noheader,nounits"])
            free_mem = float(free_mem[0:-1])    # gets rid of "\n" and converts string to float
            mem_util = 1 - (free_mem / self.total_mem)
        else:
            mem_util = 0
        return mem_util
