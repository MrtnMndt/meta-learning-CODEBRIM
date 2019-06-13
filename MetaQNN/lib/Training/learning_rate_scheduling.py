import math


class LearningRateScheduler:
    """
    learning rate schedule for SGD with warm restarts (Loshchilov & Hutter - https://arxiv.org/abs/1608.03983)

    Parameters:
            wr_epochs (int): epochs defining one warm restart cycle
            dataset_size (int): amount of samples in dataset
            batch_size (int): chosen batch size
            min_learning_rate (float): minimum learning rate to use in warm restarts
            max_learning_rate (float): maximum (initial) learning rate to use in warm restarts
            wr_mul (int): factor to grow warm restart cycle length after each cycle

    Attributes:
        scheduler_epoch (int): ongoing epoch in the learning rate schedule
    """

    def __init__(self, wr_epochs, dataset_size, batch_size, max_learning_rate, wr_mul=2, min_learning_rate=1e-5):

        self.max_lr = max_learning_rate
        self.scheduler_epoch = 0

        # Warm Restarts
        self.wr_epochs = wr_epochs
        self.min_lr = min_learning_rate
        self.wr_mul = wr_mul
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def adjust_learning_rate(self, optimizer, batch_count):
        """
        applies adjusted learning rate to the optimizer
        :param optimizer: optimizer for minimizing the net loss
        :param batch_count: counter for batches in an epoch
        :return adjusted learning rate
        """
        lr = self.__warm_restart_learning_rate(batch_count)

        # apply the new learning rate to the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def __warm_restart_cycle_length(self):
        """
        adjusts the warm restart cycle length
        """

        if self.scheduler_epoch > 0:
            if self.scheduler_epoch % self.wr_epochs == 0:
                self.wr_epochs = self.wr_epochs * self.wr_mul
                self.scheduler_epoch = 0

    def __warm_restart_learning_rate(self, batch_count):
        """
        calculates a learning rate according to a warm restart schedule as proposed by Ilya Loshchilov and Frank Hutter
        in SGDR: Stochastic Descent with Warm Restarts (https://arxiv.org/abs/1608.03983)
        :param batch_count: counter for batches in an epoch
        :return adjusted learning rate
        """
        curstep = self.scheduler_epoch + (1 / math.floor(self.dataset_size / float(self.batch_size))) * batch_count

        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos((curstep / float(self.wr_epochs)) * math.pi))

        self.__warm_restart_cycle_length()

        return lr
