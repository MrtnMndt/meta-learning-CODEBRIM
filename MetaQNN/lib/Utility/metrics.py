class AverageMeter(object):
    """
    computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
        self.val = self.avg = self.sum = self.count = 0

    def reset(self):
        """
        resets all metric attributes
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        updates each of the metric attributes
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
