import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    @classmethod
    def from_args(cls, args):
        return cls(args.log_dir, args.purge_step, args.log_filename_suffix)

    def __init__(self, log_dir, purge_step, filename_suffix):
        self.writer = SummaryWriter(log_dir=log_dir, purge_step=purge_step, filename_suffix=filename_suffix)
        self.reset_batch_state()

    def reset_batch_state(self):
        self.batch_state = {}

    def log_step(self, metrics):
        for key in metrics.keys():
            if key not in self.batch_state.keys():
                self.batch_state[key] = []

            self.batch_state[key].append(metrics[key])

    def log_epoch(self, epoch, prefix):
        epoch_metrics = {}
        for key in self.batch_state.keys():
            epoch_metrics[key] = np.mean(self.batch_state[key])

        for key in self.batch_state.keys():
            self.writer.add_scalar("{}/{}".format(prefix, key), epoch_metrics[key], global_step=epoch)
        self.reset_batch_state()

        return epoch_metrics

    def flush(self):
        self.writer.flush()
