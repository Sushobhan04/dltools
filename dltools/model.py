import os
import logging

import torch
import torch.nn as nn
from . import utils

def load_model(filepath, updated_cfg=None, device=None, verbose=True):
    """Load a model of model_class using the data in filepath

    Args:
        filepath (string): complete path for file with model data
        model_class (nn.Module): module inherited from nn.Module for the model
        cfg (dict, Optional): config file to override the saved configuration. Default to None
        verbose (bool, Optional): A flag to print message when model loaded. Defaults to True
    """
    model_dict = torch.load(filepath)
    cfg = utils.Config()
    cfg.__dict__ = model_dict["cfg"]

    if updated_cfg is not None:
        cfg.__dict__.update(updated_cfg.__dict__)

    if device is not None:
        cfg.device = device

    model = utils.name_to_class(model_dict["class"], cfg.model_module)(cfg)
    model.load(model_dict, cfg.device)

    logging.info(f"loaded model from {filepath}")

    if verbose:
        print(f"loaded model from {filepath}")

    return model


class BaseModel(nn.Module):
    """A base class for all models.
    Contains methods:
        save: save a model
        load: load a model
        init_optimizer: initialize Adam optimizer
    """
    def __init__(self, name="test0", description=""):
        super().__init__()

        self.name = name
        self.description = description
        self.epochs = 0
        self.optimizer = None
        self.best_metric = 0.0

    def save(self, filepath):
        path, filename = filepath.rsplit("/", 1)

        if not os.path.exists(path):
            os.makedirs(path)

        model_dict = {
            "class": self.__class__.__name__,
            "name": self.name,
            "description": self.description,
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epochs": self.epochs,
            "best_metric": self.best_metric,
            "cfg": self.cfg.__dict__
        }

        torch.save(model_dict, filepath)

    def load(self, model_dict, device="cpu", verbose=True):
        self.load_state_dict(model_dict["model"])
        self.to(device)

        if "optimizer" in model_dict.keys():
            self.init_optimizer()
            self.optimizer.load_state_dict(model_dict["optimizer"])
        if "best_metric" in model_dict.keys():
            self.best_metric = model_dict["best_metric"]
        else:
            self.best_metric = 0.0

        self.epochs = model_dict["epochs"]
        self.name = model_dict["name"]

    def init_optimizer(self, lr=1e-4, betas=(0.9, 0.999)):
        parameters = [params for params in self.parameters() if params.requires_grad]
        self.optimizer = torch.optim.Adam(parameters, lr=lr, betas=betas)
