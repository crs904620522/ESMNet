import torch
import torch.distributions as dist
from torch import nn
import os
from lf2disp.ESMNet import models, training, generation
from lf2disp.ESMNet.datafield.HCInew_dataloader import HCInew
Datadict = {
    'HCInew':HCInew,
}

def get_model(cfg, dataset=None, device=None):
    model = models.ESMNet(cfg, device=device)
    return model


def get_dataset(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    type = cfg['data']['dataset']
    dataset = Datadict[type](cfg, mode=mode)
    return dataset


def get_trainer(model, optimizer, cfg, criterion, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''

    trainer = training.Trainer(
        model, optimizer,
        device=device,
        criterion=criterion,
        cfg=cfg,
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''

    generator = generation.GeneratorDepth(
        model,
        device=device,
        cfg=cfg,
    )
    return generator
