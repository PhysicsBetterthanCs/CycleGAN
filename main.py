from models import Generator, Discriminator
import torch
import torch.nn as nn
import torch.optim as optim
from config import config
import itertools


def train(generator, discriminator):

    g_mtp = Generator().to(config.DEVICE)
    g_ptm = Generator().to(config.DEVICE)
    d_mtp = Discriminator().to(config.DEVICE)
    d_ptm = Discriminator().to(config.DEVICE)

    g_optimizer = optim.Adam(itertools.chain(g_mtp.parameters(),g_ptm.parameters()),
                             lr=config.LEARNING_RATE,betas=(0.5, 0.999))
    d_ptm_optimizer = optim.Adam(d_ptm.parameters(),lr=config.LEARNING_RATE,betas=(0.5, 0.999))
    d_mtp_optimizer = optim.Adam(d_mtp.parameters(),lr=config.LEARNING_RATE,betas=(0.5, 0.999))

    l1 = nn.L1Loss()
    mse = nn.MSELoss()




