import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import itertools
import tqdm
import torch
import numpy as np
import torch.nn.init as init

from config import config
from models import Generator, Discriminator
from models.dataset import MyDataset


def update_req_grad(models, requires_grad=True):
    for model in models:
        for param in model.parameters():
            param.requires_grad = requires_grad


def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def save_checkpoint(state, save_path):
    torch.save(state, save_path)


class sample_fake(object):
    def __init__(self, max_imgs=50):
        self.max_imgs = max_imgs
        self.cur_img = 0
        self.imgs = list()

    def __call__(self, imgs):
        ret = list()
        for img in imgs:
            if self.cur_img < self.max_imgs:
                self.imgs.append(img)
                ret.append(img)
                self.cur_img += 1
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_imgs)
                    ret.append(self.imgs[idx])
                    self.imgs[idx] = img
                else:
                    ret.append(img)
        return ret


def train(generator, discriminator):
    g_mtp = Generator().to(config.DEVICE)
    g_ptm = Generator().to(config.DEVICE)
    d_m = Discriminator().to(config.DEVICE)
    d_p = Discriminator().to(config.DEVICE)

    g_optimizer = optim.Adam(itertools.chain(g_mtp.parameters(), g_ptm.parameters()),
                             lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    d_p_optimizer = optim.Adam(d_p.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    d_m_optimizer = optim.Adam(d_m.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(itertools.chian(d_p.parameters(),d_m.parameters(),
                             lr=config.LEARNING_RATE), betas=(0.5, 0.999))

    sample_monet = sample_fake()
    sample_photo = sample_fake()-
    gen_lr = lr_sched(config.DECAY_EPOCH, epochs)
    desc_lr = lr_sched(config.DECAY_EPOCH, epochs)
    gen_lr_sched = torch.optim.lr_scheduler.LambdaLR(g_optimizer, gen_lr.step)
    desc_lr_sched = torch.optim.lr_scheduler.LambdaLR(d_optimizer, desc_lr.step)
    gen_stats = AvgStats()
    desc_stats = AvgStats()

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    train_transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ]

    train_transform = transforms.Compose(train_transform)
    train_data = MyDataset(train_transform)

    loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    loop = tqdm(loader, leave=True)

    for epoch in range(config.EPOCHS):

        avg_gen_loss = 0.0
        avg_desc_loss = 0.0

        for images in loop:
            monet = images['A'].to(config.DEVICE)
            photo = images['B'].to(config.DEVICE)

            fake_photo = g_mtp(monet)
            fake_monet = g_ptm(photo)

            cycle_monet = g_ptm(fake_photo)
            cycle_photo = g_mtp(fake_monet)

            id_monet = g_ptm(monet)
            id_photo = g_mtp(photo)

            id_loss_monet = l1(id_monet, monet) * config.LMBDA * config.COEF
            id_loss_photo = l1(id_photo, photo) * config.LMBDA * config.COEF

            cycle_loss_monet = l1(cycle_monet, monet) * config.LMBDA
            cycle_loss_photo = l1(cycle_photo, photo) * config.LMBDA

            monet_desc = d_m(fake_monet)
            photo_desc = d_p(fake_photo)

            real = torch.ones(monet_desc.size()).to(config.DEVICE)

            adv_loss_monet = mse(monet_desc, real)
            adv_loss_photo = mse(photo_desc, real)

            total_gen_loss = (cycle_loss_monet + adv_loss_monet + cycle_loss_photo + adv_loss_photo + id_loss_monet +
                              id_loss_photo)

            avg_gen_loss += total_gen_loss.item()

            total_gen_loss.backward()
            g_optimizer.step()

            update_req_grad([d_m, d_p], True)
            d_m_optimizer.zero_grad()

            fake_monet = sample_monet([fake_monet.cpu().data.numpy()])[0]
            fake_photo = sample_photo([fake_photo.cpu().data.numpy()])[0]
            fake_monet = torch.tensor(fake_monet).to(config.DEVICE)
            fake_photo = torch.tensor(fake_photo).to(config.DEVICE)

            monet_desc_real = d_m(monet)
            monet_desc_fake = d_m(fake_monet)
            photo_desc_real = d_p(photo)
            photo_desc_fake = d_p(fake_photo)

            real = torch.ones(monet_desc_real.size()).to(config.DEVICE)
            fake = torch.zeros(monet_desc_fake.size()).to(config.DEVICE)

            # Descriminator losses
            # --------------------
            monet_desc_real_loss = mse(monet_desc_real, real)
            monet_desc_fake_loss = mse(monet_desc_fake, fake)
            photo_desc_real_loss = mse(photo_desc_real, real)
            photo_desc_fake_loss = mse(photo_desc_fake, fake)

            monet_desc_loss = (monet_desc_real_loss + monet_desc_fake_loss) / 2
            photo_desc_loss = (photo_desc_real_loss + photo_desc_fake_loss) / 2
            total_desc_loss = monet_desc_loss + photo_desc_loss
            avg_desc_loss += total_desc_loss.item()

            # Backward
            monet_desc_loss.backward()
            photo_desc_loss.backward()
            self.adam_desc.step()

            t.set_postfix(gen_loss=total_gen_loss.item(), desc_loss=total_desc_loss.item())

        save_dict = {
            'epoch': epoch + 1,
            'gen_mtp': gan.gen_mtp.state_dict(),
            'gen_ptm': gan.gen_ptm.state_dict(),
            'desc_m': gan.desc_m.state_dict(),
            'desc_p': gan.desc_p.state_dict(),
            'optimizer_gen': gan.adam_gen.state_dict(),
            'optimizer_desc': gan.adam_desc.state_dict()
        }
        save_checkpoint(save_dict, 'current.ckpt')

        avg_gen_loss /= photo_dl.__len__()
        avg_desc_loss /= photo_dl.__len__()
        time_req = time.time() - start_time

        gen_stats.append(avg_gen_loss, time_req)
        desc_stats.append(avg_desc_loss, time_req)

        print("Epoch: (%d) | Generator Loss:%f | Discriminator Loss:%f" %
              (epoch + 1, avg_gen_loss, avg_desc_loss))

        gen_lr_sched.step()
        desc_lr_sched.step()


class lr_sched():
    def __init__(self, config.DECAY_EPOCHs=100, total_epochs=200):
        self.config.DECAY_EPOCHs = config.DECAY_EPOCHs
        self.total_epochs = total_epochs

    def step(self, epoch_num):
        if epoch_num <= self.config.DECAY_EPOCHs:
            return 1.0
        else:
            fract = (epoch_num - self.config.DECAY_EPOCHs)  / (self.total_epochs - self.config.DECAY_EPOCHs)
            return 1.0 - fract


class AvgStats(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.its = []

    def append(self, loss, it):
        self.losses.append(loss)
        self.its.append(it)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)