import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import config
from models.Discriminator import Discriminator
from models.Generator import Generator
from models.dataset import MyDataset


# CycleGAN架构
class CycleGAN():
    def __init__(self):
        self.avg_g_loss = 0
        self.avg_d_loss = 0

        self.d_p = Discriminator()
        self.d_m = Discriminator()
        self.g_ptm = Generator()
        self.g_mtp = Generator()

        self.gen_stats = AvgStats()
        self.desc_stats = AvgStats()

        self.init_models()

        self.opt_disc = optim.Adam(
            list(self.d_m.parameters()) + list(self.d_p.parameters()),
            lr=config.LEARNING_RATE,
            betas=(0.5, 0.999),
        )
        self.opt_gen = optim.Adam(
            list(self.g_ptm.parameters()) + list(self.g_mtp.parameters()),
            lr=config.LEARNING_RATE,
            betas=(0.5, 0.999),
        )

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()

    def init_models(self):
        self.d_p = self.d_p.to(config.DEVICE)
        self.d_m = self.d_m.to(config.DEVICE)
        self.g_ptm = self.g_ptm.to(config.DEVICE)
        self.g_mtp = self.g_mtp.to(config.DEVICE)

    def train(self, dataset):

        for epoch in range(config.NUM_EPOCHS):

            total_g_loss = 0
            total_d_loss = 0
            m_reals = 0
            m_fakes = 0

            loop = tqdm(dataset, leave=True)

            for idx, (monet, photo) in enumerate(loop):
                monet = monet.to(config.DEVICE)
                photo = photo.to(config.DEVICE)

                # Generator loss
                with torch.cuda.amp.autocast():
                    self.opt_gen.zero_grad()

                    # identity loss
                    id_monet = self.g_ptm(monet)
                    id_photo = self.g_mtp(photo)
                    id_monet_loss = self.l1(id_monet, monet) * config.LMBDA * config.COEF
                    id_photo_loss = self.l1(id_photo, photo) * config.LMBDA * config.COEF
                    id_loss = id_photo_loss + id_monet_loss

                    fake_monet = self.g_ptm(photo)
                    d_m_fake = self.d_m(fake_monet)
                    loss_g_m = self.mse(d_m_fake, torch.ones_like(d_m_fake))

                    fake_photo = self.g_mtp(monet)
                    d_p_fake = self.d_p(fake_photo)
                    loss_g_p = self.mse(d_p_fake, torch.ones_like(d_p_fake))

                    loss_g = loss_g_m + loss_g_p

                    # cycle loss
                    cycle_monet = self.g_ptm(fake_photo)
                    cycle_photo = self.g_mtp(fake_monet)
                    cycle_photo_loss = self.l1(cycle_photo, photo)
                    cycle_monet_loss = self.l1(cycle_monet, monet)

                    g_loss = (
                            loss_g
                            + cycle_photo_loss * config.LMBDA
                            + cycle_monet_loss * config.LMBDA
                            + id_loss
                    )

                    total_g_loss += g_loss.item()

                # 反向传播
                self.g_scaler.scale(g_loss).backward()
                self.g_scaler.step(self.opt_gen)
                self.g_scaler.update()

                with torch.cuda.amp.autocast():
                    self.opt_disc.zero_grad()

                    # Discrimator loss
                    d_m_real = self.d_m(monet)
                    d_m_fake = self.d_m(fake_monet.detach())
                    m_reals += d_m_real.mean().item()
                    m_fakes += d_m_fake.mean().item()
                    d_m_real_loss = self.mse(d_m_real, torch.ones_like(d_m_real))
                    d_m_fake_loss = self.mse(d_m_fake, torch.zeros_like(d_m_fake))
                    d_m_loss = d_m_real_loss + d_m_fake_loss

                    d_p_real = self.d_p(photo)
                    d_p_fake = self.d_p(fake_photo.detach())
                    d_p_real_loss = self.mse(d_p_real, torch.ones_like(d_p_real))
                    d_p_fake_loss = self.mse(d_p_real, torch.zeros_like(d_p_fake))
                    d_p_loss = d_p_real_loss + d_p_fake_loss

                    d_loss = (d_m_loss + d_p_loss) / 2

                    total_d_loss += d_loss.item()

                self.d_scaler.scale(d_loss).backward()
                self.d_scaler.step(self.opt_disc)
                self.d_scaler.update()

                self.avg_d_loss = total_d_loss / dataset.__len__()
                self.avg_g_loss = total_g_loss / dataset.__len__()

                loop.set_postfix(m_real=m_reals / (idx + 1), m_fake=m_fakes / (idx + 1))

            # 保存weights
            save_dict = {
                'epoch': epoch + 1,
                'g_mtp': self.g_mtp.state_dict(),
                'g_ptm': self.g_ptm.state_dict(),
                'd_m': self.d_m.state_dict(),
                'd_p': self.d_p.state_dict(),
                'optimizer_gen': self.opt_gen.state_dict(),
                'optimizer_desc': self.opt_disc.state_dict()
            }

            # 保存检查点
            save_checkpoint(save_dict, 'current.ckpt')

            # 输出每个Epoch的loss
            print("Epoch: (%d) | Generator Loss:%f | Discriminator Loss:%f" % (epoch +
                                                                               1, self.avg_g_loss, self.avg_d_loss))
            # 保存至loss字典
            self.gen_stats.append(self.avg_g_loss)
            self.desc_stats.append(self.avg_d_loss)


# 保存loss
class AvgStats(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []

    def append(self, loss):
        self.losses.append(loss)


# 保存检查点
def save_checkpoint(state, save_path):
    torch.save(state, save_path)


# 对图片进行反归一化处理
def unnorm(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)

    return img


# 加载数据集
train_data = MyDataset(config.ROOT_MONET, config.ROOT_PHOTO)
loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, drop_last=True,
                    pin_memory=True)
# 初始化gan
gan = CycleGAN()

# 初始化weights
save_dict = {
    'epoch': 0,
    'g_mtp': gan.g_mtp.state_dict(),
    'g_ptm': gan.g_ptm.state_dict(),
    'd_m': gan.d_m.state_dict(),
    'd_p': gan.d_p.state_dict(),
    'optimizer_gen': gan.opt_gen.state_dict(),
    'optimizer_desc': gan.opt_disc.state_dict()
}

# 训练
gan.train(loader)

# 生成loss函数图
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.plot(gan.gen_stats.losses, 'r', label='Generator Loss')
plt.plot(gan.desc_stats.losses, 'b', label='Descriminator Loss')
plt.legend()
plt.show()

# 重新加载照片数据集，以备生成最终图片
ph = tqdm(loader, leave=True)
# %%
for i, (monet, photo) in enumerate(ph):
    with torch.no_grad():
        pred_monet = gan.g_ptm(photo.to("cuda")).cpu().detach()
    pred_monet = unnorm(pred_monet)
    trans = transforms.ToPILImage()
    img = trans(pred_monet[0]).convert("RGB")
    img.save("./images/" + str(i + 1) + ".jpg")
