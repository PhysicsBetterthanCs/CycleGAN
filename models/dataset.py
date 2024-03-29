import glob
from config import  config
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, transform, mode='train'):
        self.files_A = glob.glob(config.MONET_DATA_PATH)
        self.files_B = glob.glob(config.PHOTOS_DATA_PATH)
        self.transform = transform
        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)

    def __len__(self):
        return max(self.len_A, self.len_B)

    def __getitem__(self, idx):
        i = idx % self.len_A
        j = idx % self.len_B

        img_A = Image.open(self.files_A[i])
        img_B = Image.open(self.files_B[j])

        return {"A": self.transform(img_A), "B": self.transform(img_B)}