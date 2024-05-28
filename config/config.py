import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 50
BATCH_SIZE = 5
NUM_WORKERS = 0
LEARNING_RATE = 2e-5
LMBDA = 10
ROOT_MONET = "E:\github\CycleGAN\data\monet_jpg"
ROOT_PHOTO = "E:\github\CycleGAN\data\photo_jpg"
COEF = 0.5