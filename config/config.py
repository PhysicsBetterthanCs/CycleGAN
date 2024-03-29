import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 1
NUM_WORKERS = 4
LEARNING_RATE = 2e-5
LMBDA = 10
COEFFICIENT = 0.5
DECAY_EPOCH = 0
MONET_DATA_PATH = "./data/photo_jpg/*jpg"
PHOTOS_DATA_PATH = "./data/monet_jpg/*jpg"
