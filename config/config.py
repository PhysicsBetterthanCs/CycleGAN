import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 1
WORKERS = 4
LEARNING_RATE = 2e-5
MONET_DATA_PATH = "./data/photo_jpg/*jpg"
PHOTOS_DATA_PATH = "./data/monet_jpg/*jpg"
