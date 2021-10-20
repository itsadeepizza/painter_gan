import torchvision
import torch
from model.cyclegan import Generator, Discriminator
from loader import ImageDataset
from tqdm import tqdm
import random
import numpy as np

num_epochs = 100
batch_size = 1
lr = 0.0002
momentum = 0.9
# Load dataset
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Dataset: https://www.kaggle.com/c/gan-getting-started/data
monet_path_train = "dataset/train/monet/"
photo_path_train = "dataset/train/photos/"
monet_dataset = ImageDataset(monet_path_train)
photo_dataset = ImageDataset(photo_path_train)
monet_dataloader = torch.utils.data.Dataloader(monet_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1
                                               )
photo_dataloader = torch.utils.data.Dataloader(photo_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1
                                               )
# Optimizer
optimizer = torch.optim.SGD(lr=lr, momentum=momentum)

# Load model
# TODO
F = Generator() # photo generator
G = Generator() # monet generator
D_monet = Discriminator() # 1 real, 0 fake
D_photo = Discriminator()



def gan_loss(guess, correct):
    """loss L2"""
    return (guess - correct)**2

def cycle_loss(original, reconstructed):
    """Loss L1"""
    diff = original - reconstructed
    loss = torch.abs(diff).sum()
    return loss

def train_one_epoch(epoch, optimizer, F, G, D_photo, D_monet, photo_dl, monet_dl):
    n= 200
    for i in range(n):
        optimizer.zero_grad()
        # Choose random photo and painting
        photo = random.choice(photo_dl)
        monet = random.choice(monet_dl)

        # generate fakes
        fake_monet = G(photo)
        fake_photo = F(monet)
        # generate cycles
        reconstructed_photo = F(fake_monet)
        reconstructed_monet = G(fake_photo)

        # Cycle loss
        cycleloss_FG = cycle_loss(reconstructed_photo, photo)
        cycleloss_GF = cycle_loss(reconstructed_monet, monet)

        # GAN loss
        # D_photo vs F
        dloss_photo_real = gan_loss(D_photo(photo), 1)
        dloss_photo_fake = gan_loss(D_photo(fake_photo), 0)
        # D_monet vs G
        dloss_monet_real = gan_loss(D_monet(monet), 1)
        dloss_monet_fake = gan_loss(D_monet(fake_monet), 0)

        # total loss and backpropagation
        l = 10
        total_loss = l * (cycleloss_FG + cycleloss_GF) + dloss_photo_real + dloss_photo_fake + dloss_monet_real + dloss_monet_fake

        total_loss.backward()
        optimizer.step()

        # Drop gradients
        # TODO


for epoch in tqdm(range(num_epochs)):
    train_one_epoch(epoch, F, G, D_photo, D_monet, photo_dataloader, monet_dataloader)

