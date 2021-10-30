import torchvision
import torch
from torch.utils.tensorboard import SummaryWriter
from model.cyclegan import Generator, Discriminator
from loader import ImageDataset
from tqdm import tqdm
import random
import numpy as np

num_epochs = 200
batch_size = 1
lr = 0.0002
momentum = 0.9
# Load dataset
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Dataset: https://www.kaggle.com/c/gan-getting-started/data
monet_path_train = "dataset/train/monet/"
photo_path_train = "dataset/train/photos/"
photo_path_test = "dataset/test/photos/"
monet_dataset = ImageDataset(monet_path_train)
photo_dataset = ImageDataset(photo_path_train)
photo_dataset_test = ImageDataset(photo_path_test)
monet_dataloader = torch.utils.data.DataLoader(monet_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1
                                               )
photo_dataloader = torch.utils.data.DataLoader(photo_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1
                                               )
# Optimizer


# Load model

F = Generator().to(device) # photo generator
G = Generator().to(device) # monet generator
D_monet = Discriminator().to(device) # 1 real, 0 fake
D_photo = Discriminator().to(device)

opt_F = torch.optim.SGD(lr=lr, params=F.parameters(), momentum=momentum)
opt_G = torch.optim.SGD(lr=lr, params=G.parameters(), momentum=momentum)
opt_Dm = torch.optim.SGD(lr=lr, params=D_monet.parameters(), momentum=momentum)
opt_Dp = torch.optim.SGD(lr=lr, params=D_photo.parameters(), momentum=momentum)


def train_one_epoch(epoch, F, G, D_photo, D_monet, photo_dl, monet_dl):
    n= 100
    gan_loss = torch.nn.BCEWithLogitsLoss()
    #gan_loss = torch.nn.MSELoss()
    cycle_loss = torch.nn.L1Loss()
    for i in range(n):
        photo1 = random.choice(photo_dl)
        monet1 = random.choice(monet_dl)

        #photo2 = random.choice(photo_dl)
        #monet2 = random.choice(monet_dl)

        photo = torch.stack([photo1])
        monet = torch.stack([monet1])
        photo = photo.to(device)
        monet = monet.to(device)

        opt_F.zero_grad()
        opt_G.zero_grad()
        opt_Dm.zero_grad()
        opt_Dp.zero_grad()

        # Choose random photo and painting


        # generate fakes
        fake_monet = G(photo)
        fake_photo = F(monet)
        # generate cycles
        reconstructed_photo = F(fake_monet)
        reconstructed_monet = G(fake_photo)

        # Cycle loss
        cycleloss_FG = cycle_loss(photo, reconstructed_photo)
        cycleloss_GF = cycle_loss(monet, reconstructed_monet)

        # GAN loss generator
        gloss_monet_fake = gan_loss(D_monet(fake_monet), torch.FloatTensor([[1]]).to(device))
        floss_photo_fake = gan_loss(D_photo(fake_photo), torch.FloatTensor([[1]]).to(device))

        gan_generator_losses = gloss_monet_fake.sum() + floss_photo_fake.sum()

        opt_Dm.zero_grad()
        opt_Dp.zero_grad()

        print(D_photo(photo))
        # GAN loss discriminator
        # D_photo vs F
        dloss_photo_real = gan_loss(D_photo(photo), torch.FloatTensor([[1]]).to(device))
        dloss_photo_fake = gan_loss(D_photo(fake_photo.detach()), torch.FloatTensor([[0]]).to(device))
        # D_monet vs G
        dloss_monet_real = gan_loss(D_monet(monet), torch.FloatTensor([[1]]).to(device))
        dloss_monet_fake = gan_loss(D_monet(fake_monet.detach()), torch.FloatTensor([[0]]).to(device))

        # total loss and backpropagation
        l = 10
        dloss_photo = (dloss_photo_real + dloss_photo_fake).sum()
        dloss_monet = (dloss_monet_real + dloss_monet_fake).sum()
        cycle_losses = (cycleloss_FG + cycleloss_GF).sum()
        total_loss = l * cycle_losses + dloss_photo + dloss_monet + gan_generator_losses
        total_loss.backward()


        opt_Dm.step()
        opt_Dp.step()
        opt_F.step()
        opt_G.step()


        print("total loss is ", total_loss.item())
        #print("cycle loss is ", cycle_losses.item())
        #print("d photo loss is ", dloss_photo.item())
        #print("d monet loss is ", dloss_monet.item())




        #Tensorboard
        writer.add_scalar("cycle loss",
                          cycle_losses.item(),
                          epoch * n + i)
        writer.add_scalar("d_photo loss",
                          dloss_photo.item(),
                          epoch * n + i)
        writer.add_scalar("d_monet loss",
                          dloss_monet.item(),
                          epoch * n + i)
        writer.add_scalar("gan F loss",
                           floss_photo_fake.item(),
                           epoch * n + i)
        writer.add_scalar("gan G loss",
                           gloss_monet_fake.item(),
                           epoch * n + i)
        writer.add_scalar("total loss",
                          total_loss.item(),
                          epoch * n + i)



        if i%3 == 0:
            # create grid of images
            monet_grid = torchvision.utils.make_grid(monet.cpu())
            photo_grid = torchvision.utils.make_grid(photo.cpu())
            fake_monet_grid = torchvision.utils.make_grid(fake_monet.cpu())
            fake_photo_grid = torchvision.utils.make_grid(fake_photo.cpu())
            reconstructed_photo_grid = torchvision.utils.make_grid(reconstructed_photo.cpu())
            reconstructed_monet_grid = torchvision.utils.make_grid(reconstructed_monet.cpu())

            # write to tensorboard
            writer.add_image("photo", photo_grid)
            writer.add_image('fake_monet', fake_monet_grid)
            writer.add_image("monet", monet_grid)
            writer.add_image('fake_photo', fake_photo_grid)
            writer.add_image('reconstructed_photo', reconstructed_photo_grid)
            writer.add_image('reconstructed_monet', reconstructed_monet_grid)







photo_sampler = torch.utils.data.RandomSampler(photo_dataset, replacement=False)
monet_sampler = torch.utils.data.RandomSampler(monet_dataset, replacement=False)
#Start tensor board
import datetime
import os

# Create directories for logs
log_dir = "runs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_dir = log_dir + "/summary"
models_dir = log_dir + "/models"
test_dir = log_dir + "/test"
os.mkdir(log_dir)
os.mkdir(summary_dir)
os.mkdir(models_dir)
os.mkdir(test_dir)

# Start tensorboard
#type "tensorboard --logdir=runs" in terminal
writer = SummaryWriter(summary_dir)

def test_one_epoch(F, epoch):
    im_ten = photo_dataset_test[0].unsqueeze(0).to(device)
    with torch.no_grad():
        new_im = F(im_ten)
        torchvision.utils.save_image(new_im.cpu(), f"{test_dir}/epoch{epoch:03d}.png")

for epoch in tqdm(range(num_epochs)):
    print("Epoch:", epoch)
    torch.save(F.state_dict(), models_dir + "/F.pth")
    torch.save(G.state_dict(), models_dir + "/G.pth")
    torch.save(D_monet.state_dict(), models_dir + "/D_monet.pth")
    torch.save(D_photo.state_dict(), models_dir + "/D_photo.pth")
    train_one_epoch(epoch, F, G, D_photo, D_monet, photo_dataset, monet_dataset)
    test_one_epoch(F, epoch)

