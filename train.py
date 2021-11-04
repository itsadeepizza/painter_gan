import torchvision
import torch
from torch.utils.tensorboard import SummaryWriter
from model.cyclegan import Generator, Discriminator
from loader import ImageDataset
from tqdm import tqdm
import random
import datetime
import os
import numpy as np


def load_models(path=None, epoch=None):
    """Create models and load weights from a path if specified"""
    G_photo = Generator().to(device)  # photo generator
    G_monet= Generator().to(device)  # monet generator
    D_monet = Discriminator().to(device)  # 1=real, 0=fake
    D_photo = Discriminator().to(device)
    if path is not None:
        G_photo.load_state_dict(torch.load(path + f"/G_photo_Ep{epoch:03d}.pth"))
        G_monet.load_state_dict(torch.load(path+  f"/G_monet_Ep{epoch:03d}.pth"))
        D_monet.load_state_dict(torch.load(path + f"/D_monet_Ep{epoch:03d}.pth"))
        D_photo.load_state_dict(torch.load(path + f"/D_photo_Ep{epoch:03d}.pth"))
        G_photo.eval()
        G_monet .eval()
        D_monet.eval()
        D_photo.eval()
    return G_photo, G_monet , D_monet, D_photo


def train_discriminators(D_photo, D_monet):
    dloss_photo = torch.zeros(1).to(device)
    dloss_monet = torch.zeros(1).to(device)
    for real_photo, fake_photo, real_monet, fake_monet in zip(real_photos, fake_photos, real_monets, fake_monets):
        # D_photo vs G_photo
        dloss_photo_real = gan_loss(D_photo(real_photo), 1)
        dloss_photo_fake = gan_loss(D_photo(fake_photo), 0)
        # D_monet vs G
        dloss_monet_real = gan_loss(D_monet(real_monet), 1)
        dloss_monet_fake = gan_loss(D_monet(fake_monet), 0)


        dloss_photo = (dloss_photo_real + dloss_photo_fake).sum()
        dloss_monet = (dloss_monet_real + dloss_monet_fake).sum()
        dloss_total = (dloss_photo + dloss_monet)/10

        # Update backpropagation
        dloss_total.backward()
        opt_Dm.step()
        opt_Dp.step()

        # Drop discriminant gradient for this loss
        opt_Dm.zero_grad()
        opt_Dp.zero_grad()
    return dloss_monet, dloss_photo


def train_one_epoch(epoch, G_photo, G_monet , D_photo, D_monet, photo_dl, monet_dl):
    n = min(len(photo_dl), len(monet_dl))

    # Create iterator
    monet_iterator = iter(monet_dl)
    photo_iterator = iter(photo_dl)
    for i in range(n):
        # iterate through the dataloader
        photo = next(photo_iterator)
        monet = next(monet_iterator)
        # Load to GPU
        photo = photo.to(device)
        monet = monet.to(device)
        # Drop gradients
        opt_G_photo.zero_grad()
        opt_G_monet.zero_grad()
        opt_Dm.zero_grad()
        opt_Dp.zero_grad()

        #=============================
        # IMAGE GENERATION
        #=============================
        # generate fakes
        fake_monet = G_monet (photo)
        fake_photo = G_photo(monet)
        # generate cycles
        reconstructed_photo = G_photo(fake_monet)
        reconstructed_monet = G_monet(fake_photo)




        #========================
        # LOSS GENERATOR
        #========================

        # Cycle loss
        cycleloss_G_photoG_monet= cycle_loss(photo, reconstructed_photo)
        cycleloss_G_monetG_photo = cycle_loss(monet, reconstructed_monet)
        cycle_losses = (cycleloss_G_photoG_monet + cycleloss_G_monetG_photo).sum()

        # no need of gradients for discriminant
        D_monet.requires_grad = False
        D_photo.requires_grad = False

        # GAN loss generator
        gloss_monet_fake = gan_loss(D_monet(fake_monet), 1)
        floss_photo_fake = gan_loss(D_photo(fake_photo), 1)
        gan_generator_losses = gloss_monet_fake.sum() + floss_photo_fake.sum()

        generator_loss = l * cycle_losses + gan_generator_losses
        generator_loss.backward()

        # Now we need gradients for discriminant
        D_monet.requires_grad = True
        D_photo.requires_grad = True

        # Update backpropagation for generators
        opt_G_photo.step()
        opt_G_monet.step()


        # =========================
        # GAN loss discriminator
        # ========================
        if len(real_photos) >= 5:
            del real_photos[0]
            del fake_photos[0]
            del real_monets[0]
            del fake_monets[0]
        real_photos.append(photo)
        fake_photos.append(fake_photo.detach())
        real_monets.append(monet)
        fake_monets.append(fake_monet.detach())
        opt_Dm.zero_grad()
        opt_Dp.zero_grad()
        dloss_monet, dloss_photo = train_discriminators(D_photo, D_monet)

        total_loss = generator_loss.item() +  dloss_monet.item()+  dloss_photo.item()
        print("total loss is ", total_loss)
        #print("cycle loss is ", cycle_losses.item())
        #print("d photo loss is ", dloss_photo.item())
        #print("d monet loss is ", dloss_monet.item())


        #Upload losses to Tensorboard
        writer.add_scalar("cycle loss",
                          cycle_losses.item(),
                          epoch * n + i)
        writer.add_scalar("d_photo loss",
                          dloss_photo.item(),
                          epoch * n + i)
        writer.add_scalar("d_monet loss",
                          dloss_monet.item(),
                          epoch * n + i)
        writer.add_scalar("gan G_photo loss",
                           floss_photo_fake.item(),
                           epoch * n + i)
        writer.add_scalar("gan G_monetloss",
                           gloss_monet_fake.item(),
                           epoch * n + i)
        writer.add_scalar("total loss",
                          total_loss,
                          epoch * n + i)


        # Upload images to tensorboard
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
        # Free memory
        torch.cuda.empty_cache()

def test_one_epoch(G_monet, G_photo, epoch):
    photo = photo_dataset_test[0].unsqueeze(0).to(device)
    monet = monet_dataset_test[0].unsqueeze(0).to(device)
    with torch.no_grad():
        fake_monet = G_monet(photo)
        torchvision.utils.save_image(fake_monet.cpu(), f"{test_dir}/fake_monet-epoch{epoch:03d}.png")
        fake_photo = G_photo(monet)
        torchvision.utils.save_image(fake_photo.cpu(), f"{test_dir}/fake_photo-epoch{epoch:03d}.png")

def save_models(epoch, G_photo, G_monet , D_photo, D_monet):
    torch.save(G_photo.state_dict(), f"{models_dir}/G_photo_Ep{epoch:03d}.pth")
    torch.save(G_monet.state_dict(), f"{models_dir}/G_monet_Ep{epoch:03d}.pth")
    torch.save(D_monet.state_dict(), f"{models_dir}/D_monet_Ep{epoch:03d}.pth")
    torch.save(D_photo.state_dict(), f"{models_dir}/D_photo_Ep{epoch:03d}.pth")

#=============================
# CHOICE OF HYPERPARAMETERS
#=============================
num_epochs = 400
batch_size = 1
lr = 0.0001 #0.0002
momentum = 0.9
l = 10 # ratio CYCLE loss / GAN LOSS

# LOSSES

def gan_loss(test, label):
    import torch.nn.functional as F
    if label==1:
        label = torch.ones(test.shape, device=device)
    else:
        label = torch.zeros(test.shape, device=device)
    loss = F.binary_cross_entropy_with_logits(test, label)
    return loss



#gan_loss = torch.nn.MSELoss()
cycle_loss = torch.nn.L1Loss()


# Choose device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Dataset: https://www.kaggle.com/c/gan-getting-started/data
# Load dataset
monet_path_train = "dataset/train/monet/"
photo_path_train = "dataset/train/photos/"
monet_path_test = "dataset/test/monet/"
photo_path_test = "dataset/test/photos/"
monet_dataset = ImageDataset(monet_path_train)
photo_dataset = ImageDataset(photo_path_train)
monet_dataset_test = ImageDataset(monet_path_test)
photo_dataset_test = ImageDataset(photo_path_test)
monet_dataloader = torch.utils.data.DataLoader(monet_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0
                                               )
photo_dataloader = torch.utils.data.DataLoader(photo_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0
                                               )

# Load model (if path is None create a new model
G_photo, G_monet, D_monet, D_photo = load_models(path="runs/fit/20211101-071822/models", epoch=0)


# Optimizer
opt_G_photo = torch.optim.Adam(lr=lr, params=G_photo.parameters())
opt_G_monet = torch.optim.Adam(lr=lr, params=G_monet.parameters())
opt_Dm = torch.optim.Adam(lr=lr, params=D_monet.parameters())
opt_Dp = torch.optim.Adam(lr=lr, params=D_photo.parameters())

photo_sampler = torch.utils.data.RandomSampler(photo_dataset, replacement=False)
monet_sampler = torch.utils.data.RandomSampler(monet_dataset, replacement=False)

# Create directories for logs
now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "runs/fit/" + now_str
summary_dir = "runs/fit/" + now_str
models_dir = log_dir + "/models"
test_dir = log_dir + "/test"
os.mkdir(log_dir)
#os.mkdir(summary_dir)
os.mkdir(models_dir)
os.mkdir(test_dir)


test_one_epoch(G_monet, G_photo,  0)

# Start tensorboard
#type "tensorboard --logdir=runs" in terminal
#type "tensorboard dev upload --logdir=runs/fit/NAME_DIRECTORY" for upload the log (no images)

writer = SummaryWriter(summary_dir)


real_photos = []
fake_photos = []
real_monets = []
fake_monets = []


for epoch in tqdm(range(1, num_epochs)):
    print("Epoch:", epoch)
    if epoch%5==0:
        save_models(epoch, G_photo, G_monet , D_photo, D_monet)
    train_one_epoch(epoch, G_photo, G_monet , D_photo, D_monet, photo_dataloader, monet_dataloader)
    test_one_epoch(G_monet, G_photo,  epoch)

