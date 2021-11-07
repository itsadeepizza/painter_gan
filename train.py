import torchvision
import torch
from torch.utils.tensorboard import SummaryWriter
from model.cyclegan import Generator, Discriminator
from loader import ImageDataset
from tqdm import tqdm
import random
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np


def gan_loss(test, label):
    import torch.nn.functional as F
    if label==1:
        label = torch.ones(test.shape, device=device)
    else:
        label = torch.zeros(test.shape, device=device)
    loss = F.binary_cross_entropy_with_logits(test, label)
    return loss


cycle_loss = torch.nn.L1Loss()


def enable_grad(models, enable):
    for model in models:
        for param in model.parameters():
            param.requires_grad = enable

class Trainer:
    def __init__(self, summary_dir, path=None, epoch=None):
        """Create models and load weights from a path if specified"""
        self.G_photo = Generator().to(device)  # photo generator
        self.G_monet = Generator().to(device)  # monet generator
        self.D_monet = Discriminator().to(device)  # 1=real, 0=fake
        self.D_photo = Discriminator().to(device)
        if path is not None:
            self.G_photo.load_state_dict(torch.load(path + f"/G_photo_Ep{epoch:03d}.pth"))
            self.G_monet.load_state_dict(torch.load(path + f"/G_monet_Ep{epoch:03d}.pth"))
            self.D_monet.load_state_dict(torch.load(path + f"/D_monet_Ep{epoch:03d}.pth"))
            self.D_photo.load_state_dict(torch.load(path + f"/D_photo_Ep{epoch:03d}.pth"))
            self.G_photo.eval()
            self.G_monet.eval()
            self.D_monet.eval()
            self.D_photo.eval()

        # Optimizer
        self.opt_G_photo = torch.optim.Adam(lr=lr, params=self.G_photo.parameters())
        self.opt_G_monet = torch.optim.Adam(lr=lr, params=self.G_monet.parameters())
        self.opt_Dm = torch.optim.Adam(lr=lr, params=self.D_monet.parameters())
        self.opt_Dp = torch.optim.Adam(lr=lr, params=self.D_photo.parameters())

        self.writer = SummaryWriter(summary_dir)

    def train_discriminators(self, epoch, iter, real_photo, fake_photo, real_monet, fake_monet):
        enable_grad([self.G_photo, self.G_monet], False)
        enable_grad([self.D_photo, self.D_monet], True)

        # D_photo vs G_photo
        self.opt_Dp.zero_grad()
        dloss_photo_real = gan_loss(self.D_photo(real_photo), 1)
        dloss_photo_fake = gan_loss(self.D_photo(fake_photo.detach()), 0)
        dloss_photo = (dloss_photo_real + dloss_photo_fake).sum()
        dloss_photo.backward()
        self.opt_Dp.step()

        # D_monet vs G_monet
        self.opt_Dm.zero_grad()
        dloss_monet_real = gan_loss(self.D_monet(real_monet), 1)
        dloss_monet_fake = gan_loss(self.D_monet(fake_monet.detach()), 0)

        dloss_monet = (dloss_monet_real + dloss_monet_fake).sum()
        dloss_monet.backward()

        self.opt_Dm.step()

        self.writer.add_scalar("d_photo loss",
                          dloss_photo.item(),
                          iter)

        self.writer.add_scalar("d_monet loss",
                          dloss_monet.item(),
                          iter)

    def train_generators(self, epoch, iter, real_photo, fake_photo, real_monet, fake_monet):
        enable_grad([self.G_photo, self.G_monet], True)
        enable_grad([self.D_photo, self.D_monet], False)
        # generate cycles
        reconstructed_photo = self.G_photo(fake_monet)
        reconstructed_monet = self.G_monet(fake_photo)

        self.opt_G_photo.zero_grad()
        self.opt_G_monet.zero_grad()
        # Cycle loss
        cycleloss_G_photoG_monet = cycle_loss(real_photo, reconstructed_photo)
        cycleloss_G_monetG_photo = cycle_loss(real_monet, reconstructed_monet)
        cycle_losses = (cycleloss_G_photoG_monet + cycleloss_G_monetG_photo).sum()

        # GAN loss generator

        gloss_monet_fake = gan_loss(self.D_monet(fake_monet), 1)
        floss_photo_fake = gan_loss(self.D_photo(fake_photo), 1)
        gan_generator_losses = gloss_monet_fake.sum() + floss_photo_fake.sum()

        generator_loss = l * cycle_losses + gan_generator_losses
        generator_loss.backward()

        # Update backpropagation for generators
        self.opt_G_photo.step()
        self.opt_G_monet.step()

        self.writer.add_scalar("cycle loss",
                          cycle_losses.item(),
                          iter)
        self.writer.add_scalar("gan G_photo loss",
                          floss_photo_fake.item(),
                          iter)
        self.writer.add_scalar("gan G_monetloss",
                          gloss_monet_fake.item(),
                          iter)

    def train_one_epoch(self, epoch, photo_dl, monet_dl):
        self.G_photo.train()
        self.G_monet.train()
        self.D_photo.train()
        self.D_monet.train()

        n = min(len(photo_dl), len(monet_dl))

        # Create iterator
        monet_iterator = iter(monet_dl)
        photo_iterator = iter(photo_dl)
        for i in tqdm(range(n)):
            # iterate through the dataloader
            photo = next(photo_iterator)
            monet = next(monet_iterator)
            # Load to GPU
            photo = photo.to(device)
            monet = monet.to(device)

            #=============================
            # IMAGE GENERATION
            #=============================
            # generate fakes
            fake_monet = self.G_monet(photo)
            fake_photo = self.G_photo(monet)

            self.train_discriminators(epoch, epoch * n + i, photo, fake_photo, monet, fake_monet)
            self.train_generators(epoch, epoch * n + i, photo, fake_photo, monet, fake_monet)

            #Upload losses to Tensorboard
            '''
            
            
            
            
    
    
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
            '''

    def test_one_epoch(self, epoch):
        self.G_photo.eval()
        self.G_monet.eval()
        self.D_photo.eval()
        self.G_monet.eval()

        photo = photo_dataset_test[0].unsqueeze(0).to(device)
        monet = monet_dataset_test[0].unsqueeze(0).to(device)
        with torch.no_grad():
            fake_monet = self.G_monet(photo)
            fake_photo = self.G_photo(monet)
            reconstructed_photo = self.G_photo(fake_monet)
            reconstructed_monet = self.G_monet(fake_photo)

            monet_grid = torchvision.utils.make_grid(monet.cpu())
            photo_grid = torchvision.utils.make_grid(photo.cpu())
            fake_monet_grid = torchvision.utils.make_grid(fake_monet.cpu())
            fake_photo_grid = torchvision.utils.make_grid(fake_photo.cpu())
            reconstructed_photo_grid = torchvision.utils.make_grid(reconstructed_photo.cpu())
            reconstructed_monet_grid = torchvision.utils.make_grid(reconstructed_monet.cpu())

            # write to tensorboard
            self.writer.add_image("photo", photo_grid, epoch)
            self.writer.add_image('fake_monet', fake_monet_grid, epoch)
            self.writer.add_image("monet", monet_grid, epoch)
            self.writer.add_image('fake_photo', fake_photo_grid, epoch)
            self.writer.add_image('reconstructed_photo', reconstructed_photo_grid, epoch)
            self.writer.add_image('reconstructed_monet', reconstructed_monet_grid, epoch)

    def save_models(self, epoch):
        torch.save(self.G_photo.state_dict(), f"{models_dir}/G_photo_Ep{epoch:03d}.pth")
        torch.save(self.G_monet.state_dict(), f"{models_dir}/G_monet_Ep{epoch:03d}.pth")
        torch.save(self.D_monet.state_dict(), f"{models_dir}/D_monet_Ep{epoch:03d}.pth")
        torch.save(self.D_photo.state_dict(), f"{models_dir}/D_photo_Ep{epoch:03d}.pth")

    def run(self):
        for epoch in tqdm(range(1, num_epochs)):
            print("Epoch:", epoch)
            self.train_one_epoch(epoch, photo_dataloader, monet_dataloader)
            self.test_one_epoch(epoch)

            if epoch % 5 == 0:
                self.save_models(epoch)


#=============================
# CHOICE OF HYPERPARAMETERS
#=============================
num_epochs = 400
batch_size = 6
lr = 0.0001 #0.0002
momentum = 0.9
l = 10 # ratio CYCLE loss / GAN LOSS




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
# path = "runs/fit/20211101-071822/models"
path = None


photo_sampler = torch.utils.data.RandomSampler(photo_dataset, replacement=False)
monet_sampler = torch.utils.data.RandomSampler(monet_dataset, replacement=False)

# Create directories for logs
now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "runs/fit/" + now_str
summary_dir = "runs/fit/" + now_str
models_dir = log_dir + "/models"
test_dir = log_dir + "/test"
os.makedirs(log_dir, exist_ok=True)
#os.mkdir(summary_dir)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

trainer = Trainer(summary_dir, path=path)

trainer.run()

# Start tensorboard
#type "tensorboard --logdir=runs" in terminal
#type "tensorboard dev upload --logdir=runs/fit/NAME_DIRECTORY" for upload the log (no images)




