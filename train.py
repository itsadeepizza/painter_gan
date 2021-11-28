import torchvision
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import plot_color_curve, plot_to_image
from model.mini_cyclegan import Generator, Discriminator
from loader import ImageDataset
from tqdm import tqdm
import random
import datetime
import os
from icecream import ic
import matplotlib.pyplot as plt
import numpy as np


def gan_loss(test, label):
    import torch.nn.functional as F
    if label==1:
        label = torch.ones(test.shape, device=device)
    else:
        label = torch.zeros(test.shape, device=device)
    #loss = F.binary_cross_entropy_with_logits(test, label)
    loss = F.mse_loss(test, label)
    #loss = F.l1_loss(test, label)
    return loss


cycle_loss = torch.nn.L1Loss()
identity_loss = torch.nn.L1Loss()

def enable_grad(models, enable):
    for model in models:
        for param in model.parameters():
            param.requires_grad = enable

def to_01(img):
    return img
    return (img + 1) / 2

class Trainer:
    def __init__(self, summary_dir, path=None, epoch=None):
        """Create models and load weights from a path if specified"""
        self.G_photo = Generator().to(device)  # photo generator
        self.G_monet = Generator().to(device)  # monet generator
        self.D_monet = Discriminator().to(device)  # 1=real, 0=fake
        self.D_photo = Discriminator().to(device)
        if path is not None:
            #
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

        #initialise discriminators batch
        self.real_photos = []
        self.fake_photos = []
        self.real_monets = []
        self.fake_monets = []



    def train_discriminators(self, epoch, iter):

        enable_grad([self.D_photo, self.D_monet], True)
        dloss_photo = torch.zeros(1).to(device)
        dloss_monet = torch.zeros(1).to(device)
        self.opt_Dp.zero_grad()
        self.opt_Dm.zero_grad()
        for real_photo, fake_photo, real_monet, fake_monet in zip(self.real_photos, self.fake_photos, self.real_monets, self.fake_monets):
            # D_photo vs G_photo
            dloss_photo_real = gan_loss(self.D_photo(real_photo), 1)
            dloss_photo_fake = gan_loss(self.D_photo(fake_photo), 0)
            dloss_photo += (dloss_photo_real + dloss_photo_fake).sum()

            # D_monet vs G_monet
            dloss_monet_real = gan_loss(self.D_monet(real_monet), 1)
            dloss_monet_fake = gan_loss(self.D_monet(fake_monet), 0)
            dloss_monet += (dloss_monet_real + dloss_monet_fake).sum()
        # update model
        dloss_photo.backward()
        dloss_monet.backward()
        self.opt_Dp.step()
        self.opt_Dm.step()

        # report loss to tensorboard
        self.writer.add_scalar("d_photo loss",
                          dloss_photo.item()/len(real_photo),
                          iter)

        self.writer.add_scalar("d_monet loss",
                          dloss_monet.item()/len(real_photo),
                          iter)

    def train_generators(self, epoch, iter, real_photo, fake_photo, real_monet, fake_monet, dumb=False):
        enable_grad([self.G_photo, self.G_monet], True)
        enable_grad([self.D_photo, self.D_monet], False)
        # generate cycles
        reconstructed_photo = self.G_photo(fake_monet)
        reconstructed_monet = self.G_monet(fake_photo)

        self.opt_G_photo.zero_grad()
        self.opt_G_monet.zero_grad()
        #ic("before", fake_photo.requires_grad)
        # generate fakes again
        #fake_monet = self.G_monet(real_photo)
        #fake_photo = self.G_photo(real_monet)
        #ic("after", fake_photo.requires_grad)

        #Identity loss
        id_loss_G_photo = identity_loss(real_monet, fake_photo)
        id_loss_G_monet = identity_loss(real_photo, fake_monet)
        id_loss = (id_loss_G_photo + id_loss_G_monet).sum()
        #print(id_loss.requires_grad)

        if not dumb:
            # Cycle loss
            cycleloss_G_photoG_monet = cycle_loss(real_photo, reconstructed_photo)
            cycleloss_G_monetG_photo = cycle_loss(real_monet, reconstructed_monet)
            cycle_losses = (cycleloss_G_photoG_monet + cycleloss_G_monetG_photo).sum()

            # GAN loss generator

            gloss_monet_fake = gan_loss(self.D_monet(fake_monet), 1)
            floss_photo_fake = gan_loss(self.D_photo(fake_photo), 1)
            gan_generator_losses = gloss_monet_fake.sum() + floss_photo_fake.sum()

            generator_loss = l * cycle_losses + gan_generator_losses + m * id_loss
        else:
            generator_loss = m * id_loss

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

            enable_grad([self.G_photo, self.G_monet], True)
            enable_grad([self.D_photo, self.D_monet], True)

            #=============================
            # IMAGE GENERATION
            #=============================
            # generate fakes
            fake_monet = self.G_monet(photo)
            fake_photo = self.G_photo(monet)

            self.train_generators(epoch, epoch * n + i, photo, fake_photo, monet, fake_monet)
            self.train_discriminators(epoch, epoch * n + i, photo, fake_photo, monet, fake_monet)


            #Upload losses to Tensorboard


    
            # Upload images to tensorboard
            if i%7 == 0:

                monet_cpu = monet.detach().cpu()
                photo_cpu = photo.detach().cpu()
                fake_photo_cpu = fake_photo.detach().cpu()
                fake_monet_cpu = fake_monet.detach().cpu()

                fig_monet, ax = plt.subplots(1, 1)
                plot_color_curve(ax, monet_cpu[0], label="monet", c="r")
                plot_color_curve(ax, fake_photo_cpu[0], label="fake_photo", c="r", linestyle='--')
                plot_color_curve(ax, photo_cpu[0], label="photo", c="blue")
                plot_color_curve(ax, fake_monet_cpu[0], label="fake_monet", c="blue", linestyle='--')
                ax.legend()
                plot_tb = plot_to_image(fig_monet)




                # create grid of images

                #reconstructed_photo_grid = torchvision.utils.make_grid(reconstructed_photo.cpu())
                #reconstructed_monet_grid = torchvision.utils.make_grid(reconstructed_monet.cpu())
    
                # write to tensorboard
                self.writer.add_image("[train] photo", all_grid, iteration)
                self.writer.add_image("Color curves", plot_tb, iteration)

                #writer.add_image('reconstructed_photo', reconstructed_photo_grid)
                #writer.add_image('reconstructed_monet', reconstructed_monet_grid)


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

            monet_cpu = monet.detach().cpu()
            fake_photo_cpu = fake_photo.detach().cpu()



            monet_grid = torchvision.utils.make_grid(monet_cpu)
            photo_grid = torchvision.utils.make_grid(photo.detach().cpu())
            fake_monet_grid = torchvision.utils.make_grid(fake_monet.detach().cpu())
            fake_photo_grid = torchvision.utils.make_grid(fake_photo_cpu)
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

# For using multi-threads in Windows
if __name__=="__main__":
    #=============================
    # CHOICE OF HYPERPARAMETERS
    #=============================
    num_epochs = 4000
    batch_size = 8
    lr = 0.0002 #0.0002
    momentum = 0.9
    l = 2 # ratio CYCLE loss / GAN LOSS
    m = l * 0.5



    # Choose device
    # uncomment below
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device used: ", device)
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
                                                   num_workers=6
                                                   )
    photo_dataloader = torch.utils.data.DataLoader(photo_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=6
                                                   )

    # Load model (if path is None create a new model
    # path = "runs/fit/20211101-071822/models"
    path = None


    photo_sampler = torch.utils.data.RandomSampler(photo_dataset, replacement=False)
    monet_sampler = torch.utils.data.RandomSampler(monet_dataset, replacement=False)

    # Create directories for logs
    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "runs/fit/" + now_str
    summary_dir = log_dir + "/summary"
    models_dir = log_dir + "/models"
    test_dir = log_dir + "/test"
    os.makedirs(log_dir, exist_ok=True)
    #os.mkdir(summary_dir)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    trainer = Trainer(summary_dir, path=path, epoch=340)

    trainer.run()

    # Start tensorboard
    #type "tensorboard --logdir=runs" in terminal
    #type "tensorboard dev upload --logdir=runs/fit/NAME_DIRECTORY" for upload the log (no images)




