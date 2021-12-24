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

"""
 - Aggiungere bias ?
 - Mettere il dropout
 - Reflection padding
 - Controllare la patchgan
 - cambiare il lr con l'avnazare delle epoch
 FATTO! - Applicare l'identity loss su immagini che corrispondono all'insieme d'arrivo previsto (e non quello di partenza)
 FATTO! - implementare fakesampler
 FATTO! - Due layer per il residual block


"""


class FakeSampler():
    import random
    def __init__(self, size=50):
        self.samples = []
        self.size = size

    def add(self, sample):
        """Add a sample to sampler"""
        self.samples.append(sample)

    def get(self):
        """Get a sample and remove it from the list if there are already size samples"""
        list_size = len(self.samples)
        idx = random.randint(0, list_size - 1)
        if list_size < self.size:
            return self.samples[idx]
        return self.samples.pop(idx)


class RollingAverage:
    def __init__(self, size=10):
        self.pool = [0] * size
        self.idx = 0
        self.size = size

    def add(self, x):
        self.pool[self.idx] = x
        self.idx += 1
        if self.idx == self.size:
            self.idx = 0

    def mean(self):
        return np.mean(self.pool)


def gan_loss(test, label):
    import torch.nn.functional as F
    if label==1:
        label = torch.ones(test.shape, device=device)
    else:
        label = torch.zeros(test.shape, device=device)
    #loss = F.binary_cross_entropy_with_logits(test, label)
    #loss = F.mse_loss(test, label)
    loss = F.l1_loss(test, label)
    return loss


cycle_loss = torch.nn.L1Loss()
identity_loss = torch.nn.L1Loss()

def enable_grad(models, enable):
    for model in models:
        for param in model.parameters():
            param.requires_grad = enable

def to_01(img):
    """COnvert image with pixels in range [-1,1] to [0,1]"""
    return img
    return (img + 1) / 2

class Trainer:
    def __init__(self, summary_dir, path=None, epoch=None):
        """Create models and load weights from a path if specified"""
        self.G_photo = Generator().to(device)  # photo generator
        self.G_monet = Generator().to(device)  # monet generator
        self.D_monet = Discriminator().to(device)  # 1=real, 0=fake
        self.D_photo = Discriminator().to(device)

        self.avg_disc_photo = RollingAverage(rolling_av_size)
        self.avg_disc_monet = RollingAverage(rolling_av_size)
        self.avg_gen_photo = RollingAverage(rolling_av_size)
        self.avg_gen_monet = RollingAverage(rolling_av_size)

        self.fake_photo_sampler = FakeSampler(sampler_size)
        self.fake_monet_sampler = FakeSampler(sampler_size)

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



    def _train_disc_helper(self, D, opt, real, fake_sampler):
        """Train a discriminator using a batch of images"""
        enable_grad([D], True)
        dloss = torch.zeros(1).to(device)
        opt.zero_grad()
        dloss_real = gan_loss(D(real), 1)
        fake = fake_sampler.get()
        dloss_fake = gan_loss(D(fake), 0)
        dloss += (dloss_real + dloss_fake).sum()
        # update model
        dloss.backward()
        opt.step()
        return dloss.item()

    def train_discriminator_photo(self, epoch, iter):
        dloss_photo = self._train_disc_helper(self.D_photo, self.opt_Dp, self.photo, self.fake_photo_sampler)
        self.writer.add_scalar("d_photo loss",
                               dloss_photo,
                               self.iteration)
        return dloss_photo

    def train_discriminator_monet(self, epoch, iter):
        dloss_monet = self._train_disc_helper(self.D_monet, self.opt_Dm, self.monet, self.fake_monet_sampler)
        self.writer.add_scalar("d_monet loss",
                               dloss_monet,
                               self.iteration)
        return dloss_monet



    def train_generators(self, dumb=False, train_monet=True, train_photo=True):
        enable_grad([self.G_photo], train_photo)
        enable_grad([self.G_monet], train_monet)
        enable_grad([self.D_photo, self.D_monet], False)

        self.opt_G_photo.zero_grad()
        self.opt_G_monet.zero_grad()

        # generate fakes
        self.fake_monet = self.G_monet(self.photo)
        self.fake_photo = self.G_photo(self.monet)
        # generate cycles
        reconstructed_photo = self.G_photo(self.fake_monet)
        reconstructed_monet = self.G_monet(self.fake_photo)

        #ic("before", fake_photo.requires_grad)
        #ic("after", fake_photo.requires_grad)

        #Identity loss
        # generate clones (monet and photo are transformed to images in the same set
        self.clone_monet = self.G_monet(self.monet)
        self.clone_photo = self.G_photo(self.photo)

        id_loss_G_photo = identity_loss(self.photo, self.clone_photo)
        id_loss_G_monet = identity_loss(self.monet, self.clone_monet)
        id_loss = (id_loss_G_photo + id_loss_G_monet).sum()
        #print(id_loss.requires_grad)

        if not dumb:
            # Cycle loss
            cycleloss_G_photoG_monet = cycle_loss(self.photo, reconstructed_photo)
            cycleloss_G_monetG_photo = cycle_loss(self.monet, reconstructed_monet)
            cycle_losses = (cycleloss_G_photoG_monet + cycleloss_G_monetG_photo).sum()

            # GAN loss generator
            gloss_monet_fake = gan_loss(self.D_monet(self.fake_monet), 1)
            floss_photo_fake = gan_loss(self.D_photo(self.fake_photo), 1)
            gan_generator_losses = gloss_monet_fake.sum() + floss_photo_fake.sum()

            generator_loss = l * cycle_losses + gan_generator_losses + m * id_loss
        else:
            generator_loss = m * id_loss

        if train_photo or train_monet:
            generator_loss.backward()

        # Update backpropagation for generators
        self.opt_G_photo.step()
        self.opt_G_monet.step()


        if not dumb:
            self.writer.add_scalar("cycle loss",
                              cycle_losses.item(),
                              self.iteration)
            self.writer.add_scalar("gan G_photo loss",
                              floss_photo_fake.item(),
                              self.iteration)
            self.writer.add_scalar("gan G_monetloss",
                              gloss_monet_fake.item(),
                              self.iteration)

        self.writer.add_scalar("Identity loss",
                               id_loss.item(),
                               self.iteration)

        return gloss_monet_fake.item(), floss_photo_fake.item()

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
            self.photo = next(photo_iterator)
            self.monet = next(monet_iterator)
            # Load to GPU
            self.photo = self.photo.to(device)
            self.monet = self.monet.to(device)

            # Enable all grads
            enable_grad([self.G_photo, self.G_monet], True)
            enable_grad([self.D_photo, self.D_monet], True)

            #=============================
            # IMAGE GENERATION
            #=============================

            self.iteration = epoch * n + i

            better_disc_monet = bool(self.avg_gen_monet.mean() > (self.avg_disc_monet.mean() * threshold))
            better_disc_photo = bool(self.avg_gen_photo.mean() > (self.avg_disc_photo.mean() * threshold))

            if alternate_training:
                print("Monet Discriminator > Generator: ", better_disc_monet, "Photo Discriminator > Generator: ", better_disc_photo)

            disc_monet_loss_gen, disc_photo_loss_gen = self.train_generators(False, better_disc_monet or not alternate_training, better_disc_photo or not alternate_training)
            self.avg_gen_monet.add(disc_monet_loss_gen)
            self.avg_gen_photo.add(disc_photo_loss_gen)

            #=========================================
            # Add last images to discriminator batch
            #=======================================

            # update fake_samplers
            self.fake_monet_sampler.add(self.fake_monet.detach())
            self.fake_photo_sampler.add(self.fake_photo.detach())

            if not better_disc_monet or not alternate_training:
                disc_monet_loss_disc = self.train_discriminator_monet(epoch,  self.iteration)
                self.avg_disc_monet.add(disc_monet_loss_disc)

            if not better_disc_photo or not alternate_training:
                disc_photo_loss_disc = self.train_discriminator_photo(epoch, self.iteration)
                self.avg_disc_photo.add(disc_photo_loss_disc)

            # Upload images to tensorboard
            if i%7 == 0:
                self.update_images_tensorboard()


    def update_images_tensorboard(self):
        # Detach images
        monet_cpu = self.monet.detach().cpu()
        photo_cpu = self.photo.detach().cpu()
        fake_photo_cpu = self.fake_photo.detach().cpu()
        fake_monet_cpu = self.fake_monet.detach().cpu()

        monet_cpu = self.monet.detach().cpu()
        photo_cpu = self.photo.detach().cpu()
        fake_photo_cpu = self.fake_photo.detach().cpu()
        fake_monet_cpu = self.fake_monet.detach().cpu()

        # Create color level curve image
        fig_monet, ax = plt.subplots(1, 1)
        plot_color_curve(ax, monet_cpu[0], label="monet", c="r")
        plot_color_curve(ax, fake_photo_cpu[0], label="fake_photo", c="r", linestyle='--')
        plot_color_curve(ax, photo_cpu[0], label="photo", c="blue")
        plot_color_curve(ax, fake_monet_cpu[0], label="fake_monet", c="blue", linestyle='--')
        ax.legend()
        plot_tb = plot_to_image(fig_monet)
        plt.close(fig_monet)
        # Make a grid of all images with 4 rows
        all_images = torch.cat([fake_photo_cpu, monet_cpu, photo_cpu, fake_monet_cpu])
        all_grid = torchvision.utils.make_grid(to_01(all_images).cpu(), nrow=batch_size)
        # create grid of images

        # reconstructed_photo_grid = torchvision.utils.make_grid(reconstructed_photo.cpu())
        # reconstructed_monet_grid = torchvision.utils.make_grid(reconstructed_monet.cpu())

        # write to tensorboard
        self.writer.add_image("[train] photo", all_grid, self.iteration)
        self.writer.add_image("Color curves", plot_tb, self.iteration)

        # writer.add_image('reconstructed_photo', reconstructed_photo_grid)
        # writer.add_image('reconstructed_monet', reconstructed_monet_grid)

    def test_one_epoch(self, epoch):
        self.G_photo.eval()
        self.G_monet.eval()
        self.D_photo.eval()
        self.G_monet.eval()

        self.photo = photo_dataset_test[0].unsqueeze(0).to(device)
        self.monet = monet_dataset_test[0].unsqueeze(0).to(device)
        with torch.no_grad():
            self.fake_monet = self.G_monet(self.photo)
            self.fake_photo = self.G_photo(self.monet)
            reconstructed_photo = self.G_photo(self.fake_monet)
            reconstructed_monet = self.G_monet(self.fake_photo)

            monet_cpu = self.monet.detach().cpu()
            fake_photo_cpu = self.fake_photo.detach().cpu()

            monet_grid = torchvision.utils.make_grid(monet_cpu)
            photo_grid = torchvision.utils.make_grid(self.photo.detach().cpu())
            fake_monet_grid = torchvision.utils.make_grid(self.fake_monet.detach().cpu())
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
    batch_size = 5
    lr = 0.0002 #0.0002

    # Loss ratio parameters
    l = 10 # ratio CYCLE loss / GAN LOSS
    m = l * 0.5


    # Parameters alternate training
    alternate_training = False
    rolling_av_size = 10
    threshold = 1
    # PArameters fake sampler
    sampler_size = 50



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
    path = "runs/fit/20211216-230745/models"


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

    trainer = Trainer(summary_dir, path=path, epoch=230)

    trainer.run()

    # Start tensorboard
    #type "tensorboard --logdir=runs" in terminal
    #type "tensorboard dev upload --logdir=runs/fit/NAME_DIRECTORY" for upload the log (no images)




