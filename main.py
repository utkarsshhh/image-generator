# Importing the libraries
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


## Defining the layer of the Generator

def generator_layer(input_dim,output_dim):
    '''

    This function creates a layer of the generator neural network.

    Input:
    input_dim: It is the dimension of the layer's input, a scaler
    output_dim: It is the dimension of the layer's output vector

    Output:
    returns the output of the layer after linear transformation followed by batch normalisation
    and ReLU activation function

    '''
    return nn.Sequential(
        nn.Linear(input_dim,output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace = True)
    )

## Defining the Generator

class Generator(nn.Module):
    '''
    Generator class
    Inputs:
        z_dim: the dimension of the noise vector
        im_dim: the dimension of the output images
        hidden_dim: the dimension for the hidden layer of the neural network
    '''

    def __init__(self,z_dim =10,im_dim = 784,hidden_dim=128):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            generator_layer(z_dim,hidden_dim),
            generator_layer(hidden_dim,hidden_dim*2),
            generator_layer(hidden_dim*2,hidden_dim*4),
            generator_layer(hidden_dim*4,hidden_dim*8),
            nn.Linear(hidden_dim*8,im_dim),
            nn.Sigmoid()
        )
    def forward(self,noise):
        '''
        This function is for completing one forward pass of the geneartor

        Inputs:
        noise: The noise vector for image generation

        Outputs:
        return: generated images
        '''
        return self.gen(noise)

def noise_gen(n_samples,z_dim):
    '''
        Function for creating noise vector

        Inputs:
        z_dim: The dimension of the noise vector
        n_samples: The number of samples to be generated

        :Outputs:
        returns a random generated tensor of shape (n_samples,z_dim)
    '''

    return torch.randn(n_samples,z_dim)

def discriminator_layer(input_dim,output_dim):
    '''
    This function creates layer in the discriminator neural network
    Inputs:
    input_dim: the dimension of the input vector
    output_dim: the dimension of the output vector

    Output:
    returns a neural network layer according to the input and output dimensions
    '''

    return nn.Sequential(
        nn.Linear(input_dim,output_dim),
        nn.LeakyReLU(0.2,True)
    )

class Discriminator(nn.Module):
    '''
    defining the Discriminator Class
    Inputs:
    im_dim: the dimension of the images, fitted for the images used, a scaler
    hidden_dim: the hidden layer dimension
    '''

    def __init__(self,im_dim=784,hidden_dim=128):
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            discriminator_layer(im_dim,hidden_dim*4),
            discriminator_layer(hidden_dim*4,hidden_dim*2),
            discriminator_layer(hidden_dim*2,hidden_dim),
            nn.Linear(hidden_dim,1)
        )

    def forward(self,image):
        '''
        function for completing a forward pass of the discriminator
        Inputs:
        image: a flattened vector of the image
        '''
        return self.disc(image)

## Setting up the parameters

criterion = nn.BCEWithLogitsLoss()
n_epochs = 120
z_dim= 64
display_step = 500
batch_size = 128
lr = 0.00001

dataloader = DataLoader(
    MNIST('.',download=True,transform=transforms.ToTensor()),
    batch_size = batch_size,
    shuffle=True
)
device = 'cpu'

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(),lr = lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(),lr=lr)

## Defining loss functions

def get_disc_loss(gen,disc,criterion,real,num_images,z_dim,device):
    '''
    returns the loss of the discriminator against the given inputs

    Inputs:
    gen: the generator model, which returns an image
    disc: the discriminator model, which returns a single dimension
        prediction value for an image fake/real
    criterion: loss function which is used to compare the predictions
        of the discriminator with the actual labels
    real: a batch of real images
    num_images: the number of images to be generated, same as the number
        real images
    z_dim: the dimension of the noise vector
    device: the device type (cpu)

    Outputs:
    disc_loss: a torch scale value for the current batch

    '''

    noise_vectors = noise_gen(num_images,z_dim)
    fake_images = gen(noise_vectors)
    pred_fake = disc(fake_images.detach())
    pred_true = disc(real)
    disc_loss = (criterion(pred_fake,torch.zeros_like(pred_fake)) + criterion(pred_true,torch.ones_like(pred_true)))/2
    return disc_loss

def get_gen_loss(gen,disc,criterion,num_images,z_dim,device):
    '''
    returns the generator loss

    Inputs:
    gen: the generator model
    disc: the discriminator model
    criterion: the loss function that is used to compare the real lables
        and the prediction of the discriminator
    num_images: the number of images to be generated
    z_dim: the dimension of th noise vector, a scaler
    device: the device (cpu)

    Output:
    gen_loss - a torch scaler loss value of the generator model
    '''

    noise_vectors = noise_gen(num_images,z_dim)
    fake_images = gen(noise_vectors)
    pred_fake = disc(fake_images)
    gen_loss = criterion(pred_fake,torch.ones_like(pred_fake))
    return gen_loss

## Generating images

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True
gen_loss = False
error = False
z_dim = 64

for epoch in range(n_epochs):

    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

        real = real.view(cur_batch_size,-1).to(device)
        disc_opt.zero_grad()
        disc_loss = get_disc_loss(gen,disc,criterion,real,cur_batch_size,z_dim,device)
        disc_loss.backward(retain_graph = True)
        disc_opt.step()

        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen,disc,criterion,cur_batch_size,z_dim,device)
        gen_loss.backward(retain_graph = True)
        gen_opt.step()

        mean_discriminator_loss += disc_loss.item()/display_step
        mean_generator_loss += gen_loss.item()/display_step

        if cur_step % display_step == 0 and cur_step > 0:
            print(
                f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = noise_gen(cur_batch_size, z_dim)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1





