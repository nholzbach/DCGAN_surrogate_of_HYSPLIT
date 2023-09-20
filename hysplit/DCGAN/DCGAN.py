from __future__ import print_function
#%matplotlib inline
import argparse
import os, csv, time
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from IPython.display import HTML
from tqdm import tqdm
import h5py
# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

########## VARIABLES ##########
# Root directory for dataset
dataroottrain = "input_data/training"
datarootinput = "input_data/input"

# a description of the run to go into stats file
description = "added 2022-12 and 12 data, for longer (50 epochs)"
test_num = 22

# Number of workers for dataloader
workers = 0

# Batch size during training
batch_size = 128

# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. 
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 1589

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# the number of output channels (at the moment just 1, total_emissions)
nco = 1

# this is generated at the end of the code, but if it's already generated, load it here
test_vector = torch.load("results_images/test_vector.pt")
load_output = np.load("results_images/test_outputs.npy")
test_output = [torch.tensor(image_data) for image_data in load_output]



class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, training=True):
        self.data_dir = data_dir
        self.transform = transform
        self.training = training
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)

        if self.training == True:
            flat_input_read = []
            output_read = []

            # Read the data from the CSV file
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                # Read the first row (flat_input)
                flat_input_read = next(reader)
                # Read the rest of the rows (output)
                for row in reader:
                    output_read.append(row)

            flat_input = np.array(flat_input_read, dtype=np.float64)
            flat_input = torch.from_numpy(flat_input)
            
            output = np.array(output_read, dtype=np.float64)
            output = np.expand_dims(output, axis=0)
            data = np.transpose(output, (1, 2, 0))
        
        else:
            flat_input_read = []

            # Read the data from the CSV file
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                # Read the first row (flat_input)
                flat_input_read = next(reader)
                # Read the rest of the rows (output)

            flat_input = np.array(flat_input_read, dtype=np.float64)
            data = flat_input   
            
            
            
        if self.transform:
            data = self.transform(data)
        else:    
            data = torch.from_numpy(data)
        return data, flat_input
    

class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [-1, 1].
    """
    def __call__(self, tensor):
        min_vals = tensor.min(dim=1, keepdim=True)[0]
        max_vals = tensor.max(dim=1, keepdim=True)[0]
        
        dist = max_vals - min_vals
        dist[dist == 0.] = 1.0
        
        scale = 2.0 / dist
        
        tensor = tensor.sub(min_vals).mul(scale).sub(1.0)
        return tensor
    
class PyTMinMaxScalerFixedRange(object):
    """
    Scales a tensor from the range [min_val, max_val] to the range [-1, 1].
    """
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        dist = self.max_val - self.min_val
        if dist == 0.:
            scale = 0.0  # To avoid division by zero
        else:
            scale = 2.0 / dist
        tensor = tensor.sub(self.min_val).mul(scale).sub(1.0)
        return tensor


transforms = transforms.Compose([
    transforms.ToTensor(),

    PyTMinMaxScalerFixedRange(0.0, 35.0)

])

# load training data
dataset = CustomDataset(dataroottrain, transform=transforms, training = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

num_inputs = len(os.listdir(dataroottrain)) 

# this is not really relevant, helps visualise the data coming in and out
testimgs = []

for i, (data,input) in enumerate(dataloader):
    print(i, (data.shape, input.shape))
    
    # for the last batch the size, save the images:
    if i == len(dataloader)-1:
        for j in range(data.shape[0]):
            testimgs.append(data[j])


for i in range(len(test_output)):
    image_tensor = test_output[i]
    # Convert the tensor to a NumPy array
    image_array = image_tensor.squeeze().numpy()
    print(min(image_array.flatten()), max(image_array.flatten()))
    # Plot the image using matplotlib
    plt.imshow(image_array, cmap='gray')  # 'gray' colormap for grayscale images
    plt.axis('off')  # Turn off axis labels
    plt.colorbar()
    plt.show()



class CustomGeneratorLoss(nn.Module):
    def __init__(self, bce_weight=1.0, reconstruction_weight=0.1, use_mse=True):
        super(CustomGeneratorLoss, self).__init__()
        self.bce_weight = bce_weight
        self.reconstruction_weight = reconstruction_weight
        self.use_mse = use_mse


        if use_mse:
            self.reconstruction_loss = nn.MSELoss()
        else:
            self.reconstruction_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()

    def forward(self, output, target):
        # Calculate BCE loss
        bce_loss = self.bce_loss(output, target)

        # Calculate "negative loss" term
        reconstruction_loss = self.reconstruction_loss(output, target)
        
        # Combine the losses with weights
        total_loss = self.bce_weight * bce_loss + self.reconstruction_weight * reconstruction_loss 


        return total_loss

criterionG = CustomGeneratorLoss(use_mse=True)

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
       
       
# Create the generator
netG = Generator(ngpu, nz,ngf,nc).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
#  to ``mean=0``, ``stdev=0.02``.
netG.apply(weights_init)

# Print the model to check how the generator object is structured
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # print(input)
        return self.main(input)

       
# Same as generator, apply weights_init and print model structure
# Create the Discriminator
netD = Discriminator(ngpu, nc, ndf).to(device)


# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)

# Print the model
print(netD)       

# Initialize the ``BCELoss`` function (Binary Cross Entropy)
criterionD = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_input = test_vector.unsqueeze(-1).unsqueeze(-1).double().to(device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
# according to paper: learning rate 0.0002 and Beta1 = 0.5
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# TRAINING LOOP
# Lists to keep track of progress
img_list = []
tracking_images = []
G_losses = []
D_losses = []
iters = 0
netD.double()
netG.double()
print("Starting Training Loop...")
start_time = time.time() 

for epoch in tqdm(range(num_epochs)):
    # For each batch in the dataloader
    for batch_idx, (data, input_vector) in enumerate(dataloader):
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        
        netD.zero_grad()
        # Format batch
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        # print('batch size:', b_size)
        label = torch.full((b_size,), real_label, 
                           dtype=real_cpu.dtype, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterionD(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        # noise = torch.randint(0, 2, size=(batch_size, nz)).float() #from simple gan (binary input)
        # OR
        # noise = torch.randn(b_size, nz, 1, 1, device=device, dtype=torch.double)
        # load 'noise' with input vector
        noise = input_vector.unsqueeze(2).unsqueeze(3).double().to(device)
        
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterionD(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        # ############################
        # # (2) Update G network: maximize log(D(G(z)))
        # ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterionG(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        

        if (iters>4000) and (errG.item() < 1.0):
            print('saving model state, iteration:', iters)
            model_state = {
                'iteration': iters,
                'generator_state_dict': netG.state_dict(),
                'discriminator_state_dict': netD.state_dict(),
            }
            torch.save(model_state, f'results_images/test_{test_num}/model_state_threshold_{iters}.pth')
        
        # Check how the generator is doing by saving G's output on fixed input vector
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                test_display = netG(fixed_input).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
    
        
    if ((epoch+1)%1==0):
        
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (epoch+1, num_epochs,
                         errD.item(), errG.item()))
        
        
        plt.figure(figsize=(8,8))
        plt.axis("off")
        pictures=vutils.make_grid(test_display[torch.randint(len(test_display), (10,))],nrow=5,padding=2, normalize=True)
        plt.imshow(np.transpose(pictures,(1,2,0)))
        plt.show()
        
        with torch.no_grad():
            generated_images = netG(fixed_input).detach().cpu()
            tracking_images.append(generated_images.numpy())
        
end_time = time.time()
total_training_time = end_time - start_time
print(f"Total training time: {total_training_time:.2f} seconds")


num_inputs = len(os.listdir(dataroottrain)) 
# save statistics for later:
stats = {
    "G_losses": G_losses,
    "D_losses": D_losses,
    "total_training_time": total_training_time,
    "batch_size": batch_size,
    "epochs": num_epochs,
    "iterations": iters,
    # "img_list": img_list,
    "description": description,
    'test_num': test_num,
    # "last_images": fake.detach().numpy(),
    # "last training input": input_vector.numpy(),
    "number of inputs": num_inputs,
    # "tracking_images": tracking_images,
    # "test vector": test_vector.numpy(),
    # "test outputs": test_outputs
}
# save dict to csv
filename = f"stats/stats_{batch_size}_{num_epochs}_{num_inputs}_test{test_num}.pkl"
f = open(f"{filename}","wb")
pkl.dump(stats,f)
f.close()

# PLOT Loss versus training iteration
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()



# GENERATING DATA
dataset_test = CustomDataset(dataroottrain, transform=None, training = False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
    
all_generated_images = []
input_order = []
for i, (data,input) in enumerate(dataloader_test):
    print(i, (data.shape))
    with torch.no_grad():
        netG.eval()  # Set the Generator to evaluation mode
        generated_images = netG(data.unsqueeze(2).unsqueeze(3)).view(-1, 64, 64).cpu()
        all_generated_images.append(generated_images)
        input_order.append(data)
        
flat_output = [item for sublist in all_generated_images for item in sublist]
flat_input = [item for sublist in input_order for item in sublist]
flat_input_tensor = torch.stack(flat_input)
# save
torch.save(flat_input_tensor, f"results_images/test_{test_num}/full_input_100epoch.pt")

# Save the list of tensors to an HDF5 file
with h5py.File(f'results_images/test_{test_num}/tensors_100epochs.h5', 'w') as hf:
    for idx, tensor in enumerate(flat_output):
        hf.create_dataset(f'tensor_{idx}', data=tensor)


# saving one set of test images and input
# done once, doesn't need to happen again
num_samples = 64
random_indices = random.sample(range(len(input)), num_samples)

test_vector = [input[idx] for idx in random_indices]
test_outputs = [data[idx] for idx in random_indices]

# save these
test_vector_tensor = torch.stack(test_vector)
torch.save(test_vector_tensor, 'results_images/test_vector.pt')

test_outputs_array = np.stack([t.numpy() for t in test_outputs])

np.save('results_images/test_outputs.npy', test_outputs_array)

