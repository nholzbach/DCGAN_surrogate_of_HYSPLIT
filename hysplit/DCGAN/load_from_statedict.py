# loading state dict from file and generating data
import csv, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

# MAY NEED TO CHANGE
dataroottrain = "input_data/training"
datarootinput = "input_data/input"

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
            # nn.Sigmoid()
            # nn.ReLU(True)
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
       

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
    Transforms each channel to the range [0, 1].
    """
    def __call__(self, tensor):
        dist = (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0])
        dist[dist==0.] = 1.
        scale = 1.0 /  dist
        tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
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

# load the model - EDIT THESE PATHS/VARIABLES
test_num = 22
iter_num = 11571
PATH = f'results_images/test_{test_num}/model_state_threshold_{iter_num}.pth'
model = Generator(0,1589,64,1)
checkpoint = torch.load(PATH)
model_state_dict = checkpoint['generator_state_dict']
model.load_state_dict(model_state_dict)

# testing with some images
dataset_test = CustomDataset(dataroottrain, transform=None, training = False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=128,
                                         shuffle=True, num_workers=0)

# generate and save 
model.double()
all_generated_images = []
input_order = []
for i, (data,input) in enumerate(dataloader_test):
    print(i, (data.shape))
    data = data.double()
    with torch.no_grad():
        model.eval()  # Set the Generator to evaluation mode
        generated_images = model(data.unsqueeze(2).unsqueeze(3)).view(-1, 64, 64).cpu()
        all_generated_images.append(generated_images)
        input_order.append(data)
        
flat_output = [item for sublist in all_generated_images for item in sublist]
flat_input = [item for sublist in input_order for item in sublist]
flat_input_tensor = torch.stack(flat_input)
# save
torch.save(flat_input_tensor, f"results_images/test_{test_num}/state{iter_num}_full_input.pt")

import h5py
# Save the list of tensors to an HDF5 file
with h5py.File(f'results_images/test_{test_num}/tensors_state{iter_num}.h5', 'w') as hf:
    for idx, tensor in enumerate(flat_output):
        hf.create_dataset(f'tensor_{idx}', data=tensor)

