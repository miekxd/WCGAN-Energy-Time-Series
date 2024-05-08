import torch
import torch.nn as nn
import os

class Generator(nn.Module):
    def __init__(self, latent_dim, ts_dim, condition):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.ts_dim = ts_dim
        self.condition = condition
        self.hidden = 128
        
        self.block = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
           
        )
        self.block_cnn = nn.Sequential(
            nn.Conv1d(self.hidden,self.hidden, kernel_size=3, dilation=2, padding=2),
            nn.LeakyReLU(inplace=True),
        )
        self.block_shift = nn.Sequential(
            nn.Conv1d(self.hidden,10, kernel_size=3, dilation=2, padding=2),
            nn.LeakyReLU(inplace=True),
            
            nn.Flatten(start_dim=1),
            nn.Linear(10*self.latent_dim,256),
            nn.LeakyReLU(inplace=True),
        )
        self.noise_to_latent = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=self.hidden, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(self.hidden,self.hidden, kernel_size=5, dilation=2, padding=4),
            nn.LeakyReLU(inplace=True),
        )
        self.latent_to_output = nn.Sequential(
            nn.Linear(256, 4*(self.ts_dim-self.condition)),
        )

    def forward(self, input_data):
        
        x = self.noise_to_latent(input_data)
        
        x_block = self.block_cnn(x)
        x = x_block +x
        x_block = self.block_cnn(x)
        x = x_block +x
        x_block = self.block_cnn(x)
        x = x_block +x
        x = self.block_shift(x)
        x_block = self.block(x)
        x = x_block + x #torch.cat([x, x_block], 1)
        x_block = self.block(x)
        x = x_block + x #torch.cat([x, x_block], 1)
        x_block = self.block(x)
        x = x_block + x #torch.cat([x, x_block], 1)
        x = self.latent_to_output(x)
              
        return x[:,None,:]
    
    def save(self, path, *, filename=None, device="cpu"):
        # recommended way from http://pytorch.org/docs/master/notes/serialization.html
        self.to(device)
        if not filename is None:
            path = os.path.join(path, filename)
        torch.save(self.state_dict(), path)

    def load(self, path, *, filename=None):
        if not filename is None:
            path = os.path.join(path, filename)
        with open(path, "rb") as f:
            self.load_state_dict(torch.load(f))


class Discriminator(nn.Module):
    def __init__(self, ts_dim):
        super(Discriminator,self).__init__()

        self.ts_dim = ts_dim
        self.ts_to_feature = nn.Sequential(
            nn.Linear(4*self.ts_dim, 512),
            nn.LeakyReLU(inplace=True),
        )
        self.block = nn.Sequential(    
            nn.Linear(512, 512),
            nn.LeakyReLU(inplace=True),
        )
        self.to_score = nn.Sequential(
            nn.Linear(512, 1)
        )

            #nn.Sigmoid() #todo add acitivation or not, whole batch has same activiation?

        

    def forward(self, input_data):
        x = input_data.transpose(1, 2).contiguous().view(input_data.size(0), -1)
        x = self.ts_to_feature(x)
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x = self.to_score(x)
        
        return x
    
    def save(self, path, *, filename=None, device="cpu"):
        # recommended way from http://pytorch.org/docs/master/notes/serialization.html
        self.to(device)
        if not filename is None:
            path = os.path.join(path, filename)
        torch.save(self.state_dict(), path)

    def load(self, path, *, filename=None):
        if not filename is None:
            path = os.path.join(path, filename)
        with open(path, "rb") as f:
            self.load_state_dict(torch.load(f))