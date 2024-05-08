import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, ts_dim):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.ts_dim = ts_dim

        self.noise_to_latent = nn.Sequential(
            nn.Linear(self.latent_dim, 50),
            nn.LeakyReLU(inplace=True),
            nn.Linear(50, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 2*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2*self.ts_dim, 2*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2*self.ts_dim, self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.ts_dim, self.ts_dim),
            #nn.Tanh()
        )

    def forward(self, input_data):
        x = self.noise_to_latent(input_data)

        return x[:, None, :]
    
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

        self.features_to_score = nn.Sequential(
            nn.Linear(self.ts_dim, 2*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2*self.ts_dim, 4*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4*self.ts_dim, 5*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(5*self.ts_dim, 5*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(5*self.ts_dim, 6*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(6*self.ts_dim, 2*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2*self.ts_dim, 1)

            #nn.Sigmoid() #todo add acitivation or not, whole batch has same activiation?

        )

    def forward(self, input_data):

        x = self.features_to_score(input_data)
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