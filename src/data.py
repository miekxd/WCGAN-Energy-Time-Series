import numpy as np
import pandas as pd
import torch
import os
from sklearn.preprocessing import StandardScaler
from scipy.special import lambertw
import sys

from dense3 import Generator, Discriminator

#pip install torch==1.2.0 torchvision==0.4.0 
#pip install -U scikit-learn
#pip install statsmodels

def parse_time_interval(time_interval):
    start_time, end_time = time_interval.split(' - ')
    return start_time, end_time

class Data(object):
    def __init__(self,length, path):
        self.data = pd.read_csv(path)
        self.delta = 0.01
        self.length = length
        self.init_all()
        
        
    def init_all(self):
        self.get_scalar()
        self.store_energy()
        self.store_temp()
        self.store_feelslike()
        self.store_humidity()
        self.store_time()
        self.store_time_features()
        
        #self.store_response()
        df=pd.concat([self.data[['energy','temp']],self.df],axis=1)
        #df.to_csv("data.csv", index=False)
        print(df.shape)
        
        self.features_return=self.preprocessing(pd.concat([self.data[['energy','temp']],self.df],axis=1))  
        
        self.features_return1 = self.features_return.T    
        self.energy_return = self.features_return1[0,:]       
        self.temp_return = self.features_return1[1,:]
        #self.feelslike_return = self.features_return1[1,:]
        #self.humidity_return = self.features_return1[3,:]
        self.day_return = self.features_return1[2,:]
        self.hour_return = self.features_return1[3,:]
        
        #self.month_return = self.features_return1[5,:]
        #self.response_return = self.features_return1[4,:]
        
        self.data_augment()
        
    def store_energy(self):
        self.energy = self.data['energy'].to_numpy()    
        
    def store_temp(self):
        self.temp = self.data['temp'].to_numpy()
    
    def store_feelslike(self):
        self.feelslike = self.data['feelslike'].to_numpy()
    
    def store_humidity(self):
        self.humidity = self.data['humidity'].to_numpy()
        
    def store_time(self):
        self.time = pd.to_datetime(self.data['datetime'], format="%Y-%m-%dT%H:%M:%S")

    def store_time_features(self):
        #self.year = self.data['datetime'].dt.year
        self.month = self.time.dt.month
        self.day = self.time.dt.weekday + 1
        self.hour = self.time.dt.hour + 1 % 24 #change to 1-24 for preprocessing
        #self.minute = self.data['datetime'].dt.minute
        #self.second = self.data['datetime'].dt.second
        
        self.timedict={'day': self.day, 'hour': self.hour}
        self.df=pd.DataFrame(self.timedict)
        
    def get_scalar(self):
        self.scalar = StandardScaler()
        self.scalar2 = StandardScaler()
        
    def moving_window(self,x, length):
        return [x[i: i+ length] for i in range(0,(len(x)+1)-length, 4)]
    
    def preprocessing(self, data): #data normalization
        log_returns = np.log(data/data.shift(1)).fillna(0).to_numpy()
        #log_returns = np.reshape(log_returns, (log_returns.shape[0],5))
        
        #scale the values
        self.scalar = self.scalar.fit(log_returns)
        log_returns = np.squeeze(self.scalar.transform(log_returns))
        log_returns_w = (np.sign(log_returns)*np.sqrt(lambertw(self.delta*log_returns**2)/self.delta)).real
        #log_returns_w = log_returns_w.reshape(-1,5)
        self.scalar2 = self.scalar2.fit(log_returns_w)
        log_returns_w = np.squeeze(self.scalar2.transform(log_returns_w))
        return log_returns_w

    
    def data_augment(self):
        self.energy_return_aug = np.array(self.moving_window(self.energy_return, self.length))
        self.temp_return_aug = np.array(self.moving_window(self.temp_return, self.length))
        #self.feelslike_return_aug = np.array(self.moving_window(self.feelslike_return, self.length))
        #self.humidity_return_aug = np.array(self.moving_window(self.humidity_return, self.length))
        self.day_return_aug = np.array(self.moving_window(self.day_return, self.length))
        self.hour_return_aug = np.array(self.moving_window(self.hour_return, self.length))
        #self.month_return_aug = np.array(self.moving_window(self.month_return, self.length))
        #self.response_return_aug = np.array(self.moving_window(self.response_return, self.length))
        
        self.energy_aug = np.array(self.moving_window(self.energy, self.length))
        self.temp_aug = np.array(self.moving_window(self.temp, self.length))
        #self.feelslike_aug = np.array(self.moving_window(self.feelslike, self.length))
        #self.humidity_aug = np.array(self.moving_window(self.humidity, self.length))
        self.day_aug = np.array(self.moving_window(self.day, self.length))
        self.hour_aug = np.array(self.moving_window(self.hour, self.length))
        #self.month_aug = np.array(self.moving_window(self.month, self.length))
        #self.response_aug = np.array(self.moving_window(self.response, self.length))
        
        self.time_aug = np.array(self.moving_window(self.time, self.length))
        print("Energy_return_aug shape: ",self.energy_return_aug.shape)       
        
    def post_processing(self, return_data, init):
        return_data=return_data.T
        
        return_data = self.scalar2.inverse_transform(return_data)
        #print(np.max(return_data))
        return_data = return_data * np.exp(0.5 * self.delta * np.clip(return_data**2, -700, 700))
        return_data = self.scalar.inverse_transform(return_data)
        return_data = np.clip(return_data, -10, 10)
        return_data = np.exp(return_data)
        
        return_data = return_data[:,0]
        return_data = return_data.reshape(-1,1)
        
        post_return = np.empty((return_data.shape[0],))
        post_return[0] = init[0]
        for i in range(1,return_data.shape[0]):
            post_return[i] = post_return[i-1] * return_data[i]
        return post_return
        
    
    def post_processing_specific(self, idx, return_data, init):
        return_data=return_data.T
        
        return_data = self.scalar2.inverse_transform(return_data)
        #print(np.max(return_data))
        return_data = return_data * np.exp(0.5 * self.delta * np.clip(return_data**2, -700, 700))
        return_data = self.scalar.inverse_transform(return_data)
        return_data = np.clip(return_data, -10, 10)
        return_data = np.exp(return_data)
        
        return_data = return_data[:,idx]
        return_data = return_data.reshape(-1,1)
        
        post_return = np.empty((return_data.shape[0],))
        post_return[0] = init[idx]
        for i in range(1,return_data.shape[0]):
            post_return[i] = post_return[i-1] * return_data[i]
        return post_return
    
    def __len__(self):
        return len(self.energy_return_aug)
    
    def get_single_sample(self):
        idx = np.random.randint(self.energy_return_aug.shape[0], size=1)
        
        real_start_energy = self.energy_aug[idx, 0]
        real_start_temp = self.temp_aug[idx, 0]
        #real_start_feelslike = self.feelslike_aug[idx, 0]
        #real_start_humidity = self.humidity_aug[idx, 0]
        real_start_day = self.day_aug[idx, 0]
        real_start_hour = self.hour_aug[idx, 0]
        #real_start_month = self.month_aug[idx, 0]
        #real_start_response = self.response_aug[idx, 4]
        
        real_start_energy = np.expand_dims(real_start_energy, axis=1)
        real_start_temp = np.expand_dims(real_start_temp, axis = 1)
        #real_start_feelslike = np.expand_dims(real_start_feelslike, axis = 1)
        #real_start_humidity = np.expand_dims(real_start_humidity, axis = 1)
        real_start_day = np.expand_dims(real_start_day, axis = 1)
        real_start_hour = np.expand_dims(real_start_hour, axis = 1)
        #real_start_month = np.expand_dims(real_start_month, axis = 1)
        
        real_samples_energy = self.energy_return_aug[idx, :]
        real_samples_temp = self.temp_return_aug[idx, :]
        #real_samples_feelslike = self.feelslike_return_aug[idx, :]
        #real_samples_humidity = self.humidity_return_aug[idx, :]
        real_samples_day = self.day_return_aug[idx, :]
        real_samples_hour = self.hour_return_aug[idx, :]
        #real_samples_month = self.month_return_aug[idx, :]
        #real_samples_response = self.response_return_aug[idx, :]
        
        real_samples_energy = np.expand_dims(real_samples_energy, axis=1)
        real_samples_temp = np.expand_dims(real_samples_temp, axis = 1)
        #real_samples_feelslike = np.expand_dims(real_samples_feelslike, axis = 1)
        #real_samples_humidity = np.expand_dims(real_samples_humidity, axis = 1)
        real_samples_day = np.expand_dims(real_samples_day, axis = 1)
        real_samples_hour = np.expand_dims(real_samples_hour, axis = 1)
        #real_samples_month = np.expand_dims(real_samples_month, axis = 1)
        #real_samples_response = np.expand_dims(real_samples_response, axis = 1)
        
        real_samples = torch.from_numpy(np.concatenate((real_samples_energy, real_samples_temp, real_samples_day, real_samples_hour), axis=1))
        
        real_start_samples = np.concatenate((real_start_energy, real_start_temp, real_start_day, real_start_hour), axis=1)
        
        return real_samples.float(), real_start_samples, idx
        
    def get_samples(self, G, latent_dim, n, ts_dim, conditional, use_cuda):
        noise = torch.randn((n,4,latent_dim))
        idx = np.random.randint(self.energy_return_aug.shape[0], size=n)
        
        real_start_energy = self.energy_aug[idx, 0]
        real_start_temp = self.temp_aug[idx, 0]
        #real_start_feelslike = self.feelslike_aug[idx, 0]
        #real_start_humidity = self.humidity_aug[idx, 0]
        real_start_day = self.day_aug[idx, 0]
        real_start_hour = self.hour_aug[idx, 0]
        #real_start_month = self.month_aug[idx, 0]
        
        real_start_energy = np.expand_dims(real_start_energy, axis=1)
        real_start_temp = np.expand_dims(real_start_temp, axis = 1)
        #real_start_feelslike = np.expand_dims(real_start_feelslike, axis = 1)
        #real_start_humidity = np.expand_dims(real_start_humidity, axis = 1)
        real_start_day = np.expand_dims(real_start_day, axis = 1)
        real_start_hour = np.expand_dims(real_start_hour, axis = 1)
        
        real_samples_energy = self.energy_return_aug[idx, :]
        real_samples_temp = self.temp_return_aug[idx, :]
        #real_samples_feelslike = self.feelslike_return_aug[idx, :]
        #real_samples_humidity = self.humidity_return_aug[idx, :]
        real_samples_day = self.day_return_aug[idx, :]
        real_samples_hour = self.hour_return_aug[idx, :]
        #real_samples_month = self.month_return_aug[idx, :]
        
        real_samples_energy = np.expand_dims(real_samples_energy, axis=1)
        real_samples_temp = np.expand_dims(real_samples_temp, axis = 1)
        #real_samples_feelslike = np.expand_dims(real_samples_feelslike, axis = 1)
        #real_samples_humidity = np.expand_dims(real_samples_humidity, axis = 1)
        real_samples_day = np.expand_dims(real_samples_day, axis = 1)
        real_samples_hour = np.expand_dims(real_samples_hour, axis = 1)
        #real_samples_month = np.expand_dims(real_samples_month, axis = 1)
        
        real_samples = torch.from_numpy(np.concatenate((real_samples_energy, real_samples_temp, real_samples_day, real_samples_hour), axis=1))
        
        real_start_samples = np.concatenate((real_start_energy, real_start_temp, real_start_day, real_start_hour), axis=1)
        
        if conditional>0:
            noise[:,:,:conditional] = real_samples[:,:,:conditional]

        if use_cuda:
            noise = noise.cuda()
            real_samples = real_samples.cuda()
            G.cuda()

        y = G(noise)

        y = y.float()

        y=y.view(y.size(0), 4, -1)

    
        y = torch.cat((real_samples[:,:,:conditional].float().cpu(),y.float().cpu()), dim=2)
        
        if use_cuda:
            y = y.cuda()
        return y. float(), real_samples.float(), real_start_samples
    
    def get_samples_specific(self, idx, G, latent_dim, n, ts_dim, conditional, use_cuda):
        noise = torch.randn((n,4,latent_dim))
        idx=np.full(n, idx)

        real_start_energy = self.energy_aug[idx, 0]
        real_start_temp = self.temp_aug[idx, 0]
        #real_start_feelslike = self.feelslike_aug[idx, 0]
        #real_start_humidity = self.humidity_aug[idx, 0]
        real_start_day = self.day_aug[idx, 0]
        real_start_hour = self.hour_aug[idx, 0]
        #real_start_month = self.month_aug[idx, 0]
        #real_start_response = self.response_aug[idx, 4]
        
        real_start_energy = np.expand_dims(real_start_energy, axis=1)
        real_start_temp = np.expand_dims(real_start_temp, axis = 1)
        #real_start_feelslike = np.expand_dims(real_start_feelslike, axis = 1)
        #real_start_humidity = np.expand_dims(real_start_humidity, axis = 1)
        real_start_day = np.expand_dims(real_start_day, axis = 1)
        real_start_hour = np.expand_dims(real_start_hour, axis = 1)
        #real_start_month = np.expand_dims(real_start_month, axis = 1)
        
        real_samples_energy = self.energy_return_aug[idx, :]
        real_samples_temp = self.temp_return_aug[idx, :]
        #real_samples_feelslike = self.feelslike_return_aug[idx, :]
        #real_samples_humidity = self.humidity_return_aug[idx, :]
        real_samples_day = self.day_return_aug[idx, :]
        real_samples_hour = self.hour_return_aug[idx, :]
        #real_samples_month = self.month_return_aug[idx, :]
        #real_samples_response = self.response_return_aug[idx, :]
        
        real_samples_energy = np.expand_dims(real_samples_energy, axis=1)
        real_samples_temp = np.expand_dims(real_samples_temp, axis = 1)
        #real_samples_feelslike = np.expand_dims(real_samples_feelslike, axis = 1)
        #real_samples_humidity = np.expand_dims(real_samples_humidity, axis = 1)
        real_samples_day = np.expand_dims(real_samples_day, axis = 1)
        real_samples_hour = np.expand_dims(real_samples_hour, axis = 1)
        #real_samples_month = np.expand_dims(real_samples_month, axis = 1)
        #real_samples_response = np.expand_dims(real_samples_response, axis = 1)
        
        real_samples = torch.from_numpy(np.concatenate((real_samples_energy, real_samples_temp, real_samples_day, real_samples_hour), axis=1))
        
        real_start_samples = np.concatenate((real_start_energy, real_start_temp, real_start_day, real_start_hour), axis=1)
        
        if conditional>0:
            noise[:,:,:conditional] = real_samples[:,:,:conditional]

        if use_cuda:
            noise = noise.cuda()
            real_samples = real_samples.cuda()
            G.cuda()

        y = G(noise)

        y = y.float()

        y=y.view(y.size(0), 4, -1)

    
        y = torch.cat((real_samples[:,:,:conditional].float().cpu(),y.float().cpu()), dim=2)
        
        if use_cuda:
            y = y.cuda()
        return y.float(), real_samples.float(), real_start_samples
    
    def generate_data(self, idx, G, latent_dim, ts_dim, conditional, use_cuda):
        #generate fake data interval
        noise = torch.randn((1,4,latent_dim))
        idx = np.full(1,idx) #array of dimension 1, filled with idx number
        
        real_start_energy = self.energy_aug[idx, 0]
        real_start_temp = self.temp_aug[idx, 0]
        #real_start_feelslike = self.feelslike_aug[idx, 0]
        #real_start_humidity = self.humidity_aug[idx, 0]
        real_start_day = self.day_aug[idx, 0]
        real_start_hour = self.hour_aug[idx, 0]
        #real_start_month = self.month_aug[idx, 0]
        #real_start_response = self.response_aug[idx, 4]
        
        real_start_energy = np.expand_dims(real_start_energy, axis=1)
        real_start_temp = np.expand_dims(real_start_temp, axis = 1)
        #real_start_feelslike = np.expand_dims(real_start_feelslike, axis = 1)
        #real_start_humidity = np.expand_dims(real_start_humidity, axis = 1)
        real_start_day = np.expand_dims(real_start_day, axis = 1)
        real_start_hour = np.expand_dims(real_start_hour, axis = 1)
        #real_start_month = np.expand_dims(real_start_month, axis = 1)
        
        real_samples_energy = self.energy_return_aug[idx, :]
        real_samples_temp = self.temp_return_aug[idx, :]
        #real_samples_feelslike = self.feelslike_return_aug[idx, :]
        #real_samples_humidity = self.humidity_return_aug[idx, :]
        real_samples_day = self.day_return_aug[idx, :]
        real_samples_hour = self.hour_return_aug[idx, :]
        #real_samples_month = self.month_return_aug[idx, :]
        #real_samples_response = self.response_return_aug[idx, :]
        
        real_samples_energy = np.expand_dims(real_samples_energy, axis=1)
        real_samples_temp = np.expand_dims(real_samples_temp, axis = 1)
        #real_samples_feelslike = np.expand_dims(real_samples_feelslike, axis = 1)
        #real_samples_humidity = np.expand_dims(real_samples_humidity, axis = 1)
        real_samples_day = np.expand_dims(real_samples_day, axis = 1)
        real_samples_hour = np.expand_dims(real_samples_hour, axis = 1)
        #real_samples_month = np.expand_dims(real_samples_month, axis = 1)
        #real_samples_response = np.expand_dims(real_samples_response, axis = 1)
        
        real_samples = torch.from_numpy(np.concatenate((real_samples_energy, real_samples_temp, real_samples_day, real_samples_hour), axis=1))
        
        real_start_samples = np.concatenate((real_start_energy, real_start_temp, real_start_day, real_start_hour), axis=1)
        
        if conditional>0:
            noise[:,:,:conditional] = real_samples[:,:,:conditional]

        if use_cuda:
            noise = noise.cuda()
            real_samples = real_samples.cuda()
            G.cuda()

        y = G(noise)

        y = y.float()

        y=y.view(y.size(0), 4, -1)

    
        y = torch.cat((real_samples[:,:,:conditional].float().cpu(),y.float().cpu()), dim=2)
        
        if use_cuda:
            y = y.cuda()
            
        #data post processing (un-normalize the generated fake data)
    
        real=np.squeeze(real_samples.float().cpu().data.numpy())
        fake=np.squeeze(y.float().cpu().data.numpy())
        
        fake=fake.T
        
        fake_data = self.scalar2.inverse_transform(fake)
        #print(np.max(return_data))
        fake_data = fake_data*np.exp(0.5*self.delta*fake_data**2)
        fake_data = self.scalar.inverse_transform(fake_data)
        fake_data = np.exp(fake_data)
        
        real=real.T
        real_data = self.scalar2.inverse_transform(real)
        #print(np.max(return_data))
        real_data = real_data*np.exp(0.5*self.delta*real_data**2)
        real_data = self.scalar.inverse_transform(real_data)
        real_data = np.exp(real_data)
        
        #generate real and fake energy
        arrayno=0
        fake_energy = fake_data[:,arrayno]  
        fake_energy = fake_energy.reshape(-1,1)
        
        fake_energy_return = np.empty((fake_energy.shape[0],))
        fake_energy_return[0] = real_start_samples[0,arrayno]
        for i in range(1,fake_energy.shape[0]):
            fake_energy_return[i] = fake_energy_return[i-1] * fake_energy[i]
            
        real_energy = real_data[:,arrayno]  
        real_energy = real_energy.reshape(-1,1)
        
        real_energy_return = np.empty((real_energy.shape[0],))
        real_energy_return[0] = real_start_samples[0,arrayno]
        for i in range(1,real_energy.shape[0]):
            real_energy_return[i] = real_energy_return[i-1] * real_energy[i]
            
        arrayno=1
        fake_temp = fake_data[:,arrayno] 
        fake_temp = fake_temp.reshape(-1,1)
        
        fake_temp_return = np.empty((fake_temp.shape[0],))
        fake_temp_return[0] = real_start_samples[0,arrayno]
        for i in range(1,fake_temp.shape[0]):
            fake_temp_return[i] = fake_temp_return[i-1] * fake_temp[i]
            
        real_temp = real_data[:,arrayno] 
        real_temp = real_temp.reshape(-1,1)
        
        real_temp_return = np.empty((real_temp.shape[0],))
        real_temp_return[0] = real_start_samples[0,arrayno]
        for i in range(1,real_temp.shape[0]):
            real_temp_return[i] = real_temp_return[i-1] * real_temp[i]
        
        return fake_energy_return, fake_temp_return, real_energy_return, real_temp_return