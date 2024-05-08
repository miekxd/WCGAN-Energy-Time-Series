from data import Data

import torch
import numpy as np
import os
import json
from torch.autograd import grad as torch_grad
from eval import plt_loss, plt_progress, plt_gp, plt_lr
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
import os

def aws_s3_sync(source, destination):
    """aws s3 sync in quiet mode and time profile"""
    import time, subprocess
    cmd = ["aws", "s3", "sync", "--quiet", source, destination]
    print(f"Syncing files from {source} to {destination}")
    start_time = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    end_time = time.time()
    print("Time Taken to Sync: ", (end_time-start_time))
    return

def sync_local_checkpoints_to_s3(local_path="/opt/ml/checkpoints", s3_uri=os.path.dirname(os.path.dirname(os.getenv('SM_MODULE_DIR', '')))+'/checkpoints'):
    """ sample function to sync checkpoints from local path to s3 """

    import boto3
    #check if local path exists
    if not os.path.exists(local_path):
        raise RuntimeError("Provided local path {local_path} does not exist. Please check")

    #check if s3 bucket exists
    s3 = boto3.resource('s3')
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Provided s3 uri {s3_uri} is not valid.")

    s3_bucket = s3_uri.replace('s3://','').split('/')[0]
    print(f"S3 Bucket: {s3_bucket}")
    try:
        s3.meta.client.head_bucket(Bucket=s3_bucket)
    except Exception as e:
        raise e
    aws_s3_sync(local_path, s3_uri)
    return

def checkpoint_model(checkpoint, model, optimizer, epoch, typ):
    model_path = checkpoint+f"/{typ}.pt"
    torch.save({
        'epoch': epoch,
        f'{typ}_state_dict': model.state_dict(),
        f'{typ}_opt_state_dict': optimizer.state_dict()
    }, model_path)
    logging.info(f"{typ}{epoch} saved at {model_path}")
    return
    
class Trainer():

    def __init__(self, args, generator, critic, gen_optimizer, critic_optimizer, path, D_scheduler, G_scheduler, gp_weight=10,critic_iter=5, n_eval=100, use_cuda=False):
        self.G = generator
        self.D = critic
        self.G_opt = gen_optimizer
        self.D_opt = critic_optimizer
        self.G_scheduler = G_scheduler
        self.D_scheduler = D_scheduler
        self.batch_size = args.batch_size
        self.scorepath = path
        self.gp_weight = gp_weight
        self.critic_iter = critic_iter
        self.n_eval = n_eval
        self.use_cuda = use_cuda
        self.conditional = args.conditional
        self.ts_dim = args.ts_dim
        train_load_path ='/opt/ml/input/data/training/train.csv'
        #test_load_path = '/opt/ml/input/data/training/test.csv'
        self.data = Data(self.ts_dim, train_load_path)
        #self.test = Data(self.ts_dim, test_load_path)

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()


        
        self.latent_dim = args.latent_dim
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': [], 'LR_G': [], 'LR_D':[]}

    def train(self, epochs, cepoch):
        plot_num=0
        logger.info("Training starts ...")
        for epoch in range(cepoch,epochs):
            for i in range(self.critic_iter):
                # train the critic
                fake_batch, real_batch, start_features = self.data.get_samples(G=self.G, latent_dim=self.latent_dim, n=self.batch_size, ts_dim=self.ts_dim,conditional=self.conditional, use_cuda=self.use_cuda)
                if self.use_cuda:
                    real_batch = real_batch.cuda()
                    fake_batch = fake_batch.cuda()
                    self.D.cuda()
                    self.G.cuda()
                
                d_real = self.D(real_batch)
                d_fake = self.D(fake_batch)

                grad_penalty, grad_norm_ = self._grad_penalty(real_batch, fake_batch)
                # backprop with minimizing the difference between distribution fake and distribution real
                self.D_opt.zero_grad()
                 
                d_loss = d_fake.mean() - d_real.mean() + grad_penalty.to(torch.float32) 
                d_loss.backward()
                self.D_opt.step()
                
                if i == self.critic_iter-1:
                    self.D_scheduler.step()
                    self.losses['LR_D'].append(self.D_scheduler.get_lr())
                    self.losses['D'].append(float(d_loss))
                    self.losses['GP'].append(grad_penalty.item())
                    self.losses['gradient_norm'].append(float(grad_norm_))
            
            self.G_opt.zero_grad()
            fake_batch_critic, real_batch_critic, start_features = self.data.get_samples(G=self.G, latent_dim=self.latent_dim, n=self.batch_size, ts_dim=self.ts_dim,conditional=self.conditional, use_cuda=self.use_cuda)
            if self.use_cuda:
                real_batch_critic = real_batch_critic.cuda()
                fake_batch_critic = fake_batch_critic.cuda()
                self.D.cuda()
                self.G.cuda()
            # feed-forward
            d_critic_fake = self.D(fake_batch_critic)

            g_loss =  - d_critic_fake.mean()  # d_critic_real.mean()
            # backprop
            g_loss.backward()
            self.G_opt.step()
            self.G_scheduler.step()
            self.losses['LR_G'].append(self.G_scheduler.get_lr())
    
            # save the loss of feed forward
            self.losses['G'].append(g_loss.item())  
            
            if (epoch + 1) % self.n_eval == 0:
                
                # Generate synthetic energy time series
                fake_lines, real_lines, start_features = self.data.get_samples(G=self.G, latent_dim=self.latent_dim, n=4, ts_dim=self.ts_dim, conditional=self.conditional, use_cuda=self.use_cuda)
                
                # Convert tensor to numpy arrays
                real_lines = np.squeeze(real_lines.cpu().data.numpy())
                fake_lines = np.squeeze(fake_lines.cpu().data.numpy())
                #print("real_lines shape: ", real_lines.shape)
                
                real_lines_sliced = real_lines[:,:,self.conditional:]
                fake_lines_sliced = fake_lines[:,:,self.conditional:]
                #print("real_lines_sliced shape: ", real_lines_sliced.shape)
            
                # Compute RMSE
                rmse = np.sqrt(np.mean((real_lines_sliced - fake_lines_sliced)**2))
                energy_rmse = np.sqrt(np.mean((real_lines_sliced[:,0] - fake_lines_sliced[:,0])**2))
                
                # Log or print RMSE
                logger.info(f"Epoch: {epoch}, "
                            f"RMSE: {rmse}, "
                            f"Energy RMSE: {energy_rmse}, "
                            f"G_loss: {g_loss.item()}, " 
                            f"D_loss: {d_loss.item()} ")
                    
                if (epoch+1) % 1000 == 0: #ploting training metrics during training
                    plot_num = plot_num+1
                plt_loss(self.losses['G'], self.losses['D'], self.scorepath, plot_num)
                plt_gp(self.losses['gradient_norm'], self.losses['GP'], self.scorepath)
                plt_lr(self.losses['LR_G'],self.losses['LR_D'], self.scorepath)
                
            if (epoch + 1) % (10*self.n_eval) == 0: #model evaluation plots during training
                fake_lines, real_lines, start_features = self.data.get_samples(G=self.G, latent_dim=self.latent_dim, n=4, ts_dim=self.ts_dim,conditional=self.conditional, use_cuda=self.use_cuda)

                real_lines = np.squeeze(real_lines.cpu().data.numpy())
                fake_lines = np.squeeze(fake_lines.cpu().data.numpy())
                real_lines = np.array([self.data.post_processing(real_lines[i], start_features[i]) for i in range(real_lines.shape[0])])
                fake_lines = np.array([self.data.post_processing(fake_lines[i], start_features[i]) for i in range(real_lines.shape[0])])
                
                plt_progress(real_lines, fake_lines, epoch, self.scorepath)
                
            if (epoch + 1) % 500 ==0: #model checkpointing
                name = 'WCGAN'
                checkpoint = '/opt/ml/checkpoints'
                checkpoint_model(checkpoint, self.G, self.G_opt, epoch, "G")
                checkpoint_model(checkpoint, self.D, self.D_opt, epoch, "D")   


    def _grad_penalty(self, real_data, gen_data):
        batch_size = real_data.size()[0]
        t = torch.rand((batch_size, 1, 1), requires_grad=True)
        t = t.expand_as(real_data)

        if self.use_cuda:
            t = t.cuda()

        # mixed sample from real and fake; make approx of the 'true' gradient norm
        interpol = t * real_data.data + (1-t) * gen_data.data

        if self.use_cuda:
            interpol = interpol.cuda()
        
        prob_interpol = self.D(interpol)
        torch.autograd.set_detect_anomaly(True)
        gradients = torch_grad(outputs=prob_interpol, inputs=interpol,
                               grad_outputs=torch.ones(prob_interpol.size()).cuda() if self.use_cuda else torch.ones(
                                   prob_interpol.size()), create_graph=True, retain_graph=True)[0]
        gradients = gradients.reshape(batch_size, -1)
        #grad_norm = torch.norm(gradients, dim=1).mean()
        #self.losses['gradient_norm'].append(grad_norm.item())

        # add epsilon for stability
        eps = 1e-10
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1, dtype=torch.double) + eps)
        #gradients = gradients.cpu()
        # comment: precision is lower than grad_norm (think that is double) and gradients_norm is float
        return self.gp_weight * (torch.max(torch.zeros(1,dtype=torch.double).cuda() if self.use_cuda else torch.zeros(1,dtype=torch.double), gradients_norm.mean() - 1) ** 2), gradients_norm.mean().item()
