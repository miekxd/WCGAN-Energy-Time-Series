import torch
import torch.optim as optim
from dense3_skip2 import Generator, Discriminator
from training import Trainer
from torch.autograd import Variable
import os
import datetime
import logging
import argparse
import json
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--ts_dim", type=int, default=24)
    parser.add_argument("--conditional", type=int, default=2)
    parser.add_argument("--lr_a", type=float, default=0.00004)
    parser.add_argument("--lr_b", type=float, default=0.00004)

    return parser.parse_args()

def save_model(model_dir, model, typ):
    model.save(model_dir, filename=typ+".pth")

def is_empty(path):
    try:
        entries = os.listdir(path)
        print(entries)
    except FileNotFoundError:
        return False
    return len(entries) == 0
    
def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path+f'/{model}.pt')

    model_state_dict = checkpoint[f'{model}_state_dict']

    model_opt_state_dict = checkpoint.get(f'{model}_opt_state_dict', None)

    # Extract epoch number
    epoch = checkpoint['epoch']

    return model_state_dict, model_opt_state_dict, epoch

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

def sync_s3_checkpoints_to_local(local_path="/opt/ml/checkpoints", s3_uri=os.path.dirname(os.path.dirname(os.getenv('SM_MODULE_DIR', '')))+'/checkpoints'):
    """ sample function to sync checkpoints from s3 to local path """

    import boto3
    #try to create local path if it does not exist
    if not os.path.exists(local_path):
        print(f"Provided local path {local_path} does not exist. Creating...")
        try:
            os.makedirs(local_path)
        except Exception as e:
            raise RuntimeError(f"Failed to create {local_path}")

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
    aws_s3_sync(s3_uri, local_path)
    return

if __name__ == "__main__":
    betas = (0, 0.9)
    args=parse_args()
    checkpoint_path = '/opt/ml/checkpoints'
    sync_s3_checkpoints_to_local()
    
    
    scorepath = "/opt/ml/output/data/" + "{}-{}-{}-{}-{}-{}-{}".format(args.batch_size, args.epochs, args.latent_dim, args.ts_dim, args.conditional, args.lr_a, args.lr_b)

    plot_scorepath = scorepath +"/line_generation"
    if not os.path.exists(scorepath):
        os.makedirs(scorepath)
        os.makedirs(plot_scorepath)


    G = Generator(latent_dim=args.latent_dim, ts_dim=args.ts_dim,condition=args.conditional)
    D = Discriminator(ts_dim=args.ts_dim)
    
    G_opt = optim.RMSprop(G.parameters(), lr=args.lr_a)
    D_opt = optim.RMSprop(D.parameters(), lr=args.lr_b)

    D_scheduler = optim.lr_scheduler.CyclicLR(D_opt, base_lr = 1e-4, max_lr= 8e-4, step_size_up=100, step_size_down=900, mode ='triangular')
    G_scheduler = optim.lr_scheduler.CyclicLR(G_opt, base_lr = 1e-4, max_lr= 6e-4, step_size_up=100, step_size_down=900, mode ='triangular')
    epoch=0
    
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    
    if is_empty(checkpoint_path) == False:
        print('is empty false')
        generator_state_dict, generator_opt_state_dict, epoch = load_checkpoint(checkpoint_path, model="G")
        discriminator_state_dict, discriminator_opt_state_dict, _ = load_checkpoint(checkpoint_path, model="D")
        print("Last checkpoint epoch: ",epoch)

        G.load_state_dict(generator_state_dict)
        D.load_state_dict(discriminator_state_dict)
        
        if use_cuda:
            G.cuda()
            D.cuda()

        if generator_opt_state_dict is not None:
            G_opt = optim.RMSprop(G.parameters(), lr=args.lr_a)
            G_opt.load_state_dict(generator_opt_state_dict)

        if discriminator_opt_state_dict is not None:
            D_opt = optim.RMSprop(D.parameters(), lr=args.lr_b)
            D_opt.load_state_dict(discriminator_opt_state_dict)
        
    
    train = Trainer(args, G, D, G_opt, D_opt, scorepath, D_scheduler, G_scheduler, use_cuda=use_cuda)
    train.train(epochs=args.epochs, cepoch=epoch)

    save_model(scorepath, G, "gen")
    save_model(scorepath, D, "dis")