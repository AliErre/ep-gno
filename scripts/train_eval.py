import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

## reduce the batch_size to fit the GPU memoery, batch_size=96 ~ 40 GB memoery

## load path
import os
import sys
sys.path.append('./')
sys.path.append('./models')
from pathlib import Path

## load utils 
from util.util import *
from util.true_gaussian_process_seq import *
from models.flowMatcher import OTFuncFlowMatcherModel
from util.metrics import *
import time

## load modules 
from models.GeoLearn import GeoLearn
from models.decoder import DecoderPerceiver
from models.encoder import Encoder
from models.embeddings import ConditionerTimestep

import argparse

parser = argparse.ArgumentParser(description='Train and evaluate conditional functional flow matcher with exact OT.')
parser.add_argument('--model', type=str)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--saved_model', type=int, default=1)
parser.add_argument('--data_path', type=str, default='./data/')
parser.add_argument('--save_path', type=str, default='./saved_models/')

# data parameters
parser.add_argument('--x_dim', type=int, default=3)
parser.add_argument('--dims', type=int, nargs='+', default=[64, 64])
parser.add_argument('--query_dims', type=int, nargs='+', default=[16, 16]) # ?
parser.add_argument('--codomain', type=int, default=1)
parser.add_argument('--radius', type=float)
parser.add_argument('--n_pts_train', type=int, default=1024) # number of points in the point cloud representation of the field. consider varying this for different batches

# GP hyperparams
parser.add_argument('--kernel_length', type=float, default=0.01)
parser.add_argument('--kernel_variance', type=float, default=1.0)
parser.add_argument('--nu', type=float, default=0.5)
parser.add_argument('--sigma_min', type=float, default=1e-4)

# training and scheduler
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--step_size', type=int, default=25) # ?
parser.add_argument('--gamma', type=float, default=0.8)
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int, default=96)
parser.add_argument('--eval', type=int, default=0, help='inference mode') # 1 for generation

# model hyperparams
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--enc_depth', type=int)
parser.add_argument('--dec_depth', type=int)
parser.add_argument('--transform_type', type=str, default='nonlinear') # 'linear', 'nonlinear', 'nonlinear_kernelonly
args = parser.parse_args()
data_path = args.data_path # + ... complete
data_test_path = args.data_path # + ... complete
spath = Path(args.save_path)
spath.mkdir(parents=True, exist_ok=True)

def gen_meta_info(batch_size, n_pts, dims, query_dims):
    """
    to use at inference for generation of a field at n_pts subsampled from meshgrid with dimension dims
    """
    n_pos = point_cloud_coords(n_pts, dims) # [n_points, 3]
    # pos_data is the same for all samples in the batch
    pos_data = n_pos.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, n_pts, dim]
    query_n_pos = make_grid(query_dims) # change this function probably
    query_pos_data = query_n_pos.unsqueeze(0).repeat(batch_size, 1, 1)
    collated_batch = {}
    collated_batch['input_pos'] = pos_data.permute(0, 2, 1) # source x
    collated_batch['query_pos'] = query_pos_data.permute(0, 2, 1) # query y

    return collated_batch

def main():

    # dataloader
    # final tensor of shape [Batch, Channels, N_Points]
    field_train = torch.load() # CHANGE
    condition_train = torch.load() # CHANGE
    mesh = torch.load() # CHANGE. read mesh
    x = point_cloud_coords(args.n_pts_train, args.dims, mesh) # source train. consider varying the locations (and their number) for different batches
    x_data = x.unsqueeze(0).repeat(args.batch_size, 1, 1) # [batch_size, n_pts, dim]
    assert args.dims == args.query_dims, "source and query coordinates must have the same dimensionality."
    query_pos = make_grid(args.query_dims) # y = query train. should be same for testing?
    
    train_dataset = SimDataset(field_train, x_data, query_pos, conditioning=condition_train)
    loader_tr = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=SimulationCollator)

    field_test = torch.load() # CHANGE
    condition_test = torch.load() # CHANGE
    print('Dataloading is over')

    conditioner = ConditionerTimestep(dim=args.dim)
    model = OTFuncFlowMatcherModel(conditioner = conditioner,
                                   encoder = Encoder(input_dim=args.codomain,
                                                     ndim=args.x_dim,
                                                     radius=args.radius,
                                                     transform_type=args.transform_type,
                                                     enc_dim=args.dim,
                                                     enc_depth=args.enc_depth,
                                                     enc_num_heads=args.n_heads,
                                                     cond_dim=conditioner.cond_dim),
                                   decoder = DecoderPerceiver(input_dim=args.dim,
                                                              output_dim=args.codomain,
                                                              ndim=args.x_dim,
                                                              dim=args.dim,
                                                              depth=args.dec_depth,
                                                              num_heads=args.n_heads,
                                                              unbatch_mode='dense_to_sparse_unpadded',
                                                              cond_dim=conditioner.cond_dim))
    model = model.to(args.device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    if args.eval:
        # skip training
        print('start evaluating')




