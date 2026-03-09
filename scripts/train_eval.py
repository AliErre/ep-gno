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

def gen_meta_info(batch_size, dims, query_dims, mesh=None, n_pts=None, source_pos=None):
    """
    to use at inference for generation of a noise field at n_pts subsampled from meshgrid with dimension dims
    the output is the starting point of integration
    if source_pos is given, no sampling of the mesh happens
    source_pos should be [n_pts, dim]
    """
    if source_pos is None:
        assert mesh is not None and n_pts is not None, 'if source coordinates are not given, need to sample them'
        x = point_cloud_coords(n_pts, dims, mesh)   # [n_points, 3]
        # pos_data is the same for all samples in the batch
        pos_data = x.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, n_pts, dim]
    else:
        pos_data = source_pos.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, n_pts, dim]
    query_n_pos = make_grid(query_dims) # change this function probably
    query_pos_data = query_n_pos.unsqueeze(0).repeat(batch_size, 1, 1)
    collated_batch = {}
    collated_batch['input_pos'] = pos_data.permute(0, 2, 1) # source x [bs, dim, n_pts]
    collated_batch['query_pos'] = query_pos_data.permute(0, 2, 1) # query y [bs, dim, n_y]

    return collated_batch

def main():

    # dataloader
    # final tensor of shape [Batch, Channels, N_Points]
    field_train = torch.load() # CHANGE
    condition_train = torch.load() # CHANGE
    mesh = torch.load() # CHANGE. read mesh
    x = point_cloud_coords(args.n_pts_train, args.dims, mesh) # [n_pts, dim] source train. consider varying the locations (and their number) for different batches
    x_data = x.unsqueeze(0).repeat(args.batch_size, 1, 1) # source pos [batch_size, n_pts, dim]
    assert args.dims == args.query_dims, "source and query coordinates must have the same dimensionality."
    query_pos = make_grid(args.query_dims) # y = query train. should be same for testing?
    train_dataset = SimDataset(field_train, x_data, query_pos, conditioning=condition_train)
    loader_tr = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=SimulationCollator)

    field_val = torch.load() # CHANGE
    condition_val = torch.load() # CHANGE
    val_dataset = SimDataset(field_val, x_data, query_pos, conditioning=condition_val)
    loader_va = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=SimulationCollator)

    field_test = torch.load() # CHANGE
    condition_test = torch.load() # CHANGE
    test_dataset = SimDataset(field_test, x_data, query_pos, conditioning=condition_test)
    test_bs = 50
    loader_te = DataLoader(test_dataset, batch_size=test_bs, shuffle=False, collate_fn=SimulationCollator)
    print('Dataloading is over')

    conditioner = ConditionerTimestep(dim=args.dim)
    model = GeoLearn(conditioner = conditioner,
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
        for param in model.parameters():
            param.requires_grad = False
        model_path = os.path.join() # complete with best model path
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint, strict = False)
        fmot = OTFuncFlowMatcherModel(model, kernel_length=args.kernel_length, kernel_variance=args.kernel_variance,
                                      nu=args.nu, sigma_min=args.sigma_min, device=args.device, x_dim=args.x_dim, n_pos=x)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        fmot = OTFuncFlowMatcherModel(model, kernel_length=args.kernel_length, kernel_variance=args.kernel_variance,
                                      nu=args.nu, sigma_min=args.sigma_min, device=args.device, x_dim=args.x_dim, n_pos=x)
        fmot.train(
            train_loader=loader_tr,    # train_loader should be contain batches that are dicts and one key should be conditioning
            optimizer=optimizer,         
            epochs=args.epochs,
            scheduler=scheduler,
            test_loader=loader_va,
            eval_int=int(1),           # How often to run validation
            save_int=args.epochs,      # How often to save checkpoints
            save_path=spath,           # Path object/string for logs and .pt files
            saved_model=True           # Triggers the saving logic
        )
        print("Training is over, start evaluating")
    gen = []
    start = time.time()
    with torch.no_grad():
        for batch in loader_te:
            conditioning = batch['conditioning'].to(args.device)
            collated_batch = gen_meta_info(batch_size = len(batch), dims=args.dims, query_dims=args.query_dims, source_pos=x)
            # probably unnecessary given what is in test loader but I am trying to replicate MINO repo
            pos, query_pos = collated_batch['input_pos'], collated_batch['query_pos']
            X_int = fmot.sample(pos=pos.to(args.device), query_pos=query_pos.to(args.device), condition=conditioning, n_channels=args.codomain, n_samples=test_bs, n_eval=2).cpu()
            gen.append(X_int)
        gen = torch.vstack(gen).squeeze()

    torch.save(gen, spath / 'generated_activation_maps.pt')
    print(f"Generated {gen.shape[0]} samples in {time.time() - start:.2f}s and saved to {spath}")






