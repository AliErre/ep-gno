import numpy as np
import torch
from torchdiffeq import odeint
#from util.gaussian_process import GPPrior
from util.true_gaussian_process_seq import true_GPPrior
from torchcfm.optimal_transport import OTPlanSampler
from util.util import reshape_for_batchwise, plot_loss_curve
import time
from torch.utils.tensorboard import SummaryWriter
# remember that torchcfm actually never samples t=1. Might want to change this later, also adding stratified sampling of times
class OTFuncFlowMatcherModel:
    def __init__(self, model, kernel_length=0.01, kernel_variance=1.0, nu=0.5, sigma_min=1e-4, device='cpu',
                 dtype=torch.double, x_dim=None, n_pos=None):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.n_pos = n_pos
        self.gp = true_GPPrior(lengthscale=kernel_length, var=kernel_variance, nu=nu, device=device, x_dim=x_dim, n_pos=n_pos)
        self.ot_sampler = OTPlanSampler(method='exact')
        self.sigma_min = sigma_min
    
    def sample_gp_noise(self, x_data):
        """
        x_data here refers to the field(s) at source positions
        """
        # sample GP noise with OT
        batch_size = x_data.shape[0]
        n_channels = x_data.shape[1] # codomain cardinality
        x_0 = self.gp.sample_from_prior(n_samples=batch_size, n_channels=n_channels)
        x_0, x_data = self.ot_sampler.sample_plan(x_0, x_data)

        return x_0
    
    def simulate(self, t, x_0, x_data): #x_t
        # t: [batch_size,]
        # x_data: [batch_size, n_channels, n_point]
        # samples from p_t(x | x_data)

        batch_size = x_data.shape[0]
        n_channels = x_data.shape[1]
        # sample from prior GP
        noise = self.gp.sample_from_prior(n_samples=batch_size, n_channels=n_channels) #eps?
        # mean and variance at t along the exact OT path (Tong et al. 2023)
        t = reshape_for_batchwise(t, 2) # [bs, n_chan, n_pos]
        mu_t = t * x_data + (1 - t) * x_0
        x_t = mu_t + self.sigma_min * noise # sigma_t = sigma_min (constant in this path)
        assert x_t.shape == x_data.shape

        return x_t
    
    def get_conditional_target_fields(self, x0, x1): # normally should depend on t but not for exact OT
        return x1 - x0
    
def train(self, train_loader, optimizer, epochs, scheduler=None, test_loader=None, eval_int=1,
              save_int=0, generate=False, save_path=None, saved_model=False):
        
        tr_losses = []
        te_losses = []
        eval_eps = []
        best_te_loss = float('inf')
        
        # Initialize TensorBoard Writer
        writer = SummaryWriter(log_dir=str(save_path / 'logs')) if save_path else None
        
        model = self.model
        device = self.device

        for ep in range(1, epochs + 1):
            ##### Training Phase #####
            t0 = time.time()
            model.train()
            tr_loss = 0.0

            for batch_pack in train_loader:
                batch = batch_pack['input_feat'].to(device)
                pos = batch_pack['input_pos'].to(device)
                query_pos = batch_pack['query_pos'].to(device)
                conditioning = batch_pack['conditioning'].to(device) if 'conditioning' in batch_pack else None
                
                x_0, x_data = self.sample_gp_noise(batch)
                
                # Stratified sampling
                batch_size = batch.shape[0]
                edges = torch.linspace(0, 1, batch_size + 1, device=device)
                t = edges[:-1] + torch.rand(batch_size, device=device) * (1.0 / batch_size)
                t[-1] = 1.0
                t = t[torch.randperm(batch_size)]

                x_t = self.simulate(t, x_0, x_data)
                target_velocity = self.get_conditional_target_fields(x_0, x_data)
                
                optimizer.zero_grad()
                pred_velocity = model(f_x=x_t, source_pos=pos, query_pos=query_pos, time_condition=t, conditioning=conditioning)
                
                loss = torch.mean((pred_velocity - target_velocity) ** 2)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()

            tr_loss /= len(train_loader)
            tr_losses.append(tr_loss)
            
            # Log Training Loss to TensorBoard
            if writer:
                writer.add_scalar('Loss/train', tr_loss, ep)
            
            if scheduler:
                scheduler.step()

            ##### Validation Phase #####
            te_loss = 0.0
            if test_loader is not None and ep % eval_int == 0:
                model.eval()
                with torch.no_grad():
                    for batch_pack in test_loader:
                        batch = batch_pack['input_feat'].to(device)
                        pos = batch_pack['input_pos'].to(device)
                        query_pos = batch_pack['query_pos'].to(device)
                        conditioning = batch_pack['conditioning'].to(device) if 'conditioning' in batch_pack else None
                        
                        x_0, x_data = self.sample_gp_noise(batch)
                        
                        batch_size = batch.shape[0]
                        edges = torch.linspace(0, 1, batch_size + 1, device=device)
                        t = edges[:-1] + torch.rand(batch_size, device=device) * (1.0 / batch_size)
                        t[-1] = 1.0
                        
                        x_t = self.simulate(t, x_0, x_data)
                        target_velocity = self.get_conditional_target_fields(x_0, x_data)
                        
                        pred_velocity = model(f_x=x_t, source_pos=pos, query_pos=query_pos, time_condition=t, conditioning=conditioning)
                        te_loss += torch.mean((pred_velocity - target_velocity) ** 2).item()
                
                te_loss /= len(test_loader)
                te_losses.append(te_loss)
                eval_eps.append(ep)
                
                # Log Validation Loss to TensorBoard
                if writer:
                    writer.add_scalar('Loss/test', te_loss, ep)

            t1 = time.time()
            print(f'Epoch {ep}/{epochs} | Tr Loss: {tr_loss:.6f} | Te Loss: {te_loss:.6f} | Time: {t1-t0:.2f}s')

            ##### Bookkeeping & Saving #####
            if save_path:
                # 1. Plotting: Only pass te_loss if it actually contains data
                current_te_loss = te_losses if len(te_losses) > 0 else None
                current_te_eps = eval_eps if len(eval_eps) > 0 else None
                
                plot_loss_curve(tr_losses, save_path / 'loss.pdf', 
                                te_loss=current_te_loss, 
                                te_epochs=current_te_eps)

                if saved_model:
                    # 2. Best Model: Only compare if we just performed a validation
                    # This prevents saving a 'best' model before any validation has happened
                    if len(te_losses) > 0 and te_losses[-1] < best_te_loss:
                        best_te_loss = te_losses[-1]
                        torch.save(model.state_dict(), save_path / 'best_model.pt')
                        print(f"--> New Best Model saved at Epoch {ep} (Val Loss: {best_te_loss:.6f})")

                    # Save regular checkpoints
                    if save_int > 0 and ep % save_int == 0:
                        torch.save(model.state_dict(), save_path / f'epoch_{ep}.pt')
                        
        if writer:
            writer.close()