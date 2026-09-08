import torch
from omegaconf import OmegaConf
from zoodatasets.base_datasets import ZooDataset
from utils.util import instantiate_from_config

def test_pipeline():
    print("1. Loading minimal dataset...")
    # topk=2 ensures we only load a tiny fraction of weights for instant testing
    dataset = ZooDataset(datapath='modelzoos/zoo_config.yaml', dataset='qwen_models', topk=2, length=640, n_tok=16)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    batch = next(iter(dataloader))
    
    print("2. Initializing model...")
    config = OmegaConf.load('vit_vae/configs/base_vit_vae_config.yaml')
    model = instantiate_from_config(config.model).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.deterministic = True # Forcing Stage 1 deterministic behavior
    
    print("3. Testing Forward and Backward pass...")
    optimizer = model.configure_optimizers()
    inputs = batch['weight'].to(model.device)
    mask = batch['mask'].to(model.device)
    
    # Forward
    x, dec, mu, logvar = model(inputs)
    # Loss
    loss, logs = model.compute_loss(dec, x, mu, logvar, kl_weight=0.0, mask=mask)
    
    # Backward
    loss.backward()
    optimizer.step()
    print(f"Success! Test complete. Final Loss: {loss.item():.4f}")

if __name__ == '__main__':
    test_pipeline()