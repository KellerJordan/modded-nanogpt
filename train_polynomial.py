import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

class GeneralizedNewtonSchulz(nn.Module):
    def __init__(self, degree: int = 3, num_iterations: int = 5, init_coefficients: List[float] = None):
        super().__init__()
        self.degree = degree
        self.num_iterations = num_iterations
        if init_coefficients is None:
            self.coefficients = nn.Parameter(torch.randn(self.num_polynomial_terms))
        else:
            self.coefficients = nn.Parameter(torch.tensor(init_coefficients))
        self.initial_scale = 1.1
    
    @property
    def num_polynomial_terms(self) -> int:
        return (self.degree + 1) // 2

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X = X / torch.norm(X, p='fro', dim=(-2, -1), keepdim=True) * torch.abs(self.initial_scale)
        X = X / torch.norm(X, p='fro', dim=(-2, -1), keepdim=True) * 1.1

        for _ in range(self.num_iterations):
            XT = X.transpose(-2, -1)
            XTX = torch.bmm(XT, X)
            terms = [X]
            
            for _ in range(self.num_polynomial_terms - 1):
                terms.append(torch.bmm(terms[-1], XTX))
            
            X = sum(coeff * term for coeff, term in zip(self.coefficients, terms))
            
        return X

    def print_polynomial(self) -> str:
        terms = [f"{coeff:.3f}X(X^TX)^{i}" if i else f"{coeff:.3f}X" 
                for i, coeff in enumerate(self.coefficients.detach().cpu().tolist())
                if abs(coeff) > 1e-6]
        return " + ".join(terms)

    def evaluate_scalar(self, x: torch.Tensor) -> torch.Tensor:
        powers = torch.tensor([2 * i + 1 for i in range(len(self.coefficients))])
        return sum(coeff * torch.pow(x, power) for coeff, power in zip(self.coefficients, powers))

@dataclass
class MatrixSample:
    name: str
    shape: Tuple[int, ...]
    file_path: str

class MatrixDataset(Dataset):
    def __init__(self, checkpoint_dirs: List[str], pattern: str = "checkpoint_step*.pkl"):
        # Ensure checkpoint_dirs is a list
        if isinstance(checkpoint_dirs, str):
            checkpoint_dirs = [checkpoint_dirs]
            
        # Collect files from all directories
        files = []
        for checkpoint_dir in checkpoint_dirs:
            dir_files = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))[20:]  # Skip first checkpoints
            files.extend(dir_files)
            
        if len(files) < 2:
            raise ValueError(f"Need at least 2 checkpoints, found {len(files)} across {len(checkpoint_dirs)} directories")
        
        # Load first checkpoint from first directory to get matrix shapes
        with open(files[0], 'rb') as f:
            first_checkpoint = pickle.load(f)
        
        self.samples = [MatrixSample(name, shape, file) 
                       for file in files 
                       for name, _, shape in first_checkpoint]

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.samples[idx]
        with open(sample.file_path, 'rb') as f:
            params = pickle.load(f)
            for name, matrix, _ in params:
                if name == sample.name:
                    return torch.from_numpy(matrix)
        raise ValueError(f"Matrix {sample.name} not found in {sample.file_path}")

    def __len__(self) -> int:
        return len(self.samples)

def create_dataloader(checkpoint_dirs: List[str], batch_size: int = 8) -> DataLoader:
    dataset = MatrixDataset(checkpoint_dirs)
    shape_indices = {}
    
    for idx, sample in enumerate(dataset.samples):
        shape_indices.setdefault(sample.shape, []).append(idx)
    
    # Create a BatchSampler class instead of a generator
    class ShapeBatchSampler(torch.utils.data.Sampler):
        def __iter__(self):
            # Create all batches first
            all_batches = []
            for indices in shape_indices.values():
                indices_copy = indices.copy()
                np.random.shuffle(indices_copy)
                # Create batches for this shape
                shape_batches = [indices_copy[i:i + batch_size] 
                            for i in range(0, len(indices_copy), batch_size)]
                all_batches.extend(shape_batches)
            
            # Shuffle the order of batches
            np.random.shuffle(all_batches)
            yield from all_batches
        
        def __len__(self):
            return sum(len(indices) // batch_size + (1 if len(indices) % batch_size else 0)
                      for indices in shape_indices.values())
    
    return DataLoader(dataset, 
                     batch_sampler=ShapeBatchSampler(), 
                     collate_fn=lambda x: torch.stack(x),
                     num_workers=4)

def iterate(fn, num_iterations: int = 10):
    def wrapper(x):
        result = x
        for _ in range(num_iterations):
            result = fn(result)
        return result
    return wrapper

def naive_loss(X):
    I = torch.eye(X.size(-1)).to(X.device)
    return torch.norm(X.transpose(-2, -1) @ X - I, p='fro')


def train_newton_schulz(config: Dict[str, Any]):
    run = wandb.init(project="newton-schulz-polynomial", config=config)
    device = torch.device(config['device'])
    
    model = GeneralizedNewtonSchulz(
        degree=config['degree'],
        num_iterations=config['num_iterations'],
        init_coefficients=config['init_coefficients']
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        betas=config['adam_betas']
    )
    
    dataloader = create_dataloader(config['checkpoint_dirs'], config['batch_size'])
    
    def plot_fn(fn, title):
        fig, ax = plt.subplots()
        x = torch.linspace(-0.5, 2.0, 10000).to(device)
        y = fn(x).cpu().detach()
        ax.plot(x.cpu().numpy(), y.numpy())
        ax.set_title(title)
        ax.set_ylim(-1, 3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        img = wandb.Image(plt)
        plt.close()
        return img
    
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        num_batches = 0
        current_loss = 0

        for i, matrices in enumerate(dataloader):
            matrices = matrices.to(device)
            output = model(matrices)
            loss_fn = naive_loss
            loss = loss_fn(output) - config['alpha'] * model.coefficients[0].pow(2)
            loss.backward()
            current_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1
            
            if (i + 1) % config['accumulation_steps'] == 0:
                # torch.nn.utils.clip_grad_norm_(
                #     model.parameters(), config['max_grad_norm']
                # )
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({"loss": current_loss})
                current_loss = 0

        wandb.log({
            "epoch": epoch,
            "avg_epoch_loss": epoch_loss / num_batches,
            "polynomial": plot_fn(model.evaluate_scalar, model.print_polynomial()),
            "iterated_polynomial": plot_fn(
                iterate(model.evaluate_scalar, model.num_iterations),
                f"Iterated {model.num_iterations} times"
            )
        })
    
    run.finish()

if __name__ == "__main__":
    default_config = {
        "device": "cuda",
        "checkpoint_dirs": ["logs/reference_run", "logs/spiky_run"],
        "degree": 7,
        "num_iterations": 3,
        "learning_rate": 1e-2,
        "adam_betas": (0.9, 0.9),
        "batch_size": 32,
        "accumulation_steps": 5,
        "max_grad_norm": 1.0,
        "num_epochs": 200,
        "alpha": 1,
        "init_coefficients": [4.439, -7.364, 3.5, -0.279]
    }
    
    # Run single training
    train_newton_schulz(default_config)
    
    # Uncomment for hyperparameter sweep
    """
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'degree': {'values': [3, 5, 7]},
            'num_iterations': {'values': [2, 3, 4, 5, 10]},
            'learning_rate': {'values': [0.001, 0.01, 0.1]},
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="newton-schulz-polynomial")
    wandb.agent(sweep_id, function=train_newton_schulz)
    """