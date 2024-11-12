import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from deco import EnergyBasedModel  # Assuming the model is in deco.py
from dataset import TSPDataset, collate_fn  # Assuming the dataset is in dataset.py
import wandb
from lightning.pytorch.loggers import WandbLogger
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Energy Based Model')
    parser.add_argument('--disable_logging', action='store_true', help='Disable logging to Weights & Biases')
    parser.add_argument('--data_path', type=str, default="data/results-10000.pkl", help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--num_actions', type=int, default=104, help='Number of actions')
    parser.add_argument('--seq_len', type=int, default=8, help='Sequence length')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--project', type=str, default='deco', help='Project name')
    parser.add_argument('--wandb_name', type=str, default='1000-debug', help='Weights & Biases run name')
    args = parser.parse_args()

    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Initialize Weights & Biases logger if logging is not disabled
    wandb_logger = WandbLogger(project=args.project, log_model=True, name=args.wandb_name) if not args.disable_logging else None

    # Load dataset
    dataset = TSPDataset(args.data_path)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    # Initialize model
    model = EnergyBasedModel(args.hidden_dim, args.num_actions, args.seq_len)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        logger=wandb_logger
    )

    # Train the model
    trainer.fit(model, train_loader, train_loader)

if __name__ == "__main__":
    main()