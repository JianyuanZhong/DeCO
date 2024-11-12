import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from model.ebm import EBM, ControlVariate, PolicyNetwork
from model.encoder import SolutionEncoder, GNNTransformerEncoder

# Define the LightningModule for training
class EnergyBasedModel(pl.LightningModule):
    def __init__(self, hidden_dim, num_actions, seq_len, lr=5e-4, temperature=0.5):
        super(EnergyBasedModel, self).__init__()
        self.num_actions = num_actions
        self.x_encoder = GNNTransformerEncoder()
        self.y_encoder = SolutionEncoder()
        self.ebm = EBM(hidden_dim, self.num_actions, self.x_encoder, self.y_encoder)
        self.policy_net = PolicyNetwork(self.x_encoder, self.y_encoder, hidden_dim, self.num_actions, seq_len)
        self.control_variate = ControlVariate(self.x_encoder, self.y_encoder, hidden_dim, self.num_actions, seq_len)
        self.lr = lr
        self.seq_len = seq_len
        self.temperature = temperature  # Temperature for Gumbel-Softmax
        
    def forward(self, x, y_t, gt_actions=None):
        # Generate action logits from the policy network
        actions, actions_concrete = self.policy_net.sample(x, y_t, gt_actions=gt_actions)  # Shape: (batch_size, seq_len, num_actions)
        return actions, actions_concrete

    def training_step(self, batch, batch_idx):
        # Unpack the batch according to the new dataset structure
        problem_tensors, pyg_data, y_t, optimized_tours, input_action_sequences, target_action_sequences = batch
        batch_size = problem_tensors.size(0)

        # Create key padding mask for input_actions
        padding_mask = (input_action_sequences == 0)  # Assuming 0 is the padding index

        a_logits = self.policy_net(pyg_data, y_t, actions=input_action_sequences, padding_mask=padding_mask)
        
        # Compute cross-entropy loss for the policy network
        target_mask = (target_action_sequences == 0)
        ce_loss = F.cross_entropy(a_logits.transpose(1, 2), target_action_sequences, reduction='none')
        ce_loss = (ce_loss * (1 - target_mask.float())).mean()

        # gumbel softmax
        a_sampled_onehot = F.gumbel_softmax(a_logits, tau=self.temperature, hard=True)
        a_sampled = torch.argmax(a_sampled_onehot, dim=2)
        
        # Compute energies
        target_mask = (target_action_sequences == 0)
        energy_sampled = self.ebm(pyg_data, y_t, a_sampled_onehot, target_mask)
        energy_target = self.ebm(pyg_data, y_t, F.one_hot(target_action_sequences, num_classes=self.num_actions).float(), target_mask)
        
        # InfoNCE Loss
        temperature = 0.1  # Temperature for InfoNCE loss
        energy_diff = (energy_target - energy_sampled) / temperature
        info_nce_loss = -torch.mean(F.logsigmoid(energy_diff))
        
        # Total Loss
        loss = info_nce_loss + ce_loss
        
        # Log losses
        self.log('train_loss', loss, prog_bar=True)
        self.log('info_nce_loss', info_nce_loss, prog_bar=True)
        self.log('ce_loss', ce_loss, prog_bar=True)
        
        # Print string versions for console viewing
        if batch_idx % 100 == 0:  # Only print every 100 batches to avoid spam
            print(f"Train Sampled Actions: {a_sampled[:5].cpu().numpy().tolist()}")
            print(f"Train Target Actions: {target_action_sequences[:5].cpu().numpy().tolist()}")
        
        return loss

    def validation_step(self, batch, batch_idx):
       # Unpack the batch according to the new dataset structure
        problem_tensors, pyg_data, y_t, optimized_tours, input_action_sequences, target_action_sequences = batch
        batch_size = problem_tensors.size(0)

        # Create key padding mask for input_actions
        padding_mask = (input_action_sequences == 0)  # Assuming 0 is the padding index

        a_logits = self.policy_net(pyg_data, y_t, actions=input_action_sequences, padding_mask=padding_mask)
        
        # Compute cross-entropy loss for the policy network
        target_mask = (target_action_sequences == 0)
        ce_loss = F.cross_entropy(a_logits.transpose(1, 2), target_action_sequences, reduction='none')
        ce_loss = (ce_loss * (1 - target_mask.float())).mean()

        # gumbel softmax
        a_sampled_onehot = F.gumbel_softmax(a_logits, tau=self.temperature, hard=True)
        a_sampled = torch.argmax(a_sampled_onehot, dim=2)
        
        # Compute energies
        target_mask = (target_action_sequences == 0)
        energy_sampled = self.ebm(pyg_data, y_t, a_sampled_onehot, target_mask)
        energy_target = self.ebm(pyg_data, y_t, F.one_hot(target_action_sequences, num_classes=self.num_actions).float(), target_mask)
        
        # InfoNCE Loss
        temperature = 0.1  # Temperature for InfoNCE loss
        energy_diff = (energy_target - energy_sampled) / temperature
        info_nce_loss = -torch.mean(F.logsigmoid(energy_diff))
        
        # Total Loss
        loss = info_nce_loss + ce_loss
        
        # Log losses
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_energy_loss', info_nce_loss, prog_bar=True)
        self.log('val_ce_loss', ce_loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }