import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from model.encoder import GNNTransformerEncoder, SolutionEncoder

# Define the Control Variate Network for RELAX
class ControlVariate(nn.Module):
    def __init__(self, x_encoder, y_encoder, hidden_dim, num_actions, seq_len):
        super(ControlVariate, self).__init__()
        self.seq_len = seq_len
        self.num_actions = num_actions
        
        self.net = nn.Sequential(
            nn.Linear(x_encoder.hidden_dim + y_encoder.hidden_dim + seq_len * num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, a_concrete, x, y_t):
        # Flatten a_concrete and concatenate with x and y_t
        a_flat = a_concrete.view(a_concrete.size(0), -1)  # Shape: (batch_size, seq_len * num_actions)
        input = torch.cat([x, y_t, a_flat], dim=1)  # Shape: (batch_size, x_dim + y_dim + seq_len * num_actions)
        c = self.net(input).squeeze(-1)  # Shape: (batch_size,)
        return c

# Define the EBM model
class EBM(nn.Module):
    def __init__(self, hidden_dim, num_actions, x_encoder, y_encoder):
        super(EBM, self).__init__()
        
        self.num_actions = num_actions  # Number of possible discrete actions
        
        # Encoder for problem instance x
        self.x_encoder = x_encoder
        
        # Encoder for current solution y_t
        self.y_encoder = y_encoder
        
        # Action embedding
        self.action_embedding = nn.Linear(num_actions, hidden_dim)
        
        # Decoder for action sequence a_{1:t}
        self.action_decoder = self.build_transformer_decoder(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim*2,
            num_layers=3,
            dropout=0.1,
            activation=nn.SiLU()
        )
        
        # Fully connected layer to compute energy
        self.energy_fc = nn.Linear(hidden_dim, 1)

    def build_transformer_decoder(self, d_model, nhead, dim_feedforward, num_layers, dropout, activation):
        self.nhead = nhead
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first=True,
            norm_first=True,
        )
        decoder_norm = torch.nn.LayerNorm(d_model, eps=1e-5)
        transformer = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)

        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return transformer

    def add_positional_embeddings(self, batched_embed):
        bs, seq_len, fea = batched_embed.shape
        position = torch.arange(seq_len, device=batched_embed.device)
        
        # Generate rotation matrix elements
        dim = fea // 2
        theta = position.unsqueeze(-1) / torch.pow(10000, 2 * torch.arange(0, dim, device=batched_embed.device) / dim)
        
        # Create rotation matrices using sin and cos
        cos = torch.cos(theta).view(seq_len, -1)  # Shape: [seq_len, dim]
        sin = torch.sin(theta).view(seq_len, -1)  # Shape: [seq_len, dim]
        
        # Apply rotary embeddings
        x1 = batched_embed[..., :dim]
        x2 = batched_embed[..., dim:2*dim]
        
        # Expand cos and sin for broadcasting
        cos = cos.unsqueeze(0)  # [1, seq_len, dim]
        sin = sin.unsqueeze(0)  # [1, seq_len, dim]
        
        # Rotate the embeddings
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)
        
        return rotated

    def forward(self, x, y_t, a_seq, key_padding_mask=None):
        """
        x: Tensor of shape (batch_size, x_dim)
        y_t: Tensor of shape (batch_size, y_dim)
        a_seq: Tensor of shape (batch_size, seq_len) with discrete action indices
        """
        # Encode problem instance x
        x_enc = self.x_encoder(x)  # Shape: (batch_size, hidden_dim)
        
        # Encode current solution y_t
        y_enc = self.y_encoder(x_enc, y_t)  # Shape: (batch_size, hidden_dim)
        
        # Embed action sequence
        a_seq_emb = self.action_embedding(a_seq)  # Shape: (batch_size, seq_len, hidden_dim)
        
        # Add positional embeddings
        a_seq_emb = self.add_positional_embeddings(a_seq_emb)
        
        # Decode action sequence a_{1:t} with cross attention to y_enc
        a_dec = self.action_decoder(a_seq_emb, y_enc, tgt_key_padding_mask=key_padding_mask)  # Shape: (batch_size, seq_len, hidden_dim)
        a_dec_mean = a_dec.mean(dim=1)  # Shape: (batch_size, hidden_dim)
        
        # Compute energy
        energy = self.energy_fc(a_dec_mean)  # Shape: (batch_size, 1)
        return energy.squeeze(-1)  # Return shape: (batch_size,)

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, x_encoder, y_encoder, hidden_dim, num_actions, seq_len=15, num_nodes=100):
        super(PolicyNetwork, self).__init__()
        self.seq_len = seq_len
        self.num_actions = num_actions
        
        # Encoder for problem instance x and current solution y_t
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder
        
        # Action embedding layer
        self.action_embedding = nn.Embedding(num_actions, hidden_dim)
        
        # Decoder to produce action logits at each time step
        self.decoder = self.build_transformer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim*4,
            num_layers=3,
            dropout=0.1,
            activation=nn.SiLU()
        )
        
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.num_nodes = num_nodes
        
        # Create causal mask once during initialization
        self.register_buffer("causal_mask", torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())
    
    def build_transformer(self, d_model, nhead, dim_feedforward, num_layers, dropout, activation):
        self.nhead = nhead
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first=True,
            norm_first=True,
        )
        decoder_norm = torch.nn.LayerNorm(d_model, eps=1e-5)
        transformer = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)

        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return transformer

    def add_positional_embeddings(self, batched_embed):
        bs, seq_len, fea = batched_embed.shape
        position = torch.arange(seq_len, device=batched_embed.device)
        
        # Generate rotation matrix elements
        dim = fea // 2
        theta = position.unsqueeze(-1) / torch.pow(10000, 2 * torch.arange(0, dim, device=batched_embed.device) / dim)
        
        # Create rotation matrices using sin and cos
        cos = torch.cos(theta).view(seq_len, -1)  # Shape: [seq_len, dim]
        sin = torch.sin(theta).view(seq_len, -1)  # Shape: [seq_len, dim]
        
        # Apply rotary embeddings
        x1 = batched_embed[..., :dim]
        x2 = batched_embed[..., dim:2*dim]
        
        # Expand cos and sin for broadcasting
        cos = cos.unsqueeze(0)  # [1, seq_len, dim]
        sin = sin.unsqueeze(0)  # [1, seq_len, dim]
        
        # Rotate the embeddings
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)
        
        return rotated
    
    def forward(self, x, y_t, actions=None, padding_mask=None):
        """
        Forward pass through the transformer model to get action logits.
        
        Args:
            x: Tensor of shape (batch_size, x_dim)
            y_t: Tensor of shape (batch_size, y_dim) 
            actions: Optional tensor of shape (batch_size, seq_len) containing action indices
            
        Returns:
            action_logits: Tensor of shape (batch_size, seq_len, num_actions)
        """
        batch_size = x.size(0)
        device = y_t.device
        
        # Encode problem instance x
        x_enc = self.x_encoder(x)  # Shape: (batch_size, hidden_dim)
        
        # Encode current solution y_t
        y_enc = self.y_encoder(x_enc, y_t)  # Shape: (batch_size, hidden_dim)
        
        # Add positional embeddings
        memory = self.add_positional_embeddings(y_enc)
        
        # Get action embeddings if provided, else zero tensor
        if actions is not None:
            action_embed = self.action_embedding(actions)  # (batch_size, seq_len, d_model)
        else:
            action_zero = torch.zeros(batch_size, self.seq_len, device=device)
            action_embed = self.action_embedding(action_zero)
        action_embed = self.add_positional_embeddings(action_embed)
        
        # Apply transformer decoder
        # Use causal mask to prevent attending to future positions
        seq_len = action_embed.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=action_embed.device), diagonal=1).bool()
        
        output = self.decoder(
            tgt=action_embed,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )
        
        # Project to action logits
        action_logits = self.action_head(output)  # (batch_size, seq_len, num_actions)
        
        return action_logits

    def sample(self, x, y_t, temperature=1.0):
        """
        Sample an action sequence from the policy network
        
        Args:
            x: Tensor of shape (batch_size, x_dim)
            y_t: Tensor of shape (batch_size, y_dim)
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            actions: Tensor of shape (batch_size, seq_len) with sampled action indices
        """
        batch_size = x.size(0) // self.num_nodes
        device = y_t.device
        
        # Initialize empty action sequence
        actions = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        actions_concrete = torch.zeros(batch_size, 1, self.num_actions, dtype=torch.float, device=device)
        
        # Sample actions autoregressively
        for t in range(self.seq_len):
            logits = self.forward(x, y_t, actions)
            logit = logits[:, t] / temperature
            
            # Use Gumbel-Softmax for differentiable sampling
            next_action_concrete = F.softmax(logit, dim=-1)
            next_action = next_action_concrete.argmax(dim=-1)  # Convert one-hot to indices
            
            # Update action sequence
            actions = torch.cat([actions, next_action.unsqueeze(1)], dim=1).detach()
            actions_concrete = torch.cat([actions_concrete, next_action_concrete.unsqueeze(1)], dim=1)
            
        return actions, actions_concrete[:, 1:]


