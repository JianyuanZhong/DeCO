import torch
from torch import nn
from torch_geometric.nn import GINEConv


class GNNTransformerEncoder(nn.Module):
    def __init__(self, dropout=0.2, n_q=3, hidden_dim=256):
        super(GNNTransformerEncoder, self).__init__()
        self.conv1 = GINEConv(
            nn=torch.nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 128),
            ),
            train_eps=True,
            edge_dim=1,
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(128, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            train_eps=True,
            edge_dim=1,
        )

        self.transformer = self.build_transformer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim*2,
            num_layers=6,
            dropout=0.1,
            activation=nn.SiLU()
        )
        self.hidden_dim = hidden_dim
        self.n_q = n_q
        self.pos_embedding = nn.Parameter(torch.randn(512, hidden_dim))  # Support up to 512 nodes

    def build_transformer(
        self, d_model, nhead, dim_feedforward, num_layers, dropout, activation
    ):
        self.nhead = nhead
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first=True,
            norm_first=True,
        )
        encoder_norm = torch.nn.LayerNorm(d_model, eps=1e-5)
        transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return transformer
    
    def forward(self, data):
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = x.relu()
        x = self.conv2(x, data.edge_index, data.edge_attr)
        batched_node_embed, padding_mask, src_mask = self.bachify_gnn_outputs(x, data)
        batched_node_embed = batched_node_embed + self.pos_embedding[:batched_node_embed.size(1)]
        if self.training:
            batched_out = self.transformer(batched_node_embed, mask=src_mask, src_key_padding_mask=padding_mask)
        else:
            batched_out = self.transformer(batched_node_embed, src_key_padding_mask=padding_mask)
        # node_emb = self.unbatch_transformer_outputs(batched_out, data)
        
        return batched_out

    def encode(self, data):
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = x.relu()
        x = self.conv2(x, data.edge_index, data.edge_attr)
        batched_node_embed, padding_mask, src_mask = self.bachify_gnn_outputs(x, data)
        batched_node_embed = batched_node_embed + self.pos_embedding[:batched_node_embed.size(1)]
        batched_out = self.transformer(batched_node_embed, src_key_padding_mask=padding_mask)
        node_emb = self.unbatch_transformer_outputs(batched_out, data)
        return node_emb
    
    def bachify_gnn_outputs(self, x, data):
        # Assign each node to the corresponding example
        node_embed_list = [x[data.ptr[i-1]:data.ptr[i]] for i in range(1, len(data.ptr))]

        # Pad sequence
        batched_node_embed = nn.utils.rnn.pad_sequence(node_embed_list, batch_first=True)
        bs, seq_len, fea = batched_node_embed.shape

        # Compute adjacency matrix and src_mask in a vectorized manner
        edge_index = data.edge_index
        dropped_edges = torch.nonzero(data.drop_edge_label).view(-1)
        dropped_edges_index = edge_index[:, dropped_edges]

        adj_matrix = torch.ones((data.x.size(0), data.x.size(0)), device=x.device)
        adj_matrix[dropped_edges_index[0], dropped_edges_index[1]] = 0
        adj_matrix[dropped_edges_index[1], dropped_edges_index[0]] = 0
        adj_matrix = adj_matrix.unsqueeze(0).repeat(self.nhead, 1, 1)

        src_mask = torch.ones((bs * self.nhead, seq_len, seq_len), device=x.device)
        for i in range(bs):
            offset = i * self.nhead
            left, right = data.ptr[i], data.ptr[i+1]
            src_mask[offset:offset+self.nhead, :right-left, :right-left] = adj_matrix[:, left:right, left:right]

        # Create padding mask
        padding_mask = torch.zeros(bs, seq_len, device=x.device)
        for i in range(bs):
            left, right = data.ptr[i], data.ptr[i+1]
            padding_mask[i, :right-left] = 1

        return batched_node_embed, padding_mask, src_mask
    
    def unbatch_transformer_outputs(self, batched_out, data):
        x = torch.zeros((data.x.size(0), batched_out.shape[-1]), device=data.x.device)
        for i in range(batched_out.shape[0]):
            left, right = data.ptr[i], data.ptr[i+1]
            x[left:right] = batched_out[i, :right-left]
        return x


class SolutionEncoder(nn.Module):
    def __init__(self, dropout=0.2, hidden_dim=256, num_nodes=100):
        super(SolutionEncoder, self).__init__()
        self.encoder = self.build_transformer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim*2,
            num_layers=6,
            dropout=0.1,
            activation=nn.SiLU()
        )
        
        # Output projection MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.abs_pos_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.hidden_dim = hidden_dim
    def build_transformer(self, d_model, nhead, dim_feedforward, num_layers, dropout, activation):
        self.nhead = nhead
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first=True,
            norm_first=True,
        )
        encoder_norm = torch.nn.LayerNorm(d_model, eps=1e-5)
        transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return transformer

    def add_positional_embeddings(self, batched_node_embed):
        bs, seq_len, fea = batched_node_embed.shape
        position = torch.arange(seq_len, device=batched_node_embed.device)
        
        # Generate rotation matrix elements
        dim = fea // 2
        theta = position.unsqueeze(-1) / torch.pow(10000, 2 * torch.arange(0, dim, device=batched_node_embed.device) / dim)
        
        # Create rotation matrices using sin and cos
        cos = torch.cos(theta).view(seq_len, -1)  # Shape: [seq_len, dim]
        sin = torch.sin(theta).view(seq_len, -1)  # Shape: [seq_len, dim]
        
        # Apply rotary embeddings
        x1 = batched_node_embed[..., :dim]
        x2 = batched_node_embed[..., dim:2*dim]
        
        # Expand cos and sin for broadcasting
        cos = cos.unsqueeze(0)  # [1, seq_len, dim]
        sin = sin.unsqueeze(0)  # [1, seq_len, dim]
        
        # Rotate the embeddings
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)
        
        return rotated
        

    def forward(self, node_embed, y_t):
        # Gather node embeddings in node index order
        # node_embed: [batch_size, num_nodes, hidden_dim]
        # y_t: [batch_size, num_nodes] contains indices
        y_t = y_t - 1
        reordered_embed = torch.gather(node_embed, 1, y_t.unsqueeze(-1).expand(-1, -1, node_embed.size(-1)))
        reordered_embed = reordered_embed + self.abs_pos_embedding(y_t)
        
        # Add positional embeddings
        reordered_embed = self.add_positional_embeddings(reordered_embed)
        
        # Pass through transformer
        transformer_out = self.encoder(reordered_embed)
        
        # Project through MLP
        return self.output_mlp(transformer_out)
