import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import minimum_spanning_tree
from torch_geometric.utils import to_networkx, from_networkx, degree, unbatch, unbatch_edge_index
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.data import Data

from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

class GlobalConnectivityLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rho = 1.0  # Initial penalty coefficient
        self.alpha = 1.5  # Update multiplier for rho
        self.rho_max = 10.0  # Maximum value of rho
        self.tau = 0.1  # Tolerance for updating rho
        self.v_k_minus_1 = 0  # Store v_k for comparison in the next iteration

    def compute_minimal_1_tree(self, graph, penalties):
        """
        Compute the minimal 1-tree using PyTorch Geometric and SciPy's MST implementation.
        """
        # Extract edge_index and edge_attr from the PyG graph
        edge_index = graph.edge_index.cpu()
        edge_attr = graph.edge_attr.squeeze().detach().cpu()
        penalties = penalties.squeeze().detach().cpu()

        # Modify the edge weights using node penalties
        u, v = edge_index
        # print(penalties.shape)
        edge_weights = edge_attr + penalties[u] + penalties[v]

        # Convert to a sparse matrix format for faster MST calculation
        adj = to_scipy_sparse_matrix(edge_index, edge_weights, graph.num_nodes)
        mst = minimum_spanning_tree(adj)  # Compute MST and convert to COO format for easier manipulation

        # Create an adjacency matrix of the MST for further processing
        mst_adj = mst.toarray()
        adj = adj.toarray()

        # Identify the node with the maximum degree in the MST
        degree_array = mst_adj.sum(axis=0) + mst_adj.sum(axis=1)  # Sum of rows and columns for an undirected graph
        max_degree_node = torch.argmax(torch.tensor(degree_array)).item()

        # Find the second least-weighted edge connected to the max degree node (from the original adjacency matrix)
        candidate_edges = [(max_degree_node, v) for v in range(graph.num_nodes) if adj[max_degree_node, v] > 0]
        candidate_edges = sorted(candidate_edges, key=lambda x: adj[x[0], x[1]])

        # Add the second least-weighted edge to the MST to form the 1-tree
        if len(candidate_edges) > 1:
            u, v = candidate_edges[1]  # Second least-weighted edge
            mst_adj[u, v] = adj[u, v]  # Add the edge to the MST
            mst_adj[v, u] = adj[v, u]  # Ensure symmetry since the graph is undirected

        # Convert the MST 1-tree adjacency matrix to edge_index and edge_attr
        mst_1_tree_coo = coo_matrix(mst_adj)
        edge_index = torch.tensor([mst_1_tree_coo.row, mst_1_tree_coo.col], dtype=torch.long)
        edge_attr = torch.tensor(mst_1_tree_coo.data, dtype=torch.float)

        # Create a PyG Data object with edge_index and edge attributes
        mst_1_tree_graph = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=graph.num_nodes)

        # # Recalculate the degrees for the 1-tree
        # degree_source = degree(mst_1_tree_graph.edge_index[0], num_nodes=graph.num_nodes, dtype=torch.float)
        # degree_destination = degree(mst_1_tree_graph.edge_index[1], num_nodes=graph.num_nodes, dtype=torch.float)
        # total_degree = degree_source + degree_destination

        return mst_1_tree_graph.to(graph.x.device)


    def compute_degree_loss(self, edge_indices, mst_probs, num_nodes):
        # Create a tensor to accumulate degree counts for all nodes
        degree = torch.zeros((num_nodes,), device=mst_probs.device)

        # Unpack the edge indices
        u = edge_indices[0]
        v = edge_indices[1]

        # Use scatter_add to accumulate probabilities for the edges
        degree.scatter_add_(0, u, mst_probs)  # Update degree for source nodes
        degree.scatter_add_(0, v, mst_probs)  # Update degree for target nodes

        # Define the target degree (e.g., 2 for 2-regular graph)
        target_degree = torch.full_like(degree, fill_value=2, device=degree.device)

        # Compute the degree loss using mean squared error
        degree_loss = torch.abs(degree - target_degree).mean()

        return degree_loss


    def compute_cycle_loss(self, edge_index, tour_probs, num_nodes):
        """
        Compute the cycle loss for a single graph.
        
        :param edge_index: Tensor of shape [2, num_edges], where each column is an edge.
        :param tour_probs: Tensor of shape [num_edges], the probabilities associated with the edges.
        :param num_nodes: The number of nodes in the graph.
        :return: The computed cycle loss for the graph.
        """
        N = num_nodes  # Number of nodes in the current graph

        # Convert the edge probabilities to a dense adjacency matrix
        adj_matrix = torch.zeros((N, N), device=tour_probs.device)
        adj_matrix[edge_index[0], edge_index[1]] = tour_probs
        adj_matrix[edge_index[1], edge_index[0]] = tour_probs  # Assuming undirected graph

        # Compute the degree matrix
        degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))

        # Compute the Laplacian matrix
        laplacian_matrix = degree_matrix - adj_matrix

        # Compute the eigenvalues
        eigenvalues = torch.linalg.eigvalsh(laplacian_matrix)

        # Apply a differentiable threshold to make very small values close to 0
        a = torch.where(eigenvalues.abs() > 1e-5)
        eigenvalues = eigenvalues[a]

        # Use a smooth approximation to count non-zero eigenvalues
        sharpness = 10  # Control the sharpness of the transition
        approx_rank = torch.sum(torch.sigmoid(sharpness * torch.abs(eigenvalues)))

        # The cycle loss is the approximate rank minus 1 (we want one connected component)
        cycle_loss = (N - 1 - approx_rank).abs()
        if cycle_loss > 1:
            cycle_loss = cycle_loss * 1000

        return cycle_loss


    def compute_tree_loss(self, mst_1_tree_graph, mst_probs, edge_index):
        """
        Computes the tree loss for a given subgraph.

        :param subgraph: The current subgraph for which the minimal 1-tree is computed.
        :param mst_probs: The predicted probabilities for the edges of being in the MST.
        :param penalties: The penalties associated with the nodes of the subgraph.
        :param edge_index: The edge index tensor for the subgraph.

        :return: The computed tree loss (BCE loss).
        """

        # Create a binary label from the mst_1_tree_graph
        binary_labels = torch.zeros(edge_index.size(1), device=edge_index.device)

        # Find edges that are part of the minimal 1-tree and set them to 1
        for u, v in mst_1_tree_graph.edge_index.T:
            # Find the index in the original subgraph edge_index
            matching_edges = ((edge_index[0] == u) & (edge_index[1] == v)) | ((edge_index[0] == v) & (edge_index[1] == u))
            binary_labels[matching_edges] = 1

        # Compute BCE loss between mst_probs and binary_labels
        bce_loss = F.binary_cross_entropy(mst_probs, binary_labels)

        return bce_loss
    

    def sample_edges(self, mst_probs, n, temp=2.5):
        # Generate Gumbel noise
        gumbel_noise = (
            -torch.empty_like(mst_probs, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )
        
        # Perturb the probabilities
        perturbed_scores = mst_probs + gumbel_noise
        
        # Apply Softmax to get probabilities
        y_soft = F.softmax(perturbed_scores / temp, dim=0)

        # Sample edges
        # Straight through for n-hot
        top_n_indices = torch.topk(y_soft, n, dim=0).indices
        y_n_hot = torch.zeros_like(mst_probs, memory_format=torch.legacy_contiguous_format)

        # Scatter 1s at the top n indices
        y_n_hot.scatter_(0, top_n_indices, 1.0)

        # Straight-through method
        ret = y_n_hot - y_soft.detach() + y_soft
        return ret


    def forward_primal(self, batch, mst_logits, penalties):
        cost = 0
        component_penalty = 0
        degree_penalty = 0

        ptr = batch.ptr
        xs, edge_indices = unbatch(batch.x, batch.batch), unbatch_edge_index(batch.edge_index, batch.batch)
        edge_label = batch.edge_label.float()

        num_graphs = len(xs)

        for i, (x, edge_index) in enumerate(zip(xs, edge_indices)):
            start, end = ptr[i], ptr[i + 1]
            mask = (batch.edge_index[0] >= start) & (batch.edge_index[0] < end)
            
            penalties_ = penalties[start:end]
            edge_attr = batch.edge_attr[mask]
            labels = edge_label[mask]
            
            # # Sample n edges based on mst_probs
            n_edges = end - start
            approx_solution = self.sample_edges(mst_logits[mask], n_edges)
            # approx_solution = mst_logits[mask].sigmoid()
            # approx_solution = torch.sigmoid(10 * approx_solution)

            # edge_weights = edge_attr + penalties_[u] + penalties_[v]
            edge_weights = edge_attr.detach().squeeze()
            cost += F.binary_cross_entropy(approx_solution, labels, weight=edge_weights)

            # Compute the degree penalty on the sampled edges
            degree_penalty += (self.compute_degree_loss(edge_index, approx_solution, end - start) * penalties_ * self.rho).mean()

            # Compute other penalties as before
            # component_penalty += self.compute_cycle_loss(edge_index, approx_solution, end - start)
            # component_penalty += F.l1_loss(n_edges, torch.sum(approx_solution))

        return cost / num_graphs, component_penalty / num_graphs, degree_penalty / num_graphs

    def reset_epoch(self):
        """
        Reset the maximum violation at the beginning of the epoch.
        """
        self.v_k_minus_1 = 0  # Reuse v_k_minus_1 to track max violation during the epoch

    def forward_dual(self, batch, mst_logits, penalties, penalties_old):
        cost = 0
        dual_loss = 0
        ptr = batch.ptr
        xs, edge_indices = unbatch(batch.x, batch.batch), unbatch_edge_index(batch.edge_index, batch.batch)
        mst_probs = mst_logits.sigmoid()
        edge_label = batch.edge_label.float()

        num_graphs = len(xs)

        for i, (x, edge_index) in enumerate(zip(xs, edge_indices)):
            start, end = ptr[i], ptr[i + 1]
            mask = (batch.edge_index[0] >= start) & (batch.edge_index[0] < end)
            penalties_ = penalties[start:end]
            penalties_old_ = penalties_old[start:end]
            
            # Sample n edges based on mst_probs
            n_edges = end - start  # specify the number of edges you want to sample
            approx_solution = self.sample_edges(mst_logits[mask], n_edges)
            
            degree = torch.zeros((end - start,), device=mst_probs.device)
            u = edge_index[0]
            v = edge_index[1]

            # Use scatter_add to accumulate probabilities for the edges
            degree.scatter_add_(0, u, approx_solution)
            degree.scatter_add_(0, v, approx_solution)
            
            # Compute the dual loss
            constraint_violation_term = self.rho * torch.abs(degree - 2)
            dual_loss += torch.abs(penalties_ - (penalties_old_ + constraint_violation_term)).mean()

            # Calculate the maximum violation for this graph (graph-level)
            v_k = self.compute_graph_max_violation(penalties_, degree)
            self.v_k_minus_1 = max(self.v_k_minus_1, v_k)  # Update v_k_minus_1 to store the maximum violation

            edge_attr = batch.edge_attr[mask]
            edge_weights = edge_attr.detach().squeeze()
            labels = edge_label[mask]
            cost += F.binary_cross_entropy(approx_solution, labels, weight=edge_weights)

        return dual_loss / num_graphs, cost / num_graphs

    def compute_graph_max_violation(self, penalties, degree):
        """
        Compute the maximum violation for the entire graph
        based on the inequality and equality constraints.
        """
        # Compute violations at each node
        g_x_j = penalties  # Penalties for nodes based on inequality constraints
        h_x_j = torch.abs(degree - 2)  # Degree difference for equality constraint (target degree = 2)
        
        # Compute maximum violation for the entire graph (graph-level aggregate)
        sigma_x_j = torch.max(g_x_j, -penalties / self.rho)  # Inequality constraint violation
        
        # Get the maximum violation across the graph (isomorphism-invariant)
        v_k = torch.max(torch.norm(h_x_j, float('inf')), torch.norm(sigma_x_j, float('inf')))
        
        return v_k

    def update_rho(self):
        """
        Update the penalty coefficient rho based on the maximum violation observed during the entire epoch.
        """
        if self.v_k_minus_1 > self.tau * self.v_k_minus_1:
            self.rho = min(self.alpha * self.rho, self.rho_max)

    def end_epoch_update(self):
        """
        Call this at the end of the epoch to update rho.
        """
        self.update_rho()  # Update rho based on the maximum violation for the epoch
        self.reset_epoch()  # Reset the max violation tracker for the next epoch

    def forward(self, x):
        return x