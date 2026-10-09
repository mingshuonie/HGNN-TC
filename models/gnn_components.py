"""Shared architecture components of the FA-GNN and TA-GNN branches.

Both branches build the same message-passing stack (per-type convolutions,
edge attention, layer attention, attention pooling, regression head); the
subclasses differ only in how node/edge input features are prepared. All
submodule attribute names are kept identical across the branches so that
their state-dict keys and checkpoint layouts remain unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import (GCNConv, GATConv, GINEConv, EdgeConv, ResGatedGraphConv, SAGEConv,
                                TransformerConv, global_add_pool, global_mean_pool)


class ThermalGNNBase(nn.Module):
    """Common GNN architecture shared by the feature- and text-attributed branches."""

    def _set_gnn_type(self, gnn_type):
        """Normalize and store the GNN type.

        Previously this argument was accepted but never used, so every
        configuration silently degenerated to GINE; it is now lowercased and
        applied when the convolution stack is built.
        """
        self.gnn_type = gnn_type.lower() if isinstance(gnn_type, str) else gnn_type
        self.heads = 4  # attention heads for GAT / Transformer
        self.concat = True

    def _build_encoders(self, node_input_dim, hidden_dim, dropout, edge_input_dim):
        """Build the node normalization/encoder and the optional edge encoder."""
        # Node feature batch normalization
        self.node_bn = nn.BatchNorm1d(node_input_dim)

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Edge feature encoder (only when edge features are used)
        if edge_input_dim > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )

    def _build_shared_modules(self, hidden_dim, num_layers, dropout, edge_input_dim, gnn_type):
        """Build the shared stack: per-layer MLPs, convolutions, attentions, heads."""
        # Per-layer MLPs so each layer learns its own feature transform
        self.gnn_mlps = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.gnn_mlps.append(mlp)

        # GNN layers: build the convolution selected by gnn_type
        self.gnns = nn.ModuleList()
        if self.gnn_type == 'gine':
            for layer_idx in range(num_layers):
                gnn_layer = GINEConv(
                    nn=self.gnn_mlps[layer_idx],
                    train_eps=True,  # learnable epsilon parameter
                    edge_dim=hidden_dim if edge_input_dim > 0 else None
                )
                self.gnns.append(gnn_layer)
        elif self.gnn_type == 'gcn':
            for _ in range(num_layers):
                self.gnns.append(GCNConv(hidden_dim, hidden_dim))
        elif self.gnn_type == 'gat':
            for _ in range(num_layers):
                self.gnns.append(GATConv(hidden_dim, hidden_dim // self.heads,
                                         heads=self.heads, concat=True))
        elif self.gnn_type == 'sage':
            for _ in range(num_layers):
                self.gnns.append(SAGEConv(hidden_dim, hidden_dim))
        elif self.gnn_type == 'edge':
            for _ in range(num_layers):
                nn_edge = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )
                self.gnns.append(EdgeConv(nn_edge))
        elif self.gnn_type == 'transformer':
            for _ in range(num_layers):
                self.gnns.append(TransformerConv(hidden_dim, hidden_dim,
                                                 heads=self.heads,
                                                 edge_dim=hidden_dim if edge_input_dim > 0 else None,
                                                 concat=self.concat))
        elif self.gnn_type == 'resgated':
            for _ in range(num_layers):
                self.gnns.append(ResGatedGraphConv(hidden_dim, hidden_dim))
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

        # Layer attention weights
        self.attention_weights = nn.Parameter(torch.ones(num_layers + 1))
        # Edge attention mechanism
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Global attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=True)
        )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # expanded dimension
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim * 2),  # LayerNorm for stability
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def _initialize_weights(self):
        """Initialize weights of linear, normalization and embedding layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                init.normal_(m.weight, mean=0, std=0.1)

    def compute_edge_attention(self, x, edge_index, edge_attr=None):
        """Compute edge attention weights from concatenated endpoint features."""
        src, dst = edge_index

        # Check whether the indices are within the valid range
        num_nodes = x.size(0)
        valid_src = src < num_nodes
        valid_dst = dst < num_nodes
        valid_edges = valid_src & valid_dst

        if not torch.all(valid_edges):
            # Drop invalid edges
            src = src[valid_edges]
            dst = dst[valid_edges]

            if edge_attr is not None:
                edge_attr = edge_attr[valid_edges]

        # Guard against out-of-bounds access
        src = torch.clamp(src, 0, num_nodes - 1)
        dst = torch.clamp(dst, 0, num_nodes - 1)

        edge_features = torch.cat([x[src], x[dst]], dim=-1)
        if edge_attr is not None:
            edge_features = torch.cat([edge_features, edge_attr], dim=-1)

        attention_scores = self.edge_attention(edge_features)
        attention_weights = torch.sigmoid(attention_scores)

        return attention_weights, src, dst, edge_attr

    def _encode_edges_with_attention(self, x, edge_index, edge_attr):
        """Encode edge features and compute edge attention weights."""
        edge_attention_weights = None
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)
            if edge_attr is not None:
                edge_attention_weights, src, dst, edge_attr = self.compute_edge_attention(
                    x, edge_index, edge_attr)
                # Keep only valid edges
                edge_index = torch.stack([src, dst], dim=0)
        return edge_index, edge_attr, edge_attention_weights

    def _apply_gnn_layers(self, x, edge_index, edge_attr, edge_attention_weights):
        """Apply the GNN layers with residuals; collect per-layer features."""
        layer_features = [x]

        for i in range(self.num_layers):
            x_res = x  # residual connection

            # Call the convolution selected by gnn_type
            if self.gnn_type == 'gine':
                # GINEConv with edge features
                if edge_attr is not None:
                    # Weight edge features by the edge attention if available
                    if edge_attention_weights is not None:
                        edge_features = edge_attr * edge_attention_weights
                    else:
                        edge_features = edge_attr

                    x = self.gnns[i](x, edge_index, edge_features)
                else:
                    x = self.gnns[i](x, edge_index)
            elif self.gnn_type == 'transformer':
                if edge_attr is not None:
                    x = self.gnns[i](x, edge_index, edge_attr=edge_attr)
                else:
                    x = self.gnns[i](x, edge_index)
            else:
                # gcn / gat / sage / edge / resgated do not use edge features
                x = self.gnns[i](x, edge_index)

            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # residual connection
            layer_features.append(x)

        return layer_features

    def _attention_pool(self, layer_features, batch, batch_size):
        """Layer attention weighting followed by graph-level attention pooling."""
        layer_features = torch.stack(layer_features, dim=0)
        attention_weights = F.softmax(self.attention_weights, dim=0)
        x_weighted = torch.einsum('i,i...->...', attention_weights, layer_features)

        attention_scores = self.attention_pool(x_weighted)
        attention_scores = attention_scores.squeeze(-1)
        exp_scores = torch.exp(attention_scores - attention_scores.max())
        batch_exp_sum = torch.zeros(batch_size, device=x_weighted.device)
        batch_exp_sum = batch_exp_sum.scatter_add(0, batch, exp_scores)
        norm_weights = exp_scores / (batch_exp_sum[batch] + 1e-8)

        weighted_features = x_weighted * norm_weights.unsqueeze(-1)
        graph_embeddings = torch.zeros(batch_size, self.hidden_dim,
                                       device=x_weighted.device)
        graph_embeddings = graph_embeddings.scatter_add(
            0, batch.unsqueeze(-1).expand(-1, self.hidden_dim), weighted_features)

        return x_weighted, graph_embeddings

    def _combine_pooled_features(self, x_weighted, graph_embeddings, batch):
        """Concatenate attention, mean and sum pooling features."""
        global_mean = global_mean_pool(x_weighted, batch)
        global_sum = global_add_pool(x_weighted, batch)  # sum pooling

        combined_features = torch.cat([
            graph_embeddings,  # attention pooling features
            global_mean,  # mean pooling features
            global_sum  # sum pooling features
        ], dim=1)

        return combined_features
