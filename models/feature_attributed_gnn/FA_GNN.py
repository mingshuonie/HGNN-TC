import torch
import torch.nn as nn

from models.gnn_components import ThermalGNNBase


class ThermalFAGNN(ThermalGNNBase):
    """Feature-attributed GNN branch (element-property node/edge features)."""

    def __init__(self, gnn_type, node_input_dim, edge_input_dim=0, hidden_dim=128,
                 output_dim=1, num_layers=4, dropout=0.3, type_embedding_dim=8):
        """
        Args:
            gnn_type: convolution type ('gine', 'gcn', 'gat', 'edge', 'sage',
                'transformer', 'resgated')
            node_input_dim: node feature dimension
            edge_input_dim: edge feature dimension
            hidden_dim: hidden layer dimension
            output_dim: output dimension
            num_layers: number of GNN layers
            dropout: dropout rate
            type_embedding_dim: embedding dimension for node/edge types
        """
        super().__init__()

        self._set_gnn_type(gnn_type)

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Node and edge type embeddings (2 types each, hard-coded to match the
        # node_type / edge_type indices produced by FA_dataset)
        num_node_types = 2
        num_edge_types = 2
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        node_type_dim = type_embedding_dim
        edge_type_dim = type_embedding_dim

        # Node type embedding
        if num_node_types > 0:
            self.node_type_embedding = nn.Embedding(num_node_types, node_type_dim)
            node_input_dim += node_type_dim

        # Edge type embedding
        if num_edge_types > 0:
            self.edge_type_embedding = nn.Embedding(num_edge_types, edge_type_dim)
            edge_input_dim += edge_type_dim

        # Node/edge encoders and the shared GNN stack
        self._build_encoders(node_input_dim, hidden_dim, dropout, edge_input_dim)
        self._build_shared_modules(hidden_dim, num_layers, dropout, edge_input_dim, gnn_type)

        # Initialize weights
        self._initialize_weights()

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG Data object containing x, edge_index, edge_attr, batch

        Returns:
            output: predicted thermal conductivity
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        batch_size = int(batch.max().item() + 1)  # number of graphs in this batch

        # Node type features
        if (hasattr(data, 'node_type') and data.node_type is not None and
                self.node_type_embedding is not None):
            # Ensure node_type is an integer tensor
            node_type = data.node_type.long() if data.node_type.dtype != torch.long else data.node_type
            node_type_emb = self.node_type_embedding(node_type)

            # Ensure the dimensions match
            if x.size(0) != node_type_emb.size(0):
                node_type_emb = node_type_emb[:x.size(0)]

            x = torch.cat([x, node_type_emb], dim=-1)

        # Node feature normalization and encoding
        x = self.node_bn(x)
        x = self.node_encoder(x)

        # Edge features (with optional edge type embedding)
        edge_attr = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr

            # Edge type features
            if (hasattr(data, 'edge_type') and data.edge_type is not None and
                    self.edge_type_embedding is not None):
                # Ensure edge_type is an integer tensor
                edge_type = data.edge_type.long() if data.edge_type.dtype != torch.long else data.edge_type
                edge_type_emb = self.edge_type_embedding(edge_type)
                edge_attr = torch.cat([edge_attr, edge_type_emb], dim=-1)

        # Edge feature encoding and edge attention
        edge_index, edge_attr, edge_attention_weights = self._encode_edges_with_attention(
            x, edge_index, edge_attr)

        # GNN layers (collect per-layer features for layer attention)
        layer_features = self._apply_gnn_layers(x, edge_index, edge_attr, edge_attention_weights)

        # Layer attention weighting + graph-level attention pooling
        x_weighted, graph_embeddings = self._attention_pool(layer_features, batch, batch_size)

        # Multiple global pooling features
        combined_features = self._combine_pooled_features(x_weighted, graph_embeddings, batch)

        # Regression prediction
        output = self.regression_head(combined_features)

        return output


def create_FAGNN_model(gnn_type='gine', input_dim=18, edge_dim=0, hidden_dim=64,
                       output_dim=1, num_layers=3, dropout=0.2, type_embedding_dim=8):
    """
    Create a feature-attributed GNN model.

    Args:
        gnn_type: convolution type ('gine', 'gcn', 'gat', 'edge', 'sage',
            'transformer', 'resgated')
        input_dim: node feature dimension
        edge_dim: edge feature dimension
        hidden_dim: hidden layer dimension
        output_dim: output dimension
        num_layers: number of GNN layers
        dropout: dropout rate
        type_embedding_dim: embedding dimension for node/edge types

    Returns:
        model: the created model
    """
    model = ThermalFAGNN(
        gnn_type=gnn_type,
        node_input_dim=input_dim,
        edge_input_dim=edge_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        type_embedding_dim=type_embedding_dim
    )

    return model
