from models.gnn_components import ThermalGNNBase


class ThermalTAGNN(ThermalGNNBase):
    """Text-attributed GNN branch (autoencoder-reduced LLM embedding features)."""

    def __init__(self, gnn_type, node_input_dim, edge_input_dim=0, hidden_dim=128,
                 output_dim=1, num_layers=4, dropout=0.3):
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
        """
        super().__init__()

        self._set_gnn_type(gnn_type)

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Node/edge encoders and the shared GNN stack
        self._build_encoders(node_input_dim, hidden_dim, dropout, edge_input_dim)
        self._build_shared_modules(hidden_dim, num_layers, dropout, edge_input_dim, gnn_type)

        # Initialize weights
        self._initialize_weights()

    def forward(self, data, return_embedding=False):
        """
        Forward pass.

        Args:
            data: PyG Data object containing x, edge_index, edge_attr, batch
            return_embedding: if True, also return the pooled features

        Returns:
            output: predicted thermal conductivity
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        batch_size = int(batch.max().item() + 1)  # number of graphs in this batch

        edge_attr = data.edge_attr

        # Node feature normalization and encoding
        x = self.node_bn(x)
        x = self.node_encoder(x)

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

        if return_embedding:
            return output, combined_features
        else:
            return output


def create_TAGNN_model(gnn_type='gine', input_dim=18, edge_dim=0, hidden_dim=64,
                       output_dim=1, num_layers=3, dropout=0.2):
    """
    Create a text-attributed GNN model.

    Args:
        gnn_type: convolution type ('gine', 'gcn', 'gat', 'edge', 'sage',
            'transformer', 'resgated')
        input_dim: node feature dimension
        edge_dim: edge feature dimension
        hidden_dim: hidden layer dimension
        output_dim: output dimension
        num_layers: number of GNN layers
        dropout: dropout rate

    Returns:
        model: the created model
    """
    model = ThermalTAGNN(
        gnn_type=gnn_type,
        node_input_dim=input_dim,
        edge_input_dim=edge_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    return model
