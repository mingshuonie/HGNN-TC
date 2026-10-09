import json
import os
import pickle
import re
import warnings
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

# Suppress warnings emitted by the graph libraries (kept after their imports,
# matching the original module-loading behavior)
warnings.filterwarnings("ignore")


class NodeAutoencoder(nn.Module):
    """Node-feature autoencoder."""

    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class EdgeAutoencoder(nn.Module):
    """Edge-feature autoencoder."""

    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class TA_thermalConductivityDataset(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None,
                 node_latent_dim: int = 32, edge_latent_dim: int = 16,
                 use_autoencoder: bool = True, retrain_autoencoder: bool = False):

        self.root = root

        self.node_latent_dim = node_latent_dim
        self.edge_latent_dim = edge_latent_dim
        self.use_autoencoder = use_autoencoder
        self.retrain_autoencoder = retrain_autoencoder
        self.node_scaler = None
        self.edge_scaler = None
        self.node_ae = None
        self.edge_ae = None

        # Collect GML files
        self.gml_files = []
        for f in os.listdir(self.raw_dir):
            if f.endswith('.gml'):
                self.gml_files.append(f)
        self.gml_files.sort()

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """Raw file names."""
        return self.gml_files

    @property
    def processed_file_names(self):
        """Processed file names."""
        if self.use_autoencoder:
            return [f'data_ae_node{self.node_latent_dim}_edge{self.edge_latent_dim}.pt']
        else:
            return ['data.pt']

    def _get_autoencoder_config_path(self):
        """Autoencoder config path."""
        return os.path.join(self.processed_dir,
                            f'autoencoder_config_node{self.node_latent_dim}_edge{self.edge_latent_dim}.json')

    def _get_autoencoder_model_path(self, model_type='node'):
        """Autoencoder model save path."""
        return os.path.join(self.processed_dir,
                            f'{model_type}_ae_node{self.node_latent_dim}_edge{self.edge_latent_dim}.pth')

    def _get_scaler_path(self, scaler_type='node'):
        """Scaler save path."""
        return os.path.join(self.processed_dir,
                            f'{scaler_type}_scaler_node{self.node_latent_dim}_edge{self.edge_latent_dim}.pkl')

    def download(self):
        pass

    def _collect_all_features(self, data_list):
        """Collect all node/edge features for autoencoder training."""
        all_node_features = []
        all_edge_features = []

        for data in data_list:
            if data.x is not None and data.x.shape[0] > 0:
                all_node_features.append(data.x.numpy())
            if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
                all_edge_features.append(data.edge_attr.numpy())

        # Stack into matrices
        if all_node_features:
            node_features = np.vstack(all_node_features)
        else:
            node_features = np.array([])

        if all_edge_features:
            edge_features = np.vstack(all_edge_features)
        else:
            edge_features = np.array([])

        return node_features, edge_features

    def _train_node_autoencoder(self, node_features, device='cpu'):
        """Train the node-feature autoencoder."""
        print(f"Training node-feature autoencoder, input dim: {node_features.shape[1]}, latent dim: {self.node_latent_dim}")

        # Standardize
        self.node_scaler = StandardScaler()
        node_features_scaled = self.node_scaler.fit_transform(node_features)

        node_tensor = torch.FloatTensor(node_features_scaled)

        self.node_ae = NodeAutoencoder(node_features.shape[1], self.node_latent_dim).to(device)
        optimizer = torch.optim.Adam(self.node_ae.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training settings
        batch_size = 256
        num_epochs = 100

        dataset = torch.utils.data.TensorDataset(node_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.node_ae.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                batch_data = batch[0].to(device)

                optimizer.zero_grad()
                encoded, decoded = self.node_ae(batch_data)
                loss = criterion(decoded, batch_data)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"  Node autoencoder Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.6f}")

        return self.node_ae

    def _train_edge_autoencoder(self, edge_features, device='cpu'):
        """Train the edge-feature autoencoder."""
        print(f"Training edge-feature autoencoder, input dim: {edge_features.shape[1]}, latent dim: {self.edge_latent_dim}")

        # Standardize
        self.edge_scaler = StandardScaler()
        edge_features_scaled = self.edge_scaler.fit_transform(edge_features)

        edge_tensor = torch.FloatTensor(edge_features_scaled)

        self.edge_ae = EdgeAutoencoder(edge_features.shape[1], self.edge_latent_dim).to(device)
        optimizer = torch.optim.Adam(self.edge_ae.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training settings
        batch_size = 256
        num_epochs = 100

        dataset = torch.utils.data.TensorDataset(edge_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.edge_ae.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                batch_data = batch[0].to(device)

                optimizer.zero_grad()
                encoded, decoded = self.edge_ae(batch_data)
                loss = criterion(decoded, batch_data)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"  Edge autoencoder Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.6f}")

        return self.edge_ae

    def _apply_autoencoder(self, data, device='cpu'):
        """Reduce node/edge features with the trained autoencoders."""
        with torch.no_grad():
            if data.x is not None and data.x.shape[0] > 0 and self.node_ae is not None:
                if self.node_scaler is not None:
                    node_features_np = data.x.numpy()
                    node_features_scaled = self.node_scaler.transform(node_features_np)
                    node_tensor = torch.FloatTensor(node_features_scaled).to(device)
                else:
                    node_tensor = data.x.to(device)

                self.node_ae.eval()
                encoded_nodes, _ = self.node_ae(node_tensor)
                data.x = encoded_nodes.cpu()

            if data.edge_attr is not None and data.edge_attr.shape[0] > 0 and self.edge_ae is not None:
                if self.edge_scaler is not None:
                    edge_features_np = data.edge_attr.numpy()
                    edge_features_scaled = self.edge_scaler.transform(edge_features_np)
                    edge_tensor = torch.FloatTensor(edge_features_scaled).to(device)
                else:
                    edge_tensor = data.edge_attr.to(device)

                self.edge_ae.eval()
                encoded_edges, _ = self.edge_ae(edge_tensor)
                data.edge_attr = encoded_edges.cpu()

        return data

    def parse_gml_file(self, file_path: str):
        """Parse one GML file into a PyG Data object."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Graph label (thermal conductivity)
        label_match = re.search(r'label\s+"([^"]+)"', content)
        if label_match:
            try:
                label = float(label_match.group(1))
            except Exception:
                label = 0.0
        else:
            label = 0.0
        # Parse nodes
        node_pattern = r'node\s*\[([^\]]+)\]'
        node_matches = re.findall(node_pattern, content, re.DOTALL)

        nodes = []
        node_ids = {}
        node_id_counter = 0

        for node_str in node_matches:
            # Node id
            id_match = re.search(r'id\s+(\d+)', node_str)
            if not id_match:
                continue

            node_id = int(id_match.group(1))

            # Node features
            features = []

            # Match string-form features, e.g. features "[1273.15]"
            features_match = re.search(r'features\s+"([^"]*)', node_str)

            if features_match:
                # Parse the feature string
                features_str = features_match.group(1)
                if features_str:
                    features_str = features_str.replace('[', '')

                    if ',' not in features_str:
                        features_list = features_str
                    else:
                        features_list = features_str.replace(' ', '').split(',')

                    for feat in features_list:
                        try:
                            features.append(float(feat))
                        except Exception:
                            continue

            nodes.append(features)
            node_ids[node_id] = node_id_counter
            node_id_counter += 1

        # Graph name
        name_match = re.search(r'name\s+"([^"]+)"', content)
        if name_match:
            try:
                name = name_match.group(1)
            except Exception:
                name = 'name'
        else:
            name = 0.0

        # Parse edges
        edge_pattern = r'edge\s*\[([^\]]+)\]'
        edge_matches = re.findall(edge_pattern, content, re.DOTALL)

        edges = []
        edge_attrs = []

        for edge_str in edge_matches:
            # Source and target
            source_match = re.search(r'source\s+(\d+)', edge_str)
            target_match = re.search(r'target\s+(\d+)', edge_str)

            if not source_match or not target_match:
                continue

            source = int(source_match.group(1))
            target = int(target_match.group(1))

            # Skip edges referencing unknown nodes
            if source not in node_ids or target not in node_ids:
                continue

            edges.append((node_ids[source], node_ids[target]))

            # Edge features
            edge_features = []
            edge_features_match = re.search(r'features\s+"([^"]*)', edge_str)
            if edge_features_match:
                edge_features_str = edge_features_match.group(1)

                if edge_features_str:
                    edge_features_list = edge_features_str.replace(' ', '').replace('[', '').split(',')
                    for edge_feat in edge_features_list:
                        try:
                            if edge_feat:
                                edge_features.append(float(edge_feat))
                        except Exception:
                            continue
            edge_attrs.append(edge_features)

        # Pad node features to a common width
        max_node_dim = max(len(feats) for feats in nodes)
        padded_nodes = []
        for feats in nodes:
            if len(feats) < max_node_dim:
                padded = feats + [0.0] * (max_node_dim - len(feats))
            else:
                padded = feats[:max_node_dim]
            padded_nodes.append(padded)

        x = torch.tensor(padded_nodes, dtype=torch.float)

        if not edges:
            # Empty edge tensors when there are no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        else:
            # Edge index
            edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)

            # Pad edge features to a common width
            max_edge_dim = max(len(feats) for feats in edge_attrs)
            padded_edge_attrs = []
            for feats in edge_attrs:
                if len(feats) < max_edge_dim:
                    padded = feats + [0.0] * (max_edge_dim - len(feats))
                else:
                    padded = feats[:max_edge_dim]
                padded_edge_attrs.append(padded)

            edge_attr = torch.tensor(padded_edge_attrs, dtype=torch.float)

        # Graph label
        y = torch.tensor([[label]], dtype=torch.float)

        # Build the PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            graph_name=name
        )

        return data

    def process(self):
        """Process all GML files into the cached dataset."""
        data_list = []

        print("Parsing raw data...")
        for idx, filename in enumerate(self.gml_files):
            if idx % 100 == 0:
                print(f"Processing files: {idx}/{len(self.gml_files)}")

            file_path = os.path.join(self.raw_dir, filename)
            try:
                data = self.parse_gml_file(file_path)
                data_list.append(data)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        # Reduce features with autoencoders
        if self.use_autoencoder and len(data_list) > 0:
            print("\nReducing feature dimensionality with autoencoders...")

            # Load cached autoencoders if available
            config_path = self._get_autoencoder_config_path()
            node_model_path = self._get_autoencoder_model_path('node')
            edge_model_path = self._get_autoencoder_model_path('edge')

            if (not self.retrain_autoencoder and
                    os.path.exists(config_path) and
                    os.path.exists(node_model_path) and
                    os.path.exists(edge_model_path)):

                print("Loading trained autoencoders...")
                # Load config
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Check config compatibility
                if (config.get('node_latent_dim') == self.node_latent_dim and
                        config.get('edge_latent_dim') == self.edge_latent_dim):

                    # Feature dims determine the autoencoder input
                    node_features, edge_features = self._collect_all_features(data_list)

                    if node_features.size > 0:
                        # Load node autoencoder
                        self.node_ae = NodeAutoencoder(node_features.shape[1], self.node_latent_dim)
                        self.node_ae.load_state_dict(torch.load(node_model_path, map_location='cpu'))
                        self.node_ae.eval()

                        with open(self._get_scaler_path('node'), 'rb') as f:
                            self.node_scaler = pickle.load(f)

                    if edge_features.size > 0:
                        # Load edge autoencoder
                        self.edge_ae = EdgeAutoencoder(edge_features.shape[1], self.edge_latent_dim)
                        self.edge_ae.load_state_dict(torch.load(edge_model_path, map_location='cpu'))
                        self.edge_ae.eval()

                        with open(self._get_scaler_path('edge'), 'rb') as f:
                            self.edge_scaler = pickle.load(f)

                    print("Autoencoders loaded")
                else:
                    print("Config mismatch, retraining autoencoders...")
                    self.retrain_autoencoder = True
            else:
                self.retrain_autoencoder = True

            # Train new autoencoders
            if self.retrain_autoencoder:
                node_features, edge_features = self._collect_all_features(data_list)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Using device: {device}")

                if node_features.size > 0 and node_features.shape[1] > 0:
                    self._train_node_autoencoder(node_features, device)
                    torch.save(self.node_ae.state_dict(), node_model_path)

                    with open(self._get_scaler_path('node'), 'wb') as f:
                        pickle.dump(self.node_scaler, f)

                if edge_features.size > 0 and edge_features.shape[1] > 0:
                    self._train_edge_autoencoder(edge_features, device)
                    torch.save(self.edge_ae.state_dict(), edge_model_path)

                    with open(self._get_scaler_path('edge'), 'wb') as f:
                        pickle.dump(self.edge_scaler, f)

                # Save config
                config = {
                    'node_latent_dim': self.node_latent_dim,
                    'edge_latent_dim': self.edge_latent_dim
                }
                with open(config_path, 'w') as f:
                    json.dump(config, f)

                print("Autoencoder training complete and saved")

            # Apply the autoencoders
            print("Applying autoencoders for feature reduction...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for i, data in enumerate(data_list):
                if i % 100 == 0:
                    print(f"Reducing features: {i}/{len(data_list)}")
                data = self._apply_autoencoder(data, device)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save the processed dataset
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_dataset_statistics(self):
        """Summarize the dataset (sampled over the first 100 graphs)."""
        num_nodes_list = []
        num_edges_list = []
        node_feat_dims = []
        edge_feat_dims = []
        labels = []

        for i in range(min(100, len(self))):
            data = self[i]
            num_nodes_list.append(data.num_nodes)
            num_edges_list.append(data.num_edges)
            node_feat_dims.append(data.x.shape[1] if data.x is not None and data.x.shape[0] > 0 else 0)
            if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
                edge_feat_dims.append(data.edge_attr.shape[1])
            labels.append(data.y.item())

        # Average feature dims, ignoring zeros
        valid_node_dims = [dim for dim in node_feat_dims if dim > 0]
        valid_edge_dims = [dim for dim in edge_feat_dims if dim > 0]

        stats = {
            'num_graphs': len(self),
            'avg_nodes': np.mean(num_nodes_list) if num_nodes_list else 0,
            'avg_edges': np.mean(num_edges_list) if num_edges_list else 0,
            'node_feat_dim': int(np.mean(valid_node_dims)) if valid_node_dims else 0,
            'edge_feat_dim': int(np.mean(valid_edge_dims)) if valid_edge_dims else 0,
            'label_range': (min(labels), max(labels)) if labels else (0, 0),
            'label_mean': np.mean(labels) if labels else 0,
            'label_std': np.std(labels) if labels else 0,
            'num_node_types': len(self.node_type_mapping) if hasattr(self, 'node_type_mapping') else 0,
            'num_edge_types': len(self.edge_type_mapping) if hasattr(self, 'edge_type_mapping') else 0,
        }

        return stats


def create_text_data_loaders(data_dir, batch_size=32, train_ratio=0.7, val_ratio=0.15, seed=42,
                            node_latent_dim=256, edge_latent_dim=128, use_autoencoder=True, retrain_autoencoder=False):
    """Create train/val/test loaders and return dataset statistics."""
    print("Loading dataset...")
    dataset = TA_thermalConductivityDataset(root=data_dir,
                                            node_latent_dim=node_latent_dim,
                                            edge_latent_dim=edge_latent_dim,
                                            use_autoencoder=use_autoencoder,
                                            retrain_autoencoder=retrain_autoencoder
                                            )

    dataset_stats = dataset.get_dataset_statistics()
    print("\nDataset statistics:")
    print(f"  Number of graphs: {dataset_stats['num_graphs']}")
    print(f"  Average nodes: {dataset_stats['avg_nodes']:.2f}")
    print(f"  Average edges: {dataset_stats['avg_edges']:.2f}")
    print(f"  Node feature dim: {dataset_stats['node_feat_dim']}")
    print(f"  Edge feature dim: {dataset_stats['edge_feat_dim']}")

    print(f"  Label range: {dataset_stats['label_range'][0]:.2f} - {dataset_stats['label_range'][1]:.2f}")
    print(f"  Label mean: {dataset_stats['label_mean']:.2f}, std: {dataset_stats['label_std']:.2f}")
    # Train/val/test split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    # Deterministic split
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    print(f"\nDataset split:")
    print(f"  Train set: {len(train_indices)} graphs")
    print(f"  Validation set: {len(val_indices)} graphs")
    print(f"  Test set: {len(test_indices)} graphs")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset_stats
