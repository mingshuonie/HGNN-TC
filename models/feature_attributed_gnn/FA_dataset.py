import os
import re
import warnings
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

# Suppress warnings emitted by the graph libraries (kept after their imports,
# matching the original module-loading behavior)
warnings.filterwarnings("ignore")


class FA_thermalConductivityDataset(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None,
                 standardize_features: bool = True):
        self.root = root
        self.standardize_features = standardize_features

        # Collect GML files
        self.gml_files = []
        for f in os.listdir(self.raw_dir):
            if f.endswith('.gml'):
                self.gml_files.append(f)
        self.gml_files.sort()

        # Node/edge types seen across all graphs
        self.all_node_types = set()
        self.all_edge_types = set()

        # Feature standardization statistics
        self.node_feat_mean = None
        self.node_feat_std = None
        self.edge_feat_mean = None
        self.edge_feat_std = None

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

        # Load standardization statistics if present
        stats_path = os.path.join(self.processed_dir, 'feature_stats.pt')
        if os.path.exists(stats_path) and self.standardize_features:
            stats = torch.load(stats_path)
            self.node_feat_mean = stats.get('node_feat_mean')
            self.node_feat_std = stats.get('node_feat_std')
            self.edge_feat_mean = stats.get('edge_feat_mean')
            self.edge_feat_std = stats.get('edge_feat_std')
            print(f"Loaded feature standardization statistics")

    @property
    def raw_file_names(self):
        """Raw file names."""
        return self.gml_files

    @property
    def processed_file_names(self):
        """Processed file names."""
        return ['data.pt']

    def download(self):
        pass

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
        node_types = []  # node type strings

        for node_str in node_matches:
            # Node id
            id_match = re.search(r'id\s+(\d+)', node_str)
            if not id_match:
                continue

            node_id = int(id_match.group(1))
            # Node type
            node_type_match = re.search(r'node_type\s+"([^"]+)"', node_str)
            node_type = node_type_match.group(1) if node_type_match else "unknown"
            node_types.append(node_type)

            self.all_node_types.add(node_type)

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
        edge_pattern = r'edge\s*\[\s*((?:[^\[\]]|\[.*?\])*)\s*\]'
        edge_matches = re.findall(edge_pattern, content, re.DOTALL)

        edges = []
        edge_attrs = []
        edge_types = []  # edge type strings

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

            # Edge type
            edge_type_match = re.search(r'edge_type\s+"([^"]+)"', edge_str)
            edge_type = edge_type_match.group(1) if edge_type_match else "unknown"
            edge_types.append(edge_type)

            self.all_edge_types.add(edge_type)

            # Edge features
            edge_features = []
            edge_features_match = re.search(r'features\s+"([^"]*)', edge_str)

            if edge_features_match:
                edge_features_str = edge_features_match.group(1)
                edge_features_list = edge_features_str.replace(' ', '').split(',')
                for feat in edge_features_list:
                    if feat:
                        clean_feat = feat.strip('[]')
                        edge_features.append(float(clean_feat))

            edge_attrs.append(edge_features)

        # Pad node features to a common width
        max_node_dim = max(len(feats) for feats in nodes)
        padded_nodes = []
        for feats in nodes:
            if len(feats) < max_node_dim:
                padded = feats + [0.0] * (max_node_dim - len(feats))
            else:
                padded = feats[:max_node_dim]

            if feats:
                padded_nodes.append(padded)

        x = torch.tensor(padded_nodes, dtype=torch.float)

        if not edges:
            # Empty edge tensors when there are no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        else:
            # Edge index
            edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)

            # Fixed edge feature width
            FIXED_EDGE_DIM = 7

            # Pad/truncate edge features to the fixed width
            # max_edge_dim = max(len(feats) for feats in edge_attrs)

            padded_edge_attrs = []
            for feats in edge_attrs:
                if len(feats) < FIXED_EDGE_DIM:
                    padded = feats + [0.0] * (FIXED_EDGE_DIM - len(feats))
                else:
                    padded = feats[:FIXED_EDGE_DIM]
                padded_edge_attrs.append(padded)

            edge_attr = torch.tensor(padded_edge_attrs, dtype=torch.float)

        # Graph label
        # y = torch.tensor([[np.log(label)]], dtype=torch.float)
        y = torch.tensor([[label]], dtype=torch.float)

        # Build the PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            node_type_strs=node_types,  # temporary; converted to indices in process()
            edge_type_strs=edge_types,  # temporary; converted to indices in process()
            graph_name=name
        )

        return data

    def compute_feature_statistics(self, data_list: List[Data]) -> Tuple[Dict, Dict]:
        """Compute node/edge feature means and standard deviations."""
        # Collect all node/edge features
        all_node_features = []
        all_edge_features = []

        for data in data_list:
            if data.x is not None and data.x.shape[0] > 0 and data.x.shape[1] > 0:
                all_node_features.append(data.x)

            if data.edge_attr is not None and data.edge_attr.shape[0] > 0 and data.edge_attr.shape[1] > 0:
                all_edge_features.append(data.edge_attr)

        # Node statistics
        node_stats = {}
        if all_node_features:
            node_features = torch.cat(all_node_features, dim=0)
            node_mean = node_features.mean(dim=0)
            node_std = node_features.std(dim=0)

            # Avoid division by zero
            node_std[node_std == 0] = 1.0

            node_stats = {
                'mean': node_mean,
                'std': node_std
            }
            print(f"Node features: mean shape {node_mean.shape}, std shape {node_std.shape}")

        # Edge statistics
        edge_stats = {}
        if all_edge_features:
            edge_features = torch.cat(all_edge_features, dim=0)
            edge_mean = edge_features.mean(dim=0)
            edge_std = edge_features.std(dim=0)

            # Avoid division by zero
            edge_std[edge_std == 0] = 1.0

            edge_stats = {
                'mean': edge_mean,
                'std': edge_std
            }
            print(f"Edge features: mean shape {edge_mean.shape}, std shape {edge_std.shape}")

        return node_stats, edge_stats

    def standardize_features_in_data(self, data_list: List[Data], node_stats: Dict, edge_stats: Dict) -> List[Data]:
        """Standardize node/edge features in the data list."""
        standardized_data_list = []

        for i, data in enumerate(data_list):
            try:
                # Copy all attributes
                standardized_data = Data()
                for key in data.keys():
                    setattr(standardized_data, key, getattr(data, key))

                # Standardize node features
                if (data.x is not None and data.x.numel() > 0 and data.x.shape[1] > 0 and
                        node_stats and 'mean' in node_stats and 'std' in node_stats):
                    # Skip if dimensions mismatch
                    if data.x.shape[1] == node_stats['mean'].shape[0]:
                        # Small epsilon against division by zero
                        eps = 1e-8
                        standardized_data.x = (data.x - node_stats['mean']) / (node_stats['std'] + eps)
                    else:
                        print(
                            f"Warning ({i}): node feature dimension mismatch: {data.x.shape[1]} != {node_stats.get('mean', torch.tensor([])).shape[0]}")

                # Standardize edge features
                if (data.edge_attr is not None and data.edge_attr.numel() > 0 and data.edge_attr.shape[1] > 0 and
                        edge_stats and 'mean' in edge_stats and 'std' in edge_stats):
                    # Skip if dimensions mismatch
                    if data.edge_attr.shape[1] == edge_stats['mean'].shape[0]:
                        # Small epsilon against division by zero
                        eps = 1e-8
                        standardized_data.edge_attr = (data.edge_attr - edge_stats['mean']) / (edge_stats['std'] + eps)
                    else:
                        print(
                            f"Warning ({i}): edge feature dimension mismatch: {data.edge_attr.shape[1]} != {edge_stats.get('mean', torch.tensor([])).shape[0]}")

                standardized_data_list.append(standardized_data)
            except Exception as e:
                print(f"Error processing data {i}: {e}")
                standardized_data_list.append(data)

        return standardized_data_list

    def process(self):
        """Process all GML files into the cached dataset."""
        data_list = []

        # First pass: collect all type strings
        for idx, filename in enumerate(self.gml_files):
            if idx % 100 == 0:
                print(f"Scanning files: {idx}/{len(self.gml_files)}")

            file_path = os.path.join(self.raw_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                node_type_matches = re.findall(r'node_type\s+"([^"]+)"', content)
                self.all_node_types.update(node_type_matches)

                edge_type_matches = re.findall(r'edge_type\s+"([^"]+)"', content)
                self.all_edge_types.update(edge_type_matches)
            except Exception as e:
                print(f"Error scanning {filename}: {e}")
                continue

        # Map type strings to indices
        node_type_mapping = {type_str: idx for idx, type_str in enumerate(sorted(self.all_node_types))}
        edge_type_mapping = {type_str: idx for idx, type_str in enumerate(sorted(self.all_edge_types))}
        print(f"Found {len(node_type_mapping)} node types: {list(node_type_mapping.keys())}")
        print(f"Found {len(edge_type_mapping)} edge types: {list(edge_type_mapping.keys())}")

        self.node_type_mapping = node_type_mapping
        self.edge_type_mapping = edge_type_mapping

        for idx, filename in enumerate(self.gml_files):
            if idx % 100 == 0:
                print(f"Processing files: {idx}/{len(self.gml_files)}")

            file_path = os.path.join(self.raw_dir, filename)
            data = self.parse_gml_file(file_path)

            # Convert type strings to integer indices
            node_type_indices = [node_type_mapping.get(t, 0) for t in data.node_type_strs]
            edge_type_indices = [edge_type_mapping.get(t, 0) for t in data.edge_type_strs]

            data.node_type = torch.tensor(node_type_indices, dtype=torch.long)
            data.edge_type = torch.tensor(edge_type_indices, dtype=torch.long)

            # Drop the temporary string attributes
            delattr(data, 'node_type_strs')
            delattr(data, 'edge_type_strs')

            data_list.append(data)

        node_stats, edge_stats = self.compute_feature_statistics(data_list)
        # Persist statistics for later standardization
        stats_to_save = {
            'node_feat_mean': node_stats.get('mean') if node_stats else None,
            'node_feat_std': node_stats.get('std') if node_stats else None,
            'edge_feat_mean': edge_stats.get('mean') if edge_stats else None,
            'edge_feat_std': edge_stats.get('std') if edge_stats else None
        }
        torch.save(stats_to_save, os.path.join(self.processed_dir, 'feature_stats.pt'))

        # Standardize if requested
        if self.standardize_features and (node_stats or edge_stats):
            print("Standardizing features...")
            data_list = self.standardize_features_in_data(data_list, node_stats, edge_stats)
            print("Feature standardization complete")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save the processed dataset
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # Save type mappings
        torch.save({
            'node_type_mapping': node_type_mapping,
            'edge_type_mapping': edge_type_mapping
        }, os.path.join(self.processed_dir, 'type_mappings.pt'))

    def get(self, idx: int) -> Data:
        """Get a single (already standardized) graph."""
        data = super().get(idx)
        return data

    def get_dataset_statistics(self):
        """Summarize the dataset (sampled over the first 100 graphs)."""
        num_nodes_list = []
        num_edges_list = []
        node_feat_dims = []
        edge_feat_dims = []
        labels = []
        node_type_counts = defaultdict(int)
        edge_type_counts = defaultdict(int)

        for i in range(min(100, len(self))):
            data = self[i]
            num_nodes_list.append(data.num_nodes)
            num_edges_list.append(data.num_edges)
            node_feat_dims.append(data.x.shape[1] if data.x is not None and data.x.shape[0] > 0 else 0)
            if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
                edge_feat_dims.append(data.edge_attr.shape[1])
            labels.append(data.y.item())

            if hasattr(data, 'node_type'):
                for node_type_idx in data.node_type.tolist():
                    node_type_counts[node_type_idx] += 1

            if hasattr(data, 'edge_type'):
                for edge_type_idx in data.edge_type.tolist():
                    edge_type_counts[edge_type_idx] += 1

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
            'node_type_distribution': dict(node_type_counts),
            'edge_type_distribution': dict(edge_type_counts)
        }

        return stats

    def standardize_new_data(self, data: Data) -> Data:
        """Standardize new data using the stored statistics."""
        if not self.standardize_features:
            return data

        standardized_data = Data()
        for key in data.keys():
            setattr(standardized_data, key, getattr(data, key))

        if (data.x is not None and data.x.shape[0] > 0 and data.x.shape[1] > 0 and
                self.node_feat_mean is not None and self.node_feat_std is not None):
            if data.x.shape[1] == self.node_feat_mean.shape[0]:
                standardized_data.x = (data.x - self.node_feat_mean) / self.node_feat_std
            else:
                print(f"Warning: new data node feature dimension mismatch: {data.x.shape[1]} != {self.node_feat_mean.shape[0]}")

        if (data.edge_attr is not None and data.edge_attr.shape[0] > 0 and data.edge_attr.shape[1] > 0 and
                self.edge_feat_mean is not None and self.edge_feat_std is not None):
            if data.edge_attr.shape[1] == self.edge_feat_mean.shape[0]:
                standardized_data.edge_attr = (data.edge_attr - self.edge_feat_mean) / self.edge_feat_std
            else:
                print(f"Warning: new data edge feature dimension mismatch: {data.edge_attr.shape[1]} != {self.edge_feat_mean.shape[0]}")

        return standardized_data


def create_feature_data_loaders(data_dir, batch_size=32, train_ratio=0.7, val_ratio=0.15, seed=42, standardize=True):
    """Create train/val/test loaders and return dataset statistics."""
    print("Loading dataset...")
    dataset = FA_thermalConductivityDataset(root=data_dir, standardize_features=standardize)

    dataset_stats = dataset.get_dataset_statistics()
    print("\nDataset statistics:")
    print(f"  Number of graphs: {dataset_stats['num_graphs']}")
    print(f"  Average nodes: {dataset_stats['avg_nodes']:.2f}")
    print(f"  Average edges: {dataset_stats['avg_edges']:.2f}")
    print(f"  Node feature dim: {dataset_stats['node_feat_dim']}")
    print(f"  Edge feature dim: {dataset_stats['edge_feat_dim']}")
    if 'num_node_types' in dataset_stats:
        print(f"  Node types: {dataset_stats['num_node_types']}")
        print(f"  Edge types: {dataset_stats['num_edge_types']}")
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

    all_dataset = Subset(dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_data_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)

    return all_data_loader, train_loader, val_loader, test_loader, dataset_stats
