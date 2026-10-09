import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Ensure custom modules are importable regardless of the working directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.adaptive_weight_fusion_model import FusionModel
from models.feature_attributed_gnn.FA_GNN import create_FAGNN_model
from models.feature_attributed_gnn.FA_dataset import create_feature_data_loaders
from models.text_attributed_gnn.TA_GNN import create_TAGNN_model
from models.text_attributed_gnn.TA_dataset import create_text_data_loaders
from utils.print_train_process import PrintTrainProcess
from utils.training import add_common_args, run_training, save_results_json

os.environ["TORCH_USE_CUDA_DSA"] = "1"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train_epoch(fusion_model, feature_loader, text_loader, optimizer, criterion, device,
                epoch=None, total_epochs=None, clip_grad=None, weight_growth_rate=0.001,
                val_metrics=None, aux_loss_weight=0.5,
                aux_feature_weight=None, aux_text_weight=None):
    """Train one epoch.

    The auxiliary MSE terms on the individual branch outputs keep both branches
    healthy predictors instead of mutually compensating ones. Asymmetric weights
    (0.1 / 0.2 by default) retain each branch's complementary bias while the
    fused output stays better than either branch.
    """
    aux_f = aux_feature_weight if aux_feature_weight is not None else aux_loss_weight
    aux_t = aux_text_weight if aux_text_weight is not None else aux_loss_weight
    fusion_model.train()
    total_loss = 0.0
    total_samples = 0
    total_feature_weight = 0.0
    batch_count = 0

    # Apply weight growth for the scheduled strategy
    if hasattr(fusion_model, 'weight_learning_strategy') and fusion_model.learnable_weight:
        if fusion_model.weight_learning_strategy == 'scheduled':
            fusion_model.step_counter += 1
            fusion_model.apply_weight_growth(growth_rate=weight_growth_rate)

        # Use validation performance to adjust the weight for performance_aware
        elif fusion_model.weight_learning_strategy == 'performance_aware' and val_metrics is not None:
            fusion_model.performance_aware_weight_update(
                val_metrics['loss'],
                val_metrics.get('weight', 0.5)
            )

    pbar = tqdm(zip(feature_loader, text_loader),
                desc=f"Epoch {epoch + 1:3d}/{total_epochs:3d} [Train]",
                leave=False, dynamic_ncols=True,
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    for feature_batch, text_batch in pbar:
        feature_batch = feature_batch.to(device)
        text_batch = text_batch.to(device)

        # Ensure FA and TA labels agree
        assert torch.allclose(feature_batch.y, text_batch.y), "Label mismatch!"

        optimizer.zero_grad()

        # Forward pass
        output, feature_output, text_output, weight = fusion_model(feature_batch, text_batch, epoch, total_epochs)

        # Main loss + branch auxiliary losses
        loss = criterion(output, feature_batch.y)
        if aux_f > 0:
            loss = loss + aux_f * criterion(feature_output, feature_batch.y)
        if aux_t > 0:
            loss = loss + aux_t * criterion(text_output, feature_batch.y)

        # Backward pass
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), clip_grad)

        optimizer.step()

        # Accumulate statistics
        batch_size = feature_batch.num_graphs
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_feature_weight += weight
        batch_count += 1

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_weight = total_feature_weight / batch_count if batch_count > 0 else 0.0
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'weight': f'{avg_weight:.3f}'})

    pbar.close()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_weight = total_feature_weight / batch_count if batch_count > 0 else 0.0

    return avg_loss, avg_weight


def evaluate(fusion_model, feature_loader, text_loader, criterion, device,
             phase="Validation", epoch=None, total_epochs=None, fixed_weight=None):
    """Evaluate the model; optionally with a fixed fusion weight."""
    fusion_model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0
    total_feature_weight = 0.0
    batch_count = 0

    all_preds = []
    all_targets = []
    all_feature_preds = []
    all_text_preds = []

    if epoch is not None:
        desc = f"Epoch {epoch + 1:3d}/{total_epochs:3d} [{phase}]"
    else:
        desc = f"[{phase}]"

    pbar = tqdm(zip(feature_loader, text_loader), desc=desc, leave=False,
                dynamic_ncols=True, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    with torch.no_grad():
        for feature_batch, text_batch in pbar:
            # Move the data to the device
            feature_batch = feature_batch.to(device)
            text_batch = text_batch.to(device)

            # Ensure FA and TA labels agree
            assert torch.allclose(feature_batch.y, text_batch.y), "Label mismatch!"

            if fixed_weight is not None:
                output, feature_output, text_output, weight = fusion_model(
                    feature_batch, text_batch, epoch, total_epochs, use_fixed_weight=fixed_weight
                )
            else:
                output, feature_output, text_output, weight = fusion_model(
                    feature_batch, text_batch, epoch, total_epochs
                )

            loss = criterion(output, feature_batch.y)

            mae = torch.abs(output - feature_batch.y).sum().item()
            mse = torch.square(output - feature_batch.y).sum().item()

            batch_size = feature_batch.num_graphs
            total_loss += loss.item() * batch_size
            total_mae += mae
            total_mse += mse
            total_samples += batch_size
            total_feature_weight += weight
            batch_count += 1

            all_preds.extend(output.cpu().numpy().flatten())
            all_targets.extend(feature_batch.y.cpu().numpy().flatten())
            all_feature_preds.extend(feature_output.cpu().numpy().flatten())
            all_text_preds.extend(text_output.cpu().numpy().flatten())

    pbar.close()

    # Aggregate metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_mae = total_mae / total_samples if total_samples > 0 else 0.0
    rmse = np.sqrt(total_mse / total_samples) if total_samples > 0 else 0.0
    avg_weight = total_feature_weight / batch_count if batch_count > 0 else 0.0

    if len(all_targets) > 1:
        ss_res = np.sum((np.array(all_targets) - np.array(all_preds)) ** 2)
        ss_tot = np.sum((np.array(all_targets) - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    else:
        r2 = 0.0

    metrics = {
        'loss': avg_loss,
        'mae': avg_mae,
        'rmse': rmse,
        'r2': r2,
        'weight': avg_weight
    }

    return metrics, np.array(all_preds), np.array(all_targets), np.array(all_feature_preds), np.array(all_text_preds)


def main(args):
    """Main training routine."""
    # Create the results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"test_fusion_model_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    train_printer = PrintTrainProcess(device)

    train_printer.print_header("Fusion Model Training for Thermal Conductivity Prediction", 80)
    print(f"Device: {device}")
    print(f"Results directory: {save_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    train_printer.print_section("Loading Datasets")
    print("Loading Feature GNN data...")
    all_feature_data_loader, feature_train_loader, feature_val_loader, feature_test_loader, feature_dataset_stats = create_feature_data_loaders(
        data_dir=args.feature_attributed_gnn_data_dir,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        standardize=True
    )

    print("Loading Text GNN data...")
    text_train_loader, text_val_loader, text_test_loader, text_dataset_stats = create_text_data_loaders(
        data_dir=args.text_attributed_gnn_data_dir,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_autoencoder=True,
        retrain_autoencoder=True  # set False to reuse an existing autoencoder
    )

    print("Datasets loaded!")
    print(f"Feature GNN dataset statistics: {feature_dataset_stats}")
    print(f"Text GNN dataset statistics: {text_dataset_stats}")

    # Check data alignment
    print("\nChecking data alignment...")
    feature_train_size = len(feature_train_loader.dataset) if hasattr(feature_train_loader, 'dataset') else "unknown"
    text_train_size = len(text_train_loader.dataset) if hasattr(text_train_loader, 'dataset') else "unknown"
    print(f"Train set size - Feature: {feature_train_size}, Text: {text_train_size}")

    if feature_train_size != text_train_size:
        print(f"Warning: train set sizes mismatch! Feature: {feature_train_size}, Text: {text_train_size}")
    else:
        print("Train set sizes match!")

    train_printer.print_section("Creating Fusion Model")

    feature_gnn_model = create_FAGNN_model(
        gnn_type=args.feature_gnn_type,
        input_dim=feature_dataset_stats['node_feat_dim'],
        edge_dim=feature_dataset_stats['edge_feat_dim'],
        hidden_dim=args.hidden_dim,
        output_dim=1,
        num_layers=args.num_layers,
        dropout=args.dropout,
        type_embedding_dim=args.type_embedding_dim
    ).to(device)

    text_gnn_model = create_TAGNN_model(
        gnn_type=args.text_gnn_type,
        input_dim=text_dataset_stats['node_feat_dim'],
        edge_dim=text_dataset_stats['edge_feat_dim'],
        hidden_dim=args.hidden_dim,
        output_dim=1,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    fusion_model = FusionModel(
        feature_gnn_model,
        text_gnn_model,
        learnable_weight=args.learnable_weight,
        initial_weight=args.initial_weight,
        weight_learning_strategy=args.weight_learning_strategy,
        performance_window=20  # performance tracking window size
    ).to(device)

    print("Fusion model created!")
    print(f"Weight learning strategy: {args.weight_learning_strategy}")
    print(f"Initial feature weight: {fusion_model.get_weight():.4f}")
    print(f"Initial text weight: {1 - fusion_model.get_weight():.4f}")

    criterion = nn.MSELoss()

    # Joint optimizer over both branches and the fusion weight
    optimizer = optim.Adam(
        list(feature_gnn_model.parameters()) +
        list(text_gnn_model.parameters()) +
        (list(fusion_model.parameters()) if args.learnable_weight else []),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    train_losses = []
    train_weights = []
    val_losses = []
    val_maes = []
    val_rmses = []
    val_weights = []

    best_val_loss = float('inf')
    best_val_weight = args.initial_weight
    best_epoch = 0
    best_model_state = None
    best_fusion_model_state = None
    train_printer.print_header("Start Training", 80)

    print(f"\n {'Epoch':<6} {'Best':<5} {'Train Loss':<12} {'Val Loss':<12} "
          f"{'Val MAE':<10} {'Val R2':<8} {'Weight':<8} {'LR':<12} {'Time':<8}")
    print("-" * 80)

    start_time = time.time()
    previous_val_loss = None

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # performance_aware: feed the previous epoch's validation metrics
        val_metrics_for_training = None
        if epoch > 0 and fusion_model.weight_learning_strategy == 'performance_aware':
            val_metrics_for_training = {
                'loss': previous_val_loss,
                'weight': val_weights[-1] if val_weights else args.initial_weight
            }

        train_loss, train_weight = train_epoch(
            fusion_model, feature_train_loader, text_train_loader,
            optimizer, criterion, device, epoch, args.epochs, args.clip_grad,
            weight_growth_rate=args.weight_growth_rate,
            val_metrics=val_metrics_for_training,
            aux_loss_weight=args.aux_loss_weight,
            aux_feature_weight=args.aux_feature_weight,
            aux_text_weight=args.aux_text_weight
        )
        train_losses.append(train_loss)
        train_weights.append(train_weight)

        val_metrics, val_preds, val_targets, val_feature_preds, val_text_preds = evaluate(
            fusion_model, feature_val_loader, text_val_loader,
            criterion, device, "Validation", epoch, args.epochs
        )
        val_losses.append(val_metrics['loss'])
        val_maes.append(val_metrics['mae'])
        val_rmses.append(val_metrics['rmse'])
        val_weights.append(val_metrics.get('weight', train_weight))

        previous_val_loss = val_metrics['loss']

        if scheduler is not None:
            scheduler.step(val_metrics['loss'])

        # Track the best model
        current_weight = val_metrics.get('weight', train_weight)
        is_best = val_metrics['loss'] < best_val_loss

        if is_best:
            best_val_loss = val_metrics['loss']
            best_val_weight = current_weight
            best_epoch = epoch

            best_model_state = {
                'epoch': epoch + 1,
                'feature_model_state_dict': feature_gnn_model.state_dict(),
                'text_model_state_dict': text_gnn_model.state_dict(),
                'fusion_model_state_dict': fusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'val_weight': best_val_weight
            }

            # Fusion-weight parameters saved separately
            best_fusion_model_state = {
                'weight_logit': fusion_model.weight_logit.detach().clone() if hasattr(fusion_model,
                                                                                      'weight_logit') and fusion_model.weight_logit is not None else None,
                'weight': fusion_model.weight.detach().clone() if hasattr(fusion_model, 'weight') else None,
                'base_weight': fusion_model.base_weight.detach().clone() if hasattr(fusion_model,
                                                                                    'base_weight') else None,
                'temperature': fusion_model.temperature.detach().clone() if hasattr(fusion_model,
                                                                                    'temperature') else None,
            }

            torch.save(best_model_state, os.path.join(save_dir, 'best_fusion_model.pth'))
            torch.save(best_fusion_model_state, os.path.join(save_dir, 'best_fusion_weights.pth'))

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']

        print("\033[K", end='\r')
        train_printer.print_epoch_summary(epoch, args.epochs, train_loss, train_weight,
                                val_metrics, lr, epoch_time, is_best)

        # Detailed status every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            train_printer.print_section(f"Epoch {epoch + 1} Detailed Status", 60)
            print(f"Train loss: {train_loss:.6f}")
            print(f"Validation loss: {val_metrics['loss']:.6f}")
            print(f"Validation MAE: {val_metrics['mae']:.6f}")
            print(f"Validation RMSE: {val_metrics['rmse']:.6f}")
            print(f"Validation R2: {val_metrics['r2']:.6f}")
            print(f"Feature weight: {val_metrics.get('weight', train_weight):.4f}")
            print(f"Learning rate: {lr:.6f}")
            print(f"Time: {epoch_time:.2f}s")

        # Early stopping
        if epoch - best_epoch > args.patience:
            print(f"\n{'!' * 20} Early Stopping {'!' * 20}")
            print(f"Validation loss has not improved for {args.patience} epochs")
            print(f"Best epoch: {best_epoch + 1}, best validation loss: {best_val_loss:.6f}")
            break

    training_time = time.time() - start_time
    train_printer.print_header("Training Complete", 80)
    print(f"Training complete! Total time: {training_time:.2f}s ({training_time / 60:.1f}min)")
    print(f"Best model at epoch {best_epoch + 1}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best validation weight: {best_val_weight:.4f}")
    print()

    # Restore the best model state
    train_printer.print_section("Loading Best Model State")
    if best_model_state is not None:
        feature_gnn_model.load_state_dict(best_model_state['feature_model_state_dict'])
        text_gnn_model.load_state_dict(best_model_state['text_model_state_dict'])
        fusion_model.load_state_dict(best_model_state['fusion_model_state_dict'])

        if best_fusion_model_state is not None:
            if hasattr(fusion_model, 'weight_logit') and fusion_model.weight_logit is not None and \
                    best_fusion_model_state['weight_logit'] is not None:
                fusion_model.weight_logit.data.copy_(best_fusion_model_state['weight_logit'])
            if hasattr(fusion_model, 'weight') and fusion_model.weight is not None and best_fusion_model_state[
                'weight'] is not None:
                fusion_model.weight.data.copy_(best_fusion_model_state['weight'])
            if hasattr(fusion_model, 'base_weight') and fusion_model.base_weight is not None and \
                    best_fusion_model_state['base_weight'] is not None:
                fusion_model.base_weight.data.copy_(best_fusion_model_state['base_weight'])
            if hasattr(fusion_model, 'temperature') and fusion_model.temperature is not None and \
                    best_fusion_model_state['temperature'] is not None:
                fusion_model.temperature.data.copy_(best_fusion_model_state['temperature'])

        print(f"Loaded best model (Epoch {best_epoch + 1})")
        print(f"Feature weight after load: {fusion_model.get_weight():.4f}")

        # Verify the weight was restored
        test_weight = fusion_model.get_weight()
        print(f"Weight consistency check: best validation weight={best_val_weight:.4f}, loaded weight={test_weight:.4f}")
        if abs(test_weight - best_val_weight) > 0.001:
            print(f"Warning: weight mismatch! Difference: {abs(test_weight - best_val_weight):.4f}")
    else:
        print("Warning: no best model state found, using the current model")

    train_printer.print_header("Model Evaluation", 80)

    print("\nEvaluating on the validation set with the best weight...")
    val_metrics, val_preds, val_targets, val_feature_preds, val_text_preds = evaluate(
        fusion_model, feature_val_loader, text_val_loader,
        criterion, device, "Validation", fixed_weight=best_val_weight
    )

    print(f"\nValidation results with best weight {best_val_weight:.4f}:")
    print(f"  Loss: {val_metrics['loss']:.6f}")
    print(f"  MAE: {val_metrics['mae']:.6f}")
    print(f"  RMSE: {val_metrics['rmse']:.6f}")
    print(f"  R2: {val_metrics['r2']:.6f}")
    print(f"  Weight actually used: {val_metrics.get('weight', best_val_weight):.4f}")

    print("\nEvaluating on the test set with the best weight...")
    test_metrics, test_preds, test_targets, test_feature_preds, test_text_preds = evaluate(
        fusion_model, feature_test_loader, text_test_loader,
        criterion, device, "Test", fixed_weight=best_val_weight
    )

    train_printer.print_header("Final Evaluation Results (Best Validation Weight)", 80)

    print(f"\n{'Metric':<12} {'Validation':<20} {'Test':<20}")
    print("-" * 50)
    print(f"{'Loss':<12} {val_metrics['loss']:<20.6f} {test_metrics['loss']:<20.6f}")
    print(f"{'MAE':<12} {val_metrics['mae']:<20.6f} {test_metrics['mae']:<20.6f}")
    print(f"{'RMSE':<12} {val_metrics['rmse']:<20.6f} {test_metrics['rmse']:<20.6f}")
    print(f"{'R2':<12} {val_metrics['r2']:<20.6f} {test_metrics['r2']:<20.6f}")
    print(f"{'Weight':<12} {best_val_weight:<20.4f} {best_val_weight:<20.4f}")
    print()

    final_state = {
        'epoch': best_epoch + 1,
        'feature_model_state_dict': feature_gnn_model.state_dict(),
        'text_model_state_dict': text_gnn_model.state_dict(),
        'fusion_model_state_dict': fusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'train_weights': train_weights,
        'val_weights': val_weights,
        'final_weight': best_val_weight,
        'best_val_weight': best_val_weight,
        'args': vars(args)
    }

    torch.save(final_state, os.path.join(save_dir, 'final_fusion_model.pth'))
    print(f"Model saved to: {os.path.join(save_dir, 'final_fusion_model.pth')}")

    total_params = sum(p.numel() for p in fusion_model.parameters())
    trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    feature_params = sum(p.numel() for p in feature_gnn_model.parameters())
    text_params = sum(p.numel() for p in text_gnn_model.parameters())
    fusion_params = sum(p.numel() for p in fusion_model.parameters()) - feature_params - text_params

    results = {
        'best_epoch': best_epoch + 1,
        'best_val_loss': float(best_val_loss),
        'best_val_weight': float(best_val_weight),
        'final_weight': float(best_val_weight),
        'test_loss': float(test_metrics['loss']),
        'test_mae': float(test_metrics['mae']),
        'test_rmse': float(test_metrics['rmse']),
        'test_r2': float(test_metrics['r2']),
        'test_weight': float(best_val_weight),
        'val_loss': float(val_metrics['loss']),
        'val_mae': float(val_metrics['mae']),
        'val_rmse': float(val_metrics['rmse']),
        'val_r2': float(val_metrics['r2']),
        'val_weight': float(best_val_weight),
        'train_losses': [float(loss) for loss in train_losses],
        'val_losses': [float(loss) for loss in val_losses],
        'val_maes': [float(mae) for mae in val_maes],
        'val_rmses': [float(rmse) for rmse in val_rmses],
        'train_weights': [float(w) for w in train_weights],
        'val_weights': [float(w) for w in val_weights],
        'training_time': float(training_time),
        'optimal_weight': float(best_val_weight),
        'feature_dataset_stats': feature_dataset_stats,
        'text_dataset_stats': text_dataset_stats,
        'model_params': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'feature_params': feature_params,
            'text_params': text_params,
            'fusion_params': fusion_params
        },
        'args': vars(args)
    }

    results_file = save_results_json(results, save_dir)
    txt_file = train_printer.save_results(results, save_dir, args, feature_gnn_model, text_gnn_model, fusion_model)
    print(f"\nResults files saved:")
    print(f"  JSON: {results_file}")
    print(f"  Text: {txt_file}")

    train_printer.print_header("Training Finished", 80)
    print(f"\nAll results saved to: {save_dir}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


def init_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fusion model training script for thermal conductivity prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument('--feature_attributed_gnn_data_dir', type=str, default='models/feature_attributed_gnn/graphs',
                        help='feature_attributed_gnn data directory')
    parser.add_argument('--text_attributed_gnn_data_dir', type=str, default='models/text_attributed_gnn/graphs',
                        help='text_attributed_gnn data directory')
    add_common_args(parser, num_layers=4, patience=200)

    # Model
    parser.add_argument('--type_embedding_dim', type=int, default=8,
                        help='type embedding dimension')

    # Fusion
    parser.add_argument('--learnable_weight', action='store_true', default=True,
                        help='whether the fusion weight is learnable')
    parser.add_argument('--initial_weight', type=float, default=0.661,  # 1 = feature, 0 = text
                        help='initial weight (Feature GNN weight)')
    parser.add_argument('--aux_loss_weight', type=float, default=0.5,
                        help='symmetric auxiliary loss weight (used only when the per-branch weights are None; defaults use aux_feature_weight=0.1 / aux_text_weight=0.2)')
    parser.add_argument('--aux_feature_weight', type=float, default=0.1,
                        help='FAGNN branch auxiliary loss weight (weak regularization keeps the complementary bias; FAGNN slightly overestimates)')
    parser.add_argument('--aux_text_weight', type=float, default=0.2,
                        help='TAGNN branch auxiliary loss weight (weak regularization keeps the complementary bias; TAGNN slightly underestimates)')
    parser.add_argument('--weight_learning_strategy', type=str, default='adaptive',
                        choices=['fixed', 'adaptive', 'performance_aware', 'scheduled'],
                        help='weight learning strategy: fixed, adaptive, performance_aware, scheduled')
    parser.add_argument('--weight_growth_rate', type=float, default=0.01,
                        help='weight growth rate (for the scheduled strategy)')

    # GNN convolution types
    parser.add_argument('--feature_gnn_type', type=str, default='gine',
                        choices=['gine', 'gcn', 'gat', 'edge', 'sage', 'transformer', 'resgated'])
    parser.add_argument('--text_gnn_type', type=str, default='gine',
                        choices=['gine', 'gcn', 'gat', 'edge', 'sage', 'transformer', 'resgated'])

    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    run_training(args, main, data_dirs=[
        ('Feature GNN data', args.feature_attributed_gnn_data_dir),
        ('Text GNN data', args.text_attributed_gnn_data_dir),
    ])
