"""Shared training utilities for the three training entry points.

The standalone branch scripts (feature_model_train.py / text_model_train.py)
share an identical train/evaluate loop and result-saving skeleton; both are
hosted here. The fusion script (final_train.py) keeps its own fusion-specific
train/evaluate functions but reuses the argument parsing and entry-point
helpers defined below.
"""
import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from utils.print_train_process import PrintTrainProcess
from utils.set_seed import set_seed


def train_epoch(model, loader, optimizer, criterion, device,
                epoch=None, total_epochs=None, clip_grad=None):
    """Train a single-branch model for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader,
                desc=f"Epoch {epoch + 1:3d}/{total_epochs:3d} [Train]",
                leave=False, dynamic_ncols=True,
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        output = model(batch)

        # Compute the loss
        loss = criterion(output, batch.y)

        # Backward pass
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        # Accumulate statistics
        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Update the progress bar description
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

    pbar.close()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return avg_loss


def evaluate(model, loader, criterion, device,
             phase="Validation", epoch=None, total_epochs=None):
    """Evaluate a single-branch model; returns (metrics, predictions, targets)."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    # Progress bar description
    if epoch is not None:
        desc = f"Epoch {epoch + 1:3d}/{total_epochs:3d} [{phase}]"
    else:
        desc = f"[{phase}]"

    pbar = tqdm(loader, desc=desc, leave=False,
                dynamic_ncols=True, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    with torch.no_grad():
        for batch in pbar:
            # Move the data to the device
            batch = batch.to(device)

            # Forward pass
            output = model(batch)

            # Compute the loss
            loss = criterion(output, batch.y)

            # Compute evaluation metrics
            mae = torch.abs(output - batch.y).sum().item()
            mse = torch.square(output - batch.y).sum().item()

            # Accumulate statistics
            batch_size = batch.num_graphs
            total_loss += loss.item() * batch_size
            total_mae += mae
            total_mse += mse
            total_samples += batch_size

            # Save predictions
            all_preds.extend(output.cpu().numpy().flatten())
            all_targets.extend(batch.y.cpu().numpy().flatten())

    pbar.close()

    # Compute evaluation metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_mae = total_mae / total_samples if total_samples > 0 else 0.0
    rmse = np.sqrt(total_mse / total_samples) if total_samples > 0 else 0.0

    # Compute R2
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
        'r2': r2
    }

    return metrics, np.array(all_preds), np.array(all_targets)


def add_common_args(parser, num_layers=6, patience=100):
    """Add the argument block shared by all three training scripts.

    The two branch scripts and the fusion script share these arguments with
    identical names and types; only the ``num_layers`` and ``patience``
    defaults differ between them.
    """
    # Data arguments
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Results save directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')

    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=num_layers,
                        help='Number of model layers')
    parser.add_argument('--dropout', type=float, default=0.01,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=patience,
                        help='Early stopping patience')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Gradient clipping threshold')
    parser.add_argument('--seed', type=int, default=530, help='Random seed')


def save_results_json(results, save_dir):
    """Write the results dict to results.json."""
    results_file = os.path.join(save_dir, 'results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results_file


def save_branch_results_txt(save_dir, *, title, device, training_time, best_epoch,
                            best_val_loss, test_metrics, total_params, trainable_params):
    """Write the human-readable results summary of a standalone branch run."""
    txt_file = os.path.join(save_dir, 'results.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Training time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Duration: {training_time:.2f}s\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation loss: {best_val_loss:.6f}\n")
        f.write(f"Final test loss: {test_metrics['loss']:.6f}\n")
        f.write(f"Final test MAE: {test_metrics['mae']:.6f}\n")
        f.write(f"Final test RMSE: {test_metrics['rmse']:.6f}\n")
        f.write(f"Final test R2: {test_metrics['r2']:.6f}\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
    return txt_file


def run_branch_training(args, *, model_label, save_prefix, model_filename, dataset_stats_key,
                        gnn_type, load_data, create_model, device):
    """Run the full training pipeline for a standalone GNN branch.

    This is the shared skeleton previously duplicated in the feature/text
    training scripts; ``load_data`` and ``create_model`` supply the
    branch-specific dataset loaders and model factory. ``load_data`` must
    return ``(train_loader, val_loader, test_loader, dataset_stats)``.
    """
    # Create the results save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{save_prefix}{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    train_printer = PrintTrainProcess(device)

    # Print the training header
    train_printer.print_header(f"{model_label} Training for Thermal Conductivity Prediction", 80)
    print(f"Device: {device}")
    print(f"Results directory: {save_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Create data loaders
    train_printer.print_section("Loading Dataset")
    print(f"Loading {model_label} data...")
    train_loader, val_loader, test_loader, dataset_stats = load_data()

    print("Dataset loaded!")
    print(f"{model_label} dataset statistics: {dataset_stats}")

    # Create the model
    train_printer.print_section(f"Creating {model_label} Model")

    model = create_model(dataset_stats)

    print(f"{model_label} model created!")
    print(f"GNN type: {gnn_type}")

    # Loss function and optimizer
    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Training records
    train_losses = []
    val_losses = []
    val_maes = []
    val_rmses = []

    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    # Start training
    train_printer.print_header("Start Training", 80)

    # Print the training table header
    print(f"\n {'Epoch':<6} {'Best':<5} {'Train Loss':<12} {'Val Loss':<12} "
          f"{'Val MAE':<10} {'Val R2':<8} {'LR':<12} {'Time':<8}")
    print("-" * 80)

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 epoch, args.epochs, args.clip_grad)
        train_losses.append(train_loss)

        # Validate
        val_metrics, val_preds, val_targets = evaluate(
            model, val_loader, criterion, device, "Validation", epoch, args.epochs
        )
        val_losses.append(val_metrics['loss'])
        val_maes.append(val_metrics['mae'])
        val_rmses.append(val_metrics['rmse'])

        # Update the learning rate
        if scheduler is not None:
            scheduler.step(val_metrics['loss'])

        # Check whether this is the best model
        is_best = val_metrics['loss'] < best_val_loss

        if is_best:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch

            # Save the best model state
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics
            }

            # Save the best model
            torch.save(best_model_state, os.path.join(save_dir, f'best_{model_filename}_model.pth'))

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']

        # Clear the current line, then print
        print("\033[K", end='\r')

        # Print detailed status every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            train_printer.print_section(f"Epoch {epoch + 1} Detailed Status", 60)
            print(f"Train loss: {train_loss:.6f}")
            print(f"Validation loss: {val_metrics['loss']:.6f}")
            print(f"Validation MAE: {val_metrics['mae']:.6f}")
            print(f"Validation RMSE: {val_metrics['rmse']:.6f}")
            print(f"Validation R2: {val_metrics['r2']:.6f}")
            print(f"Learning rate: {lr:.6f}")
            print(f"Time: {epoch_time:.2f}s")

        # Early stopping check
        if epoch - best_epoch > args.patience:
            print(f"\n{'!' * 20} Early Stopping {'!' * 20}")
            print(f"Validation loss has not improved for {args.patience} epochs")
            print(f"Best epoch: {best_epoch + 1}, best validation loss: {best_val_loss:.6f}")
            break

    # Training finished
    training_time = time.time() - start_time
    train_printer.print_header("Training Complete", 80)
    print(f"Training complete! Total time: {training_time:.2f}s ({training_time / 60:.1f}min)")
    print(f"Best model at epoch {best_epoch + 1}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print()

    # Load the best model state
    train_printer.print_section("Loading Best Model State")
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        print(f"Loaded best model (Epoch {best_epoch + 1})")
    else:
        print("Warning: no best model state found, using the current model")

    # Model evaluation
    train_printer.print_header("Model Evaluation", 80)

    print("\nEvaluating on the validation set...")
    val_metrics, val_preds, val_targets = evaluate(
        model, val_loader, criterion, device, "Validation"
    )

    print(f"\nValidation set results:")
    print(f"  Loss: {val_metrics['loss']:.6f}")
    print(f"  MAE: {val_metrics['mae']:.6f}")
    print(f"  RMSE: {val_metrics['rmse']:.6f}")
    print(f"  R2: {val_metrics['r2']:.6f}")

    print("\nEvaluating on the test set...")
    test_metrics, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, "Test"
    )

    # Print the final results
    train_printer.print_header("Final Evaluation Results", 80)

    print(f"\n{'Metric':<12} {'Validation':<20} {'Test':<20}")
    print("-" * 50)
    print(f"{'Loss':<12} {val_metrics['loss']:<20.6f} {test_metrics['loss']:<20.6f}")
    print(f"{'MAE':<12} {val_metrics['mae']:<20.6f} {test_metrics['mae']:<20.6f}")
    print(f"{'RMSE':<12} {val_metrics['rmse']:<20.6f} {test_metrics['rmse']:<20.6f}")
    print(f"{'R2':<12} {val_metrics['r2']:<20.6f} {test_metrics['r2']:<20.6f}")
    print()

    # Save the final model
    final_model_path = os.path.join(save_dir, f'final_{model_filename}_model.pth')
    final_state = {
        'epoch': best_epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'args': vars(args)
    }

    torch.save(final_state, final_model_path)
    print(f"Model saved to: {final_model_path}")

    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Save results
    results = {
        'best_epoch': best_epoch + 1,
        'best_val_loss': float(best_val_loss),
        'test_loss': float(test_metrics['loss']),
        'test_mae': float(test_metrics['mae']),
        'test_rmse': float(test_metrics['rmse']),
        'test_r2': float(test_metrics['r2']),
        'val_loss': float(val_metrics['loss']),
        'val_mae': float(val_metrics['mae']),
        'val_rmse': float(val_metrics['rmse']),
        'val_r2': float(val_metrics['r2']),
        'train_losses': [float(loss) for loss in train_losses],
        'val_losses': [float(loss) for loss in val_losses],
        'val_maes': [float(mae) for mae in val_maes],
        'val_rmses': [float(rmse) for rmse in val_rmses],
        'training_time': float(training_time),
        dataset_stats_key: dataset_stats,
        'model_params': {
            'total_params': total_params,
            'trainable_params': trainable_params
        },
        'args': vars(args)
    }

    # Save results to files
    results_file = save_results_json(results, save_dir)
    txt_file = save_branch_results_txt(save_dir, title=f"{model_label} Model Training Results",
                                       device=device, training_time=training_time,
                                       best_epoch=best_epoch + 1, best_val_loss=best_val_loss,
                                       test_metrics=test_metrics, total_params=total_params,
                                       trainable_params=trainable_params)

    print(f"\nResults files saved:")
    print(f"  JSON format: {results_file}")
    print(f"  Text format: {txt_file}")

    train_printer.print_header("Training Finished", 80)
    print(f"\nAll results saved to: {save_dir}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


def run_training(args, main, data_dirs=None):
    """Validate arguments, seed the RNGs, and run a training entry point.

    ``data_dirs`` is a list of (label, path) tuples checked for existence
    before training starts.
    """
    for label, path in (data_dirs or []):
        if not os.path.exists(path):
            print(f"Error: {label} data directory does not exist: {path}")
            sys.exit(1)

    if args.train_ratio + args.val_ratio >= 1.0:
        print("Error: train_ratio + val_ratio must be less than 1.0")
        sys.exit(1)

    # Create the results save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Set random seeds
    set_seed(args.seed)

    # Run training
    try:
        results = main(args)
        if results is not None:
            print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    except Exception as e:
        print(f"\nError during training: {e}")
        traceback.print_exc()
