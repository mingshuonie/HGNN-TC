import os
from datetime import datetime


class PrintTrainProcess:
    """Console progress printing and results-file saving for training runs."""

    def __init__(self, device):
        self.device = device

    def print_epoch_summary(self, epoch, total_epochs, train_loss, train_weight, val_metrics, lr, epoch_time,
                            is_best=False):
        """Print a one-line summary of the current epoch."""
        best_marker = "*" if is_best else " "

        summary = (
            f"Epoch {epoch + 1:3d}/{total_epochs:<3d} {best_marker} | "
            f"Train Loss: {train_loss:>10.4f} | "
            f"Val Loss: {val_metrics['loss']:>10.4f} | "
            f"Val MAE: {val_metrics['mae']:>8.4f} | "
            f"Val R2: {val_metrics['r2']:>8.4f} | "
            f"Wgt: {val_metrics.get('weight', train_weight):>5.3f} | "
            f"LR: {lr:.6f} | "
            f"Time: {epoch_time:>5.1f}s"
        )
        if is_best:
            print(f"\033[1;31;40m{summary}\033[0m")
        else:
            print(summary)

    def print_header(self, title, width=80):
        """Print a banner-style title header."""
        print("\n" + "=" * width)
        print(title.center(width))
        print("=" * width)

    def print_section(self, title, width=60):
        """Print a dashed section title."""
        print("\n" + "-" * width)
        print(title)
        print("-" * width)

    def save_results(self, results, save_dir, args, feature_gnn_model, text_gnn_model, fusion_model):
        """Save the human-readable results summary to results.txt.

        The machine-readable results.json is written by
        ``utils.training.save_results_json``.
        """
        # Final fusion weight
        final_weight = fusion_model.get_weight() if hasattr(fusion_model, 'get_weight') else 0.5

        # Save human-readable text results
        txt_file = os.path.join(save_dir, 'results.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Fusion Model Training Results for Thermal Conductivity Prediction\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Training time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Duration: {results['training_time']:.2f}s ({results['training_time'] / 60:.1f}min)\n\n")

            f.write("Model fusion information:\n")
            f.write(f"  Feature GNN weight: {final_weight:.4f}\n")
            f.write(f"  Text GNN weight: {1 - final_weight:.4f}\n")
            f.write(f"  Weight learnable: {args.learnable_weight}\n")
            f.write(f"  Initial weight: {args.initial_weight}\n\n")

            f.write("Best model information:\n")
            f.write(f"  Best epoch: {results['best_epoch']}\n")
            f.write(f"  Best validation loss: {results['best_val_loss']:.6f}\n")
            f.write(f"  Best validation weight: {results.get('best_val_weight', 0.5):.4f}\n\n")

            f.write("Validation performance:\n")
            f.write(f"  Validation loss: {results['val_loss']:.6f}\n")
            f.write(f"  Validation MAE: {results['val_mae']:.6f}\n")
            f.write(f"  Validation RMSE: {results['val_rmse']:.6f}\n")
            f.write(f"  Validation R2: {results['val_r2']:.6f}\n\n")

            f.write("Test performance:\n")
            f.write(f"  Test loss: {results['test_loss']:.6f}\n")
            f.write(f"  Test MAE: {results['test_mae']:.6f}\n")
            f.write(f"  Test RMSE: {results['test_rmse']:.6f}\n")
            f.write(f"  Test R2: {results['test_r2']:.6f}\n")
            f.write(f"  Theoretical optimal weight: {results.get('optimal_weight', 0.5):.4f}\n\n")

            f.write("Model parameter statistics:\n")
            f.write(f"  Feature GNN parameters: {results['model_params']['feature_params']:,}\n")
            f.write(f"  Text GNN parameters: {results['model_params']['text_params']:,}\n")
            f.write(f"  Fusion model parameters: {results['model_params']['fusion_params']:,}\n")
            f.write(f"  Total parameters: {results['model_params']['total_params']:,}\n")
            f.write(f"  Trainable parameters: {results['model_params']['trainable_params']:,}\n\n")

            f.write("Dataset statistics:\n")
            f.write("Feature GNN dataset:\n")
            for key, value in results['feature_dataset_stats'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\nText GNN dataset:\n")
            for key, value in results['text_dataset_stats'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("Training arguments:\n")
            for key, value in results['args'].items():
                f.write(f"  {key}: {value}\n")

        return txt_file
