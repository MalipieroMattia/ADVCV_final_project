import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix
import wandb
from utils.logger import WandbLogger
from transformers import AutoTokenizer
import time
import matplotlib.pyplot as plt

from utils.misclassifications import EdgeCaseAnalyzer


class Trainer:
    """Handles training, validation, and evaluation of the model."""

    def __init__(
        self, model, train_loader, val_loader, test_loader, config, device="cpu"
    ):
        """
        Args:
            model: PriceDirectionClassifier instance
            train_loader, val_loader, test_loader: PyTorch DataLoaders
            config: Configuration dict
            device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        # Extract training params from config
        self.num_epochs = config["training"]["num_epochs"]
        self.learning_rate = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]
        self.max_grad_norm = config["training"]["max_grad_norm"]
        self.checkpoint_dir = Path(config["model"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

        # Tracking
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0

        # GPU metrics tracking
        self.gpu_metrics = {"peak_memory_gb": 0, "epoch_times": [], "throughput": []}

        # Initialize W&B logger
        run_name = config.get(
            "run_name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.logger = WandbLogger(config, run_name=run_name)

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # reset GPU memory stats at start of epoch for accurate measurement
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        epoch_start = time.time()

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )
            self.optimizer.step()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
            )

        # track epoch time and throughput
        epoch_time = time.time() - epoch_start
        self.gpu_metrics["epoch_times"].append(epoch_time)
        self.gpu_metrics["throughput"].append(total / epoch_time)

        # track GPU memory
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1e9
            self.gpu_metrics["peak_memory_gb"] = max(
                self.gpu_metrics["peak_memory_gb"], peak_mem
            )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{100 * correct / total:.2f}%",
                    }
                )

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _should_stop_early(self, val_losses, patience):
        """Check if early stopping criteria is met."""
        if len(val_losses) < patience + 1:
            return False

        # Check if validation loss hasn't improved in 'patience' epochs
        recent_losses = val_losses[-patience:]
        best_recent = min(recent_losses)

        return best_recent >= min(val_losses[:-patience])

    def train(self):
        """Main training loop."""
        print(f"\n{'=' * 60}")
        print(f"Starting Training: {self.num_epochs} epochs")
        print(f"{'=' * 60}")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Learning rate: {self.learning_rate}")

        # Show model info
        use_lora = self.config["model"].get("use_lora", False)
        if use_lora:
            lora_r = self.config["model"]["lora"].get("r", "N/A")
            print(f"Method: LoRA (r={lora_r})")
        else:
            freeze_strategy = self.config["model"].get("freeze_strategy", "unknown")
            num_frozen = self.config["model"].get("num_frozen_layers", "N/A")
            print(
                f"Method: {freeze_strategy} fine-tuning ({num_frozen} layers unfrozen)"
            )
        print(f"{'=' * 60}\n")

        # Log model architecture info once
        # Log model architecture info once
        from model.model_loader import ModelLoader

        model_loader = ModelLoader(self.config)
        total_params, trainable_params = model_loader.count_parameters(self.model)
        self.logger.log_trainable_params(
            {"Total": total_params, "Trainable": trainable_params}
        )

        # Track metrics for plotting
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        learning_rates = []

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 60)

            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            learning_rates.append(current_lr)

            # Log to W&B
            self.logger.log_metrics(
                {
                    "epoch": epoch + 1,
                    "loss/train": train_loss,  # Use 'loss/' prefix
                    "loss/val": val_loss,  # Use 'loss/' prefix
                    "accuracy/train": train_acc,  # Use 'accuracy/' prefix
                    "accuracy/val": val_acc,  # Use 'accuracy/' prefix
                    "learning_rate": current_lr,
                    "gpu/memory_gb": self.gpu_metrics["peak_memory_gb"],
                    "gpu/epoch_time_sec": self.gpu_metrics["epoch_times"][-1],
                    "gpu/throughput_samples_per_sec": self.gpu_metrics["throughput"][
                        -1
                    ],
                },
                step=epoch + 1,
            )

            # Print summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {100 * train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {100 * val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"  Epoch Time: {self.gpu_metrics['epoch_times'][-1]:.1f}s")
            print(f"  Peak Memory: {self.gpu_metrics['peak_memory_gb']:.2f} GB")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True, metric="loss")
                print(f"New best validation loss: {val_loss:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True, metric="acc")
                print(f"New best validation accuracy: {100 * val_acc:.2f}%")

            # Early stopping
            patience = self.config["training"].get("early_stopping_patience", 5)
            if self._should_stop_early(val_losses, patience):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Log final plots to W&B
        self.logger.log_loss_curves(train_losses, val_losses)
        self.logger.log_metrics_plot(
            {"train": train_accs, "val": val_accs}, metric_name="Accuracy"
        )
        self.logger.log_learning_rate(learning_rates)

        # log GPU metrics summary
        avg_epoch_time = sum(self.gpu_metrics["epoch_times"]) / len(
            self.gpu_metrics["epoch_times"]
        )
        avg_throughput = sum(self.gpu_metrics["throughput"]) / len(
            self.gpu_metrics["throughput"]
        )
        total_time = sum(self.gpu_metrics["epoch_times"])

        self.logger.log_metrics(
            {
                "gpu/peak_memory_gb": self.gpu_metrics["peak_memory_gb"],
                "gpu/avg_epoch_time_sec": avg_epoch_time,
                "gpu/avg_throughput_samples_per_sec": avg_throughput,
                "gpu/total_training_time_sec": total_time,
            }
        )

        print(f"\n{'=' * 60}")
        print("Training Complete!")
        print(f"{'=' * 60}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val Accuracy: {100 * self.best_val_acc:.2f}%")
        print(f"\n--- Resource Usage ---")
        print(f"GPU Peak Memory: {self.gpu_metrics['peak_memory_gb']:.2f} GB")
        print(f"Avg Throughput: {avg_throughput:.1f} samples/sec")
        print(f"Total Training Time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"Avg Time per Epoch: {avg_epoch_time:.1f}s")
        print(f"{'=' * 60}\n")

    def test(self, save_misclassifications=True):
        """Evaluate on test set and optionally save misclassified examples."""
        print("\n" + "=" * 60)
        print("Testing on Test Set")
        print("=" * 60 + "\n")

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_input_ids = []  # store for error analysis
        all_preds_probs = []  # collect predicted-class probabilities
        all_true_probs = []  # collect true-class probabilities

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                # probabilities
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probs, dim=1)

                # grab predicted-class prob and true-class prob
                pred_probs = probs[torch.arange(predictions.size(0)), predictions]
                true_probs = probs[torch.arange(labels.size(0)), labels]

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_input_ids.extend(input_ids.cpu().numpy())  # save for decoding
                all_preds_probs.extend(pred_probs.cpu().numpy())
                all_true_probs.extend(true_probs.cpu().numpy())
                total_loss += loss.item()

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{100 * correct / total:.2f}%",
                    }
                )

        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total

        print(f"\nTest Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {100 * accuracy:.2f}%")

        # Calculate and print class metrics
        self._print_class_metrics(all_labels, all_predictions)

        # Log to W&B
        cm = confusion_matrix(all_labels, all_predictions)
        self.logger.log_metrics({"test/loss": avg_loss, "test/acc": accuracy})

        # get class names from config
        label_map = self.config["data"].get("label_map", {})
        if label_map:
            class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
        else:
            class_names = None
        self.logger.log_confusion_matrix(cm, class_names=class_names)

        # save misclassifications for error analysis
        if save_misclassifications:
            df_error = self._save_misclassifications(
                all_input_ids,
                all_labels,
                all_predictions,
                all_preds_probs,
                all_true_probs,
            )

        model_name = self.config["model"]["name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        edgeCaseAnalyzer = EdgeCaseAnalyzer(df_error, self.model, tokenizer)
        analysis = edgeCaseAnalyzer.run_full_analysis()

        top_edge_cases = analysis.get("top_edge_cases")
        tiny_clusters = analysis.get("tiny_clusters")
        cluster_keywords = analysis.get("cluster_keywords")
        coords = analysis.get("coords")
        labels = analysis.get("labels")
        fig = analysis.get("plot")

        # log to wandb
        if fig is not None:
            wandb.log({"edge_case_plot": wandb.Image(fig)})
            plt.close(fig)

        if isinstance(top_edge_cases, pd.DataFrame):
            wandb.log(
                {
                    "top_edge_cases_table": wandb.Table(
                        dataframe=top_edge_cases.reset_index(drop=True)
                    )
                }
            )

        if (
            coords is not None
            and labels is not None
            and isinstance(df_error, pd.DataFrame)
            and "text" in df_error.columns
        ):
            rows = []
            for i, (xy, lbl, txt) in enumerate(zip(coords, labels, df_error["text"])):
                row = {
                    "text": txt,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "label": int(lbl),
                }
                if "edge_score" in df_error.columns:
                    row["edge_score"] = float(
                        df_error.iloc[i].get("edge_score", float("nan"))
                    )
                if "predicted_prob" in df_error.columns:
                    row["predicted_prob"] = float(
                        df_error.iloc[i].get("predicted_prob", float("nan"))
                    )
                if "true_prob" in df_error.columns:
                    row["true_prob"] = float(
                        df_error.iloc[i].get("true_prob", float("nan"))
                    )
                rows.append(row)
            if rows:
                cols = list(rows[0].keys())
                table_rows = [[r[c] for c in cols] for r in rows]
                wandb_table = wandb.Table(columns=cols, data=table_rows)
                wandb.log({"edge_cases_coords_table": wandb_table})

        # 4) cluster keywords (dict) -> wandb.Table
        if isinstance(cluster_keywords, dict):
            kw_rows = [[int(cid), kws] for cid, kws in cluster_keywords.items()]
            kw_table = wandb.Table(columns=["cluster_id", "keywords"], data=kw_rows)
            wandb.log({"cluster_keywords": kw_table})
        return avg_loss, accuracy, all_predictions, all_labels

    def _save_misclassifications(
        self, input_ids, true_labels, predictions, pred_probs=None, true_probs=None
    ):
        """
        Save misclassified examples to CSV and log to Wandb.

        Args:
            input_ids: List of tokenized inputs
            true_labels: List of ground truth labels
            predictions: List of predicted labels
            pred_probs: (optional) List of predicted-class probabilities
            true_probs: (optional) List of true-class probabilities
        """

        # Find misclassified indices
        misclass_indices = [
            i
            for i, (true, pred) in enumerate(zip(true_labels, predictions))
            if true != pred
        ]

        if len(misclass_indices) == 0:
            print("\n No misclassifications found.")
            return

        print(
            f"\n Found {len(misclass_indices)} misclassifications, ({100 * len(misclass_indices) / len(true_labels):.1f}% of test set)"
        )

        # get tokenizer to decode text
        model_name = self.config["model"]["name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # label mapping (update this based on your task)
        label_names = self.config["data"].get("label_map", {})
        if label_names:
            # reverse the mapping
            label_names = {v: k for k, v in label_names.items()}
        else:
            label_names = {0: "class_0", 1: "class_1", 2: "class_2"}

        # collect misclassified examples
        errors = []
        for idx in misclass_indices:
            # Decode the tokenized text
            text = tokenizer.decode(input_ids[idx], skip_special_tokens=True)

            errors.append(
                {
                    "text": text,
                    "true_label": label_names.get(
                        true_labels[idx], str(true_labels[idx])
                    ),
                    "predicted_label": label_names.get(
                        predictions[idx], str(predictions[idx])
                    ),
                    "true_label_id": true_labels[idx],
                    "predicted_label_id": predictions[idx],
                    "predicted_prob": float(pred_probs[idx])
                    if pred_probs is not None
                    else None,
                    "true_prob": float(true_probs[idx])
                    if true_probs is not None
                    else None,
                }
            )

        # create DataFrame
        df_errors = pd.DataFrame(errors)

        # save locally
        run_name = self.config["wandb"]["run_name"]
        error_file = self.checkpoint_dir / f"misclassifications_{run_name}.csv"
        df_errors.to_csv(error_file, index=False)
        print(f"Saved misclassifications to: {error_file}")

        # log to Wandb as artifact
        if self.logger.run:
            import wandb

            artifact = wandb.Artifact(
                name=f"misclassifications_{run_name}",
                type="error_analysis",
                description=f"Misclassified examples from {run_name}",
            )
            artifact.add_file(str(error_file))
            wandb.log_artifact(artifact)
            print(f"Uploaded to Wandb as artifact")

        # also log summary stats
        self.logger.log_metrics(
            {
                "test/num_misclassifications": len(misclass_indices),
                "test/misclassification_rate": len(misclass_indices) / len(true_labels),
            }
        )

        print(f"Error analysis complete")
        return df_errors

    def _print_class_metrics(self, labels, predictions):
        """Print per-class precision, recall, F1."""
        from sklearn.metrics import classification_report

        # get class names from config or use defaults
        label_map = self.config["data"].get("label_map", {})
        if label_map:
            # sort by value to get correct order
            class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
        else:
            class_names = ["class_0", "class_1", "class_2"]

        report = classification_report(labels, predictions, target_names=class_names)
        print("\nClassification Report:")
        print(report)

    def save_checkpoint(self, epoch, is_best=False, metric="loss"):
        """Save model checkpoint."""
        # Skip if local saving is disabled
        save_local = self.config["model"]["save_local"]
        if not save_local:
            print(f"  (Local checkpoint saving disabled in config)")
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "config": self.config,
        }

        if is_best:
            filename = f"best_model_{metric}.pt"
            filepath = self.checkpoint_dir / filename
            torch.save(checkpoint, filepath)
            print(f"  Saved checkpoint: {filepath}")

            # Only log to W&B if enabled in config
            if self.config["wandb"].get("log_model", False):
                self.logger.log_model(str(filepath), name=f"best_model_{metric}")
                print(f"Uploaded to W&B as artifact")
