import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TrainingPlotter:
    """Create plots for BERT fine-tuning metrics compatible with W&B logging."""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """Initialize plotter with consistent styling."""
        plt.style.use(style) if style else None
        sns.set_palette("husl")
    
    def plot_loss_curves(self, train_losses, val_losses, epochs=None):
        """Plot training and validation loss curves.
        
        Args:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            epochs: Optional list of epoch numbers
            
        Returns:
            matplotlib.figure.Figure for W&B logging
        """
        if epochs is None:
            epochs = range(1, len(train_losses) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_losses, 'o-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_losses, 's-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_metrics(self, metrics_dict, metric_name='Accuracy'):
        """Plot training and validation metrics over epochs.
        
        Args:
            metrics_dict: Dict with 'train' and 'val' keys containing metric lists
            metric_name: Name of the metric (e.g., 'Accuracy', 'F1')
            
        Returns:
            matplotlib.figure.Figure for W&B logging
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(metrics_dict['train']) + 1)
        
        ax.plot(epochs, metrics_dict['train'], 'o-', label=f'Train {metric_name}', linewidth=2)
        ax.plot(epochs, metrics_dict['val'], 's-', label=f'Val {metric_name}', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} Over Training', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, cm, class_names=None):
        """Plot confusion matrix.
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names for labels
            
        Returns:
            matplotlib.figure.Figure for W&B logging
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_learning_rate(self, learning_rates, steps=None):
        """Plot learning rate schedule.
        
        Args:
            learning_rates: List of learning rates
            steps: Optional list of step numbers
            
        Returns:
            matplotlib.figure.Figure for W&B logging
        """
        if steps is None:
            steps = range(len(learning_rates))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, learning_rates, linewidth=2, color='#e74c3c')
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    
    def plot_trainable_params(self, params_dict):
        """Visualize trainable parameters comparison (useful for LoRA).
        
        Args:
            params_dict: Dict with model names and trainable parameter counts
                        e.g., {'BERT': 110M, 'BERT+LoRA': 2.5M}
            
        Returns:
            matplotlib.figure.Figure for W&B logging
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        models = list(params_dict.keys())
        params = list(params_dict.values())
        
        bars = ax.barh(models, params, color=['#3498db', '#2ecc71'])
        ax.set_xlabel('Trainable Parameters (M)', fontsize=12)
        ax.set_title('Model Complexity Comparison', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.1f}M', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def close_figure(fig):
        """Close figure to free memory after logging to W&B."""
        plt.close(fig)
