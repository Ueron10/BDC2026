
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
from tqdm import tqdm


def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro'),
        'recall_macro': recall_score(all_labels, all_preds, average='macro'),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_per_class': f1_score(all_labels, all_preds, average=None),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    metrics['precision_per_class'] = precision_score(all_labels, all_preds, average=None)
    metrics['recall_per_class'] = recall_score(all_labels, all_preds, average=None)
    
    return metrics


def print_evaluation_report(metrics, class_names=None):
    if class_names is None:
        class_names = ['Recyclable', 'Electronic', 'Organic']
    
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Macro F1:  {metrics['f1_macro']:.4f}")
    print(f"  Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"  Macro Recall:    {metrics['recall_macro']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"\n  {class_name} (Class {i}):")
        print(f"    Precision: {metrics['precision_per_class'][i]:.4f}")
        print(f"    Recall:    {metrics['recall_per_class'][i]:.4f}")
        print(f"    F1-Score:  {metrics['f1_per_class'][i]:.4f}")
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        metrics['labels'], 
        metrics['predictions'],
        target_names=class_names,
        digits=4
    ))


def plot_confusion_matrix(metrics, class_names=None, save_path=None):
    if class_names is None:
        class_names = ['Recyclable', 'Electronic', 'Organic']
    
    cm = confusion_matrix(metrics['labels'], metrics['predictions'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2)
    axes[1, 0].set_title('Validation Macro F1-Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('F1-Score', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    if 'lr' in history:
        axes[1, 1].plot(epochs, history['lr'], 'm-', linewidth=2)
        axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Learning rate history not available', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
