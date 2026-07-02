
import os
import sys
import argparse
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from utils import set_seed, get_device, setup_paths, print_paths, count_parameters
from preprocessing import (
    load_train_data, load_test_data, split_train_val,
    create_dataloaders, create_test_dataloader, get_tta_transforms,
    save_preprocessing_data, load_preprocessing_data
)
from model import WasteClassifier, Trainer, load_model, save_training_history, load_training_history
from evaluate import evaluate_model, print_evaluation_report, plot_confusion_matrix, plot_training_history, save_evaluation_metrics, load_evaluation_metrics
from predict import predict, predict_with_tta, generate_submission, analyze_test_predictions


def main(args):
    set_seed(args.seed)
    
    device = get_device()
    print(f"Using device: {device}")
    
    paths = setup_paths(
        args.data_dir,
        args.model_dir,
        args.docs_dir,
        args.template_path,
        args.output_path
    )
    
    print_paths(paths)
    
    class_names = Config.CLASS_NAMES
    
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    
    if args.load_preprocessing:
        print("Loading preprocessing data from file...")
        train_paths, train_labels, val_paths, val_labels, test_paths, class_to_idx = load_preprocessing_data(paths['output_dir'])
    else:
        train_paths, train_labels, class_to_idx = load_train_data(paths['data_dir'])
        
        test_paths = load_test_data(paths['data_dir'])
        
        train_paths, val_paths, train_labels, val_labels = split_train_val(
            train_paths, train_labels, 
            val_split=args.val_split, 
            random_state=args.seed
        )
        
        save_preprocessing_data(train_paths, train_labels, val_paths, val_labels, 
                                test_paths, class_to_idx, paths['output_dir'])
    
    print("\n" + "="*60)
    print("STEP 2: CREATING DATALOADERS")
    print("="*60)
    
    train_loader, val_loader = create_dataloaders(
        train_paths, train_labels,
        val_paths, val_labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    test_loader = create_test_dataloader(
        test_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print("\n" + "="*60)
    print("STEP 3: TRAINING MODEL")
    print("="*60)
    model = WasteClassifier(
        num_classes=Config.NUM_CLASSES,
        pretrained=args.pretrained
    )
    
    print(f"Model architecture: efficientnet_b0")
    print(f"Number of parameters: {count_parameters(model):,}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=Config.LABEL_SMOOTHING,
        use_mixup=Config.USE_MIXUP,
        mixup_alpha=Config.MIXUP_ALPHA,
        warmup_epochs=Config.WARMUP_EPOCHS,
        min_lr=Config.MIN_LR,
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
    )
    
    if args.load_training:
        print("Loading training history from file...")
        history = load_training_history(paths['output_dir'])
        plot_training_history(
            history,
            save_path=os.path.join(paths['docs_dir'], 'training_history.png')
        )
    else:
        history = trainer.train(
            num_epochs=args.epochs,
            save_dir=paths['model_dir']
        )
        
        save_training_history(history, paths['output_dir'])
        
        plot_training_history(
            history,
            save_path=os.path.join(paths['docs_dir'], 'training_history.png')
        )
    
    best_model_path = os.path.join(paths['model_dir'], 'best_model.pth')
    model, checkpoint = load_model(best_model_path, num_classes=Config.NUM_CLASSES, device=device)
    
    print("\n" + "="*60)
    print("STEP 4: EVALUATING MODEL")
    print("="*60)
    
    if args.load_evaluation:
        print("Loading evaluation metrics from file...")
        metrics = load_evaluation_metrics(paths['output_dir'])
        print_evaluation_report(metrics, class_names)
        
        plot_confusion_matrix(
            metrics,
            class_names=class_names,
            save_path=os.path.join(paths['docs_dir'], 'confusion_matrix.png')
        )
    else:
        metrics = evaluate_model(model, val_loader, device)
        
        save_evaluation_metrics(metrics, paths['output_dir'])
        
        print_evaluation_report(metrics, class_names)
        
        plot_confusion_matrix(
            metrics,
            class_names=class_names,
            save_path=os.path.join(paths['docs_dir'], 'confusion_matrix.png')
        )
    
    print("\n" + "="*60)
    print("STEP 5: PREDICTING ON TEST SET")
    print("="*60)
    
    if Config.USE_TTA:
        tta_transforms = get_tta_transforms()
        predictions, image_paths, probabilities = predict_with_tta(
            model, test_loader, device, tta_transforms
        )
    else:
        predictions, image_paths, probabilities = predict(
            model, test_loader, device
        )
    
    analyze_test_predictions(predictions, probabilities, class_names)
    
    print("\n" + "="*60)
    print("STEP 6: GENERATING SUBMISSION")
    print("="*60)
    submission = generate_submission(
        predictions=predictions,
        image_paths=image_paths,
        template_path=paths['template_path'],
        output_path=paths['output_path']
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nModel saved to: {paths['model_dir']}")
    print(f"Submission saved to: {paths['output_path']}")
    print(f"Documentation saved to: {paths['docs_dir']}")
    print(f"\nBest Validation F1-Score: {checkpoint['val_f1']:.4f}")
    print(f"Best Validation Accuracy: {checkpoint['val_acc']:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BDC Satria Data 2026 - Waste Classification Pipeline')
    
    parser.add_argument('--data_dir', type=str, default=Config.DATA_DIR,
                        help='Directory containing train and test folders (relative or absolute)')
    parser.add_argument('--template_path', type=str, default=Config.TEMPLATE_PATH,
                        help='Path to submission template file (relative or absolute)')
    
    parser.add_argument('--model_dir', type=str, default=Config.MODEL_DIR,
                        help='Directory to save trained models (relative or absolute)')
    parser.add_argument('--output_path', type=str, default=Config.OUTPUT_PATH,
                        help='Path to save submission file (relative or absolute)')
    parser.add_argument('--docs_dir', type=str, default=Config.DOCS_DIR,
                        help='Directory to save documentation and plots (relative or absolute)')
    
    parser.add_argument('--pretrained', action='store_true', default=Config.PRETRAINED,
                        help='Use pre-trained weights')
    
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=Config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=Config.WEIGHT_DECAY,
                        help='Weight decay for regularization')
    parser.add_argument('--val_split', type=float, default=Config.VAL_SPLIT,
                        help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=Config.NUM_WORKERS,
                        help='Number of data loading workers')
    
    parser.add_argument('--seed', type=int, default=Config.SEED,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--load_preprocessing', action='store_true',
                        help='Load preprocessing data from file instead of processing from scratch')
    
    parser.add_argument('--load_training', action='store_true',
                        help='Load training history from file instead of training from scratch')
    
    parser.add_argument('--load_evaluation', action='store_true',
                        help='Load evaluation metrics from file instead of evaluating from scratch')
    
    args = parser.parse_args()
    
    main(args)
