
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


def predict(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_image_paths = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, image_paths in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_image_paths.extend(image_paths)
            all_probabilities.extend(probs.cpu().numpy())
    
    return np.array(all_predictions), all_image_paths, np.array(all_probabilities)


def generate_submission(predictions, image_paths, template_path, output_path):
    template = pd.read_csv(template_path)
    
    image_ids = []
    for path in image_paths:
        filename = os.path.basename(path)
        image_id = int(filename.split('.')[0])
        image_ids.append(image_id)
    
    submission_data = pd.DataFrame({
        'id': image_ids,
        'predicted': predictions
    })
    
    submission_data = submission_data.sort_values('id').reset_index(drop=True)
    
    submission_data = submission_data.set_index('id').loc[template['id']].reset_index()
    
    submission_data.to_csv(output_path, index=False)
    
    print(f"Submission saved to {output_path}")
    print(f"Total predictions: {len(submission_data)}")
    print(f"\nPrediction distribution:")
    print(submission_data['predicted'].value_counts().sort_index())
    
    return submission_data


def predict_with_tta(model, test_loader, device, tta_transforms=None):
    model.eval()
    
    if tta_transforms is None:
        return predict(model, test_loader, device)
    
    all_image_paths = []
    all_probabilities_list = []
    for transform_idx, transform in enumerate(tta_transforms):
        test_loader.dataset.transform = transform
        
        model_probs = []
        batch_image_paths = []
        
        with torch.no_grad():
            for images, image_paths in tqdm(test_loader, desc=f'TTA {transform_idx+1}'):
                images = images.to(device)
                
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                
                model_probs.append(probs.cpu().numpy())
                if transform_idx == 0:
                    batch_image_paths.extend(image_paths)
        
        all_probabilities_list.append(np.vstack(model_probs))
        if transform_idx == 0:
            all_image_paths = batch_image_paths
    
    avg_probs = np.mean(all_probabilities_list, axis=0)
    predictions = np.argmax(avg_probs, axis=1)
    
    return predictions, all_image_paths, avg_probs


def analyze_test_predictions(predictions, probabilities, class_names=None):
    if class_names is None:
        class_names = ['Recyclable', 'Electronic', 'Organic']
    
    print("\n" + "="*60)
    print("TEST PREDICTION ANALYSIS")
    print("="*60)
    
    print(f"\nPrediction Distribution:")
    unique, counts = np.unique(predictions, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"  {class_names[idx]} (Class {idx}): {count} ({count/len(predictions)*100:.2f}%)")
    
    confidences = np.max(probabilities, axis=1)
    print(f"\nConfidence Statistics:")
    print(f"  Mean:     {np.mean(confidences):.4f}")
    print(f"  Std:      {np.std(confidences):.4f}")
    print(f"  Min:      {np.min(confidences):.4f}")
    print(f"  Max:      {np.max(confidences):.4f}")
    print(f"  Median:   {np.median(confidences):.4f}")
    
    low_conf_threshold = 0.5
    low_conf_mask = confidences < low_conf_threshold
    print(f"\nLow Confidence Predictions (< {low_conf_threshold}): {np.sum(low_conf_mask)}")
    
    if np.sum(low_conf_mask) > 0:
        low_conf_preds = predictions[low_conf_mask]
        unique_lc, counts_lc = np.unique(low_conf_preds, return_counts=True)
        for idx, count in zip(unique_lc, counts_lc):
            print(f"    {class_names[idx]} (Class {idx}): {count}")
