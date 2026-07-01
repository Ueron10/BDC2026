
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from tqdm import tqdm
import os
import time


class LabelSmoothingCrossEntropy(nn.Module):
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, x, target):
        log_probs = torch.log_softmax(x, dim=1)
        target = torch.empty_like(x).fill_(self.smoothing / (self.num_classes - 1)).scatter_(1, target.unsqueeze(1), self.confidence)
        loss = (-target * log_probs).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class WasteClassifier(nn.Module):
    
    def __init__(self, num_classes=3, backbone='efficientnet_b0', pretrained=True):
        super(WasteClassifier, self).__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'convnext_tiny':
            self.backbone = models.convnext_tiny(pretrained=pretrained)
            num_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class Trainer:
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=1e-3, weight_decay=1e-4, 
                 label_smoothing=0.1, use_mixup=True, mixup_alpha=0.2,
                 warmup_epochs=5, min_lr=1e-6):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = LabelSmoothingCrossEntropy(model.num_classes, smoothing=label_smoothing)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.base_lr = learning_rate
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=min_lr
        )
        
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        self.best_val_f1 = 0.0
        
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        if epoch <= self.warmup_epochs:
            lr = self.base_lr * (epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_mixup and np.random.rand() < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, self.mixup_alpha)
                outputs = self.model(images)
                loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        return epoch_loss, epoch_acc, val_f1
    
    def train(self, num_epochs, save_dir='model'):
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.backbone_name}")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(epoch)
            
            val_loss, val_acc, val_f1 = self.validate()
            
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            epoch_time = time.time() - start_time
            
            print(f'\nEpoch {epoch}/{num_epochs} - {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'backbone': self.model.backbone_name
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f'Best model saved with Val F1: {val_f1:.4f}')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_f1': val_f1,
                'history': self.history
            }, os.path.join(save_dir, 'last_model.pth'))
        
        print(f'\nTraining completed. Best Val F1: {self.best_val_f1:.4f}')
        return self.history


def load_model(checkpoint_path, num_classes=3, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    backbone = checkpoint.get('backbone', 'efficientnet_b0')
    
    model = WasteClassifier(num_classes=num_classes, backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint
