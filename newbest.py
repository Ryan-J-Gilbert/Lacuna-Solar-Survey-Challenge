import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from sklearn.metrics import mean_absolute_error
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

# Fixed Albumentations version warning
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Enhanced Dataset with Metadata
class SolarPanelDataset(Dataset):
    def __init__(self, dataframe, transform=None, to_train=True):
        self.dataframe = dataframe
        self.transform = transform
        self.to_train = to_train
        self.placement_map = {"roof": 0, "openspace": 1, "r_openspace": 2, "S-unknown": 3}
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = cv2.imread(row["path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Correct color conversion
        
        # Improved metadata encoding
        metadata = torch.zeros(5)
        metadata[0] = 1.0 if row["img_origin"] == "D" else 0.0
        placement = self.placement_map.get(row["placement"], 3)
        metadata[1 + placement] = 1.0  # One-hot encoding
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        if self.to_train:
            target = torch.tensor([row["boil_nbr"], row["pan_nbr"]], dtype=torch.float32)
            return image, metadata, target
        return image, metadata

# Custom Counting Head
class CountingHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.count_regressor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        # Using log-softplus for count data
        counts = self.count_regressor(x)
        return F.softplus(counts)  # Ensures positive output for count data

# Model with Metadata and Improved Count-specific Head
class EfficientNetV2Meta(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_b3", pretrained=True, num_classes=0)
        
        # Metadata processor
        self.meta_processor = nn.Sequential(
            nn.Linear(5, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
        # Cross-attention between image and metadata
        self.img_projection = nn.Linear(self.backbone.num_features, 64)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        
        # Count-specific head
        self.counting_head = CountingHead(self.backbone.num_features + 64)
        
    def forward(self, image, metadata):
        img_features = self.backbone(image)
        meta_features = self.meta_processor(metadata)
        
        # Cross-attention between image and metadata features
        img_proj = self.img_projection(img_features).unsqueeze(0)  # [1, B, D]
        meta_proj = meta_features.unsqueeze(0)  # [1, B, D]
        attn_output, _ = self.attention(img_proj, img_proj, img_proj) # self.attention(img_proj, meta_proj, meta_proj)
        
        # Combine features
        combined = torch.cat([img_features, attn_output.squeeze(0)], dim=1)
        
        # Get count predictions
        return self.counting_head(combined)
# class EfficientNetV2Meta(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = timm.create_model("tf_efficientnetv2_b3", pretrained=True, num_classes=0)  # you can even try Larger backbone
#         self.meta_processor = nn.Sequential(
#             nn.Linear(5, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64)
#         )
#         self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)
#         self.regressor = nn.Sequential(
#             nn.Linear(self.backbone.num_features + 64, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 2),
#             nn.Softplus()  # Better for count predictions
#         )
#     def forward(self, image, metadata):
#         img_features = self.backbone(image)
#         meta_features = self.meta_processor(metadata.unsqueeze(0))
#         attn_output, _ = self.attention(meta_features, meta_features, meta_features)
#         combined = torch.cat([img_features, attn_output.squeeze(0)], dim=1)
#         return self.regressor(combined)

# Advanced Augmentation
train_transform = A.Compose([
    A.Resize(512, 512),  # Resize without cropping to preserve all elements
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.5),
    A.HueSaturationValue(p=0.3),
    # A.ShiftScaleRotate(p=0.5, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),  # Gentle transform
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Improved count-focused loss function
class CountLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0)
        self.l1 = nn.L1Loss()
        
    def forward(self, outputs, targets):
        # Weighted combination of losses
        huber_loss = self.huber(outputs, targets)
        l1_loss = self.l1(outputs, targets)
        # Higher weight to Huber which is more robust to outliers
        return 0.7 * huber_loss + 0.3 * l1_loss

# Calibration function for post-processing predictions
def calibrate_predictions(val_preds, val_truths):
    """
    Learn optimal rounding thresholds from validation data
    """
    best_mae = float('inf')
    best_thresholds = [0.5, 0.5]  # Default thresholds for boiler and panel
    
    # Grid search for optimal thresholds
    for boil_t in np.arange(0.3, 0.7, 0.05):
        for pan_t in np.arange(0.3, 0.7, 0.05):
            # Apply thresholds
            calibrated = val_preds.copy()
            calibrated[:, 0] = np.floor(calibrated[:, 0]) + (calibrated[:, 0] % 1 >= boil_t).astype(float)
            calibrated[:, 1] = np.floor(calibrated[:, 1]) + (calibrated[:, 1] % 1 >= pan_t).astype(float)
            
            # Calculate MAE
            mae = mean_absolute_error(val_truths, calibrated)
            
            if mae < best_mae:
                best_mae = mae
                best_thresholds = [boil_t, pan_t]
    
    print(f"Best calibration thresholds: Boiler={best_thresholds[0]:.2f}, Panel={best_thresholds[1]:.2f}")
    print(f"Calibrated validation MAE: {best_mae:.4f}")
    
    return best_thresholds

# Apply calibration to predictions
def apply_calibration(predictions, thresholds):
    calibrated = predictions.copy()
    calibrated[:, 0] = np.floor(calibrated[:, 0]) + (calibrated[:, 0] % 1 >= thresholds[0]).astype(float)
    calibrated[:, 1] = np.floor(calibrated[:, 1]) + (calibrated[:, 1] % 1 >= thresholds[1]).astype(float)
    return calibrated

# Training Configuration with improved learning rate strategy and early stopping
def train(fold=0, epochs=40, batch_size=16, patience=5):  # Increased epochs from 20 to 30
    train_df = pd.read_csv("Train.csv")
    train_df = train_df.groupby("ID").agg({
        "boil_nbr": "sum",
        "pan_nbr": "sum",
        "img_origin": "first",
        "placement": "first"
    }).reset_index()
    train_df["path"] = "images/" + train_df["ID"] + ".jpg"


    # print("WARNING, REDUCING DATA TO 10%. REMOVE FOR FULL MODEL!")
    # train_df = train_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(train_df))
    train_idx, val_idx = splits[fold]
    
    train_ds = SolarPanelDataset(train_df.iloc[train_idx], transform=train_transform)
    val_ds = SolarPanelDataset(train_df.iloc[val_idx], transform=test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    model = EfficientNetV2Meta().cuda()
    criterion = CountLoss()
    
    # Improved learning rate strategy with OneCycleLR
    # Define different learning rates and weight decay for different parts of the model
    # optimizer = optim.AdamW([
    #     {"params": model.backbone.parameters(), "lr": 3e-4, "weight_decay": 1e-5},  # Backbone with lower LR and decay
    #     {"params": model.meta_processor.parameters(), "lr": 3e-3, "weight_decay": 1e-4},  # Metadata processor
    #     {"params": model.attention.parameters(), "lr": 3e-3, "weight_decay": 1e-4},  # Attention layers
    #     {"params": model.regressor.parameters(), "lr": 3e-3, "weight_decay": 1e-4}  # Regressor with higher LR
    # ])
    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 3e-4, "weight_decay": 1e-5},  # Backbone with lower LR and decay
        {"params": model.meta_processor.parameters(), "lr": 3e-2, "weight_decay": 1e-4},  # Metadata processor
        {"params": model.attention.parameters(), "lr": 3e-2, "weight_decay": 1e-4},  # Attention layers
        {"params": model.counting_head.parameters(), "lr": 3e-2, "weight_decay": 1e-4}  # Regressor with higher LR
    ])
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
        div_factor=25,
        final_div_factor=10000
    )
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    scaler = GradScaler()
    
    # Early stopping variables
    best_mae = float('inf')
    epochs_no_improve = 0
    early_stop = False
    
    # For collecting validation predictions for calibration
    val_preds_for_calibration = []
    val_truths_for_calibration = []
    
    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for images, meta, targets in pbar:
            images = images.cuda(non_blocking=True)
            meta = meta.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(images, meta)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            # Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Step the scheduler after every batch
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        preds, truths = [], []
        
        with torch.no_grad():
            for images, meta, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.cuda(non_blocking=True)
                meta = meta.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                
                with autocast(device_type='cuda'):
                    outputs = model(images, meta)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                preds.append(outputs.cpu().numpy())
                truths.append(targets.cpu().numpy())
        
        # Metrics calculation
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        
        mae = mean_absolute_error(truths, preds)
        
        # Calculate also integer MAE (standard rounding)
        int_preds = np.round(preds)
        int_mae = mean_absolute_error(truths, int_preds)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Raw Val MAE: {mae:.4f} | Integer Val MAE: {int_mae:.4f}")
        
        # Save validation predictions for best model only
        if mae < best_mae:
            print(f"MAE improved from {best_mae:.4f} to {mae:.4f}")
            best_mae = mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_model_fold{fold}.pth")
            # Store the validation predictions for the best model
            val_preds_for_calibration = preds.copy()
            val_truths_for_calibration = truths.copy()
        else:
            epochs_no_improve += 1
            print(f"MAE did not improve. Epochs without improvement: {epochs_no_improve}/{patience}")
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            early_stop = True
            break
    
    # Final calibration on validation set - now using the saved best validation predictions
    # This fixes the error when early stopping is triggered
    if len(val_preds_for_calibration) > 0:  # Safety check
        calibration_thresholds = calibrate_predictions(val_preds_for_calibration, val_truths_for_calibration)
        # Save calibration thresholds
        np.save(f"calibration_thresholds_fold{fold}.npy", np.array(calibration_thresholds))
    else:
        print("Warning: No validation predictions available for calibration.")
        calibration_thresholds = [0.5, 0.5]  # Default fallback
        np.save(f"calibration_thresholds_fold{fold}.npy", np.array(calibration_thresholds))
    
    print(f"Fold {fold} completed. Best validation MAE: {best_mae:.4f}")
    return best_mae, calibration_thresholds

# Enhanced TTA (Test Time Augmentation) function
def predict_with_enhanced_tta(test_df, model_paths, calibration_paths, batch_size=32):
    test_df["path"] = "images/" + test_df["ID"] + ".jpg"
    test_ds = SolarPanelDataset(test_df, transform=test_transform, to_train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    predictions = np.zeros((len(test_df), 2))
    
    # Define TTA transforms
    print("NO TTA FOR THIS RUN!!!")
    tta_transforms = [
        # Original image
        lambda img: img,
        # Horizontal flip
        # lambda img: torch.flip(img, dims=[3]),
        # Vertical flip
        # lambda img: torch.flip(img, dims=[2]),
        # Both flips (180 rotation)
        # lambda img: torch.flip(torch.flip(img, dims=[3]), dims=[2])
    ]
    
    # Load and ensemble models
    for i, path in enumerate(model_paths):
        model = EfficientNetV2Meta().cuda()
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for images, meta in tqdm(test_loader, desc=f"Inference {i+1}/{len(model_paths)} - {path}"):
                images = images.cuda()
                meta = meta.cuda()
                
                # Apply all TTA variants
                tta_batch_preds = []
                with autocast(device_type='cuda'):
                    for transform in tta_transforms:
                        transformed_images = transform(images)
                        outputs = model(transformed_images, meta)
                        tta_batch_preds.append(outputs.cpu().numpy())
                
                # Average TTA predictions for this batch
                batch_preds = np.mean(tta_batch_preds, axis=0)
                fold_preds.append(batch_preds)
        
        # Combine all batch predictions for this fold
        fold_preds = np.vstack(fold_preds)
        
        # Add to ensemble predictions
        predictions += fold_preds / len(model_paths)
    
    # Load and average calibration thresholds
    all_thresholds = []
    for path in calibration_paths:
        thresholds = np.load(path)
        all_thresholds.append(thresholds)
    
    avg_thresholds = np.mean(all_thresholds, axis=0)
    print(f"Average calibration thresholds: Boiler={avg_thresholds[0]:.2f}, Panel={avg_thresholds[1]:.2f}")
    
    # Create calibrated predictions
    calibrated_predictions = apply_calibration(predictions, avg_thresholds)
    
    return predictions, calibrated_predictions

# Main Execution with more epochs and enhanced TTA
if __name__ == "__main__":

    training = True
    folds = 5
    model_paths = []
    calibration_paths = []

    if training:
        # Train multiple folds     
        for fold in range(folds):
            print(f"\n{'='*50}")
            print(f"Training fold {fold+1}/{folds}")
            print(f"{'='*50}")
            
            # Increased epochs and using early stopping
            best_mae, thresholds = train(
                fold=fold, 
                epochs=40,
                batch_size=32,
                patience=5
            )
            
            model_paths.append(f"best_model_fold{fold}.pth")
            calibration_paths.append(f"calibration_thresholds_fold{fold}.npy")

    else:
        model_paths = [f"best_model_fold{fold}.pth" for fold in range(folds)]
        calibration_paths = [f"calibration_thresholds_fold{fold}.npy" for fold in range(folds)]

    
    # Prepare submission with enhanced TTA
    test_df = pd.read_csv("Test.csv")
    raw_predictions, calibrated_predictions = predict_with_enhanced_tta(
        test_df, model_paths, calibration_paths, batch_size=32
    )
    
    # Create submissions
    # 1. Raw predictions
    raw_submission = pd.DataFrame({
        "ID": np.repeat(test_df["ID"].values, 2),
        "Target": raw_predictions.flatten()
    })
    raw_submission["ID"] += np.where(
        raw_submission.groupby("ID").cumcount() == 0,
        "_boil",
        "_pan"
    )
    raw_submission.to_csv("submission_raw.csv", index=False)
    
    # 2. Integer predictions (traditional rounding)
    int_submission = pd.DataFrame({
        "ID": np.repeat(test_df["ID"].values, 2),
        "Target": np.round(raw_predictions).astype(int).flatten()
    })
    int_submission["ID"] += np.where(
        int_submission.groupby("ID").cumcount() == 0,
        "_boil",
        "_pan"
    )
    int_submission.to_csv("submission_integer.csv", index=False)
    
    # 3. Calibrated predictions
    cal_submission = pd.DataFrame({
        "ID": np.repeat(test_df["ID"].values, 2),
        "Target": calibrated_predictions.astype(int).flatten()
    })
    cal_submission["ID"] += np.where(
        cal_submission.groupby("ID").cumcount() == 0,
        "_boil",
        "_pan"
    )
    cal_submission.to_csv("submission_calibrated.csv", index=False)
    
    print("\nSubmissions saved:")
    print(f"1. Raw predictions: {raw_submission.shape}")
    print(f"2. Integer predictions: {int_submission.shape}")
    print(f"3. Calibrated predictions: {cal_submission.shape}")


# Aknowledgements:
# - The code is inspired by various sources including the Zindi community and
#   contributions from users like zulo40, who provided valuable insights and
#   code snippets that helped shape this solution.