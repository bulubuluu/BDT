import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import lightning as L
import torchmetrics
import sklearn.metrics as m
import joblib
from xgboost import XGBClassifier

# --------------------------
# Set random seed
# --------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# --------------------------
# Load ROOT files with Weight column
# --------------------------
def load_root_with_weight(file_path, label=0):
    with uproot.open(file_path) as f:
        tree = f["LHEF"]
        df = tree.arrays(library="pd")
    df['label'] = label
    # Use Weight column as sample weights, clip negative values
    df['weight'] = np.clip(df['Weight'].values, 0, None)
    return df

df_sm  = load_root_with_weight("SM_LHEF.root", label=0)
df_liv = load_root_with_weight("LIV_LHEF.root", label=1)

df = pd.concat([df_sm, df_liv], ignore_index=True)

# --------------------------
# Feature selection and cleaning
# --------------------------
input_columns = [c for c in df.columns if c.startswith("Z1_") or c.startswith("Z2_")]
X = df[input_columns]
y = df['label']
w = df['weight'].values

X = X.replace([np.inf, -np.inf], np.nan)
if X.isna().sum().sum() > 0:
    X = X.fillna(X.median())

print(f"Total events: {len(df)}, Features: {len(input_columns)}")
print(f"Class 0 (SM): {sum(y==0)}, Class 1 (LIV): {sum(y==1)}")

# --------------------------
# Train/Validation/Test split
# --------------------------
X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
    X, y, w, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
    X_temp, y_temp, w_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# --------------------------
# Standard Scaling
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# --------------------------
# PyTorch Dataset with weights
# --------------------------
class DataFrameDataset(Dataset):
    def __init__(self, features, targets, weights=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets.values if hasattr(targets, 'values') else targets, dtype=torch.long)
        if weights is None:
            self.weights = torch.ones(len(targets), dtype=torch.float32)
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.weights[idx]

train_dataset = DataFrameDataset(X_train_scaled, y_train, w_train)
val_dataset   = DataFrameDataset(X_val_scaled, y_val, w_val)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# --------------------------
# Logistic Regression
# --------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train, sample_weight=w_train)
y_log_prob = log_model.predict_proba(X_test_scaled)[:,1]
joblib.dump(log_model, "models/log_model.pkl")

# --------------------------
# XGBoost (2.1.4 compatible)
# --------------------------
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    early_stopping_rounds=10
)

xgb_model.fit(
    X_train_scaled,
    y_train,
    sample_weight=w_train,
    eval_set=[(X_val_scaled, y_val)],
    sample_weight_eval_set=[w_val],
    verbose=False
)

y_xgb_prob = xgb_model.predict_proba(X_test_scaled)[:,1]
joblib.dump(xgb_model, "models/xgb_model.pkl")

# --------------------------
# PyTorch DNN
# --------------------------
class SimpleMLP(L.LightningModule):
    def __init__(self, input_size, hidden_size=128, output_size=2):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1    = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2    = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=output_size)
        self.val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.dropout2(x)
        x = self.layer3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, reduction='none')
        loss = (loss * w.to(loss.device)).mean()
        self.train_acc(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, reduction='none')
        loss = (loss * w.to(loss.device)).mean()
        self.val_acc(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

early_stop_callback = L.pytorch.callbacks.EarlyStopping(
    monitor='val_loss', mode='min', patience=5, verbose=True
)

model_mlp = SimpleMLP(input_size=X_train_scaled.shape[1])
trainer = L.Trainer(max_epochs=50, callbacks=[early_stop_callback], accelerator='auto', devices=1)
trainer.fit(model_mlp, train_loader, val_loader)

torch.save(model_mlp.state_dict(), "models/dnn_model.pt")

# --------------------------
# DNN Prediction
# --------------------------
class InferenceDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx]

test_dataset = InferenceDataset(X_test_scaled)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

model_mlp.eval()
device = next(model_mlp.parameters()).device
all_probs = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        logits = model_mlp(batch)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())

y_dnn_prob = np.concatenate(all_probs)[:,1]

# --------------------------
# Evaluation & Plots
# --------------------------
fpr_log, tpr_log, _ = m.roc_curve(y_test, y_log_prob)
fpr_xgb, tpr_xgb, _ = m.roc_curve(y_test, y_xgb_prob)
fpr_dnn, tpr_dnn, _ = m.roc_curve(y_test, y_dnn_prob)

auc_log = m.auc(fpr_log, tpr_log)
auc_xgb = m.auc(fpr_xgb, tpr_xgb)
auc_dnn = m.auc(fpr_dnn, tpr_dnn)

print(f"AUC - Logistic Regression: {auc_log:.4f}")
print(f"AUC - XGBoost:             {auc_xgb:.4f}")
print(f"AUC - DNN:                 {auc_dnn:.4f}")

plt.figure(figsize=(8,6))
plt.plot([0,1], [0,1], 'k--', label="Random")
plt.plot(fpr_log, tpr_log, label=f"Logistic (AUC={auc_log:.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={auc_xgb:.3f})")
plt.plot(fpr_dnn, tpr_dnn, label=f"DNN (AUC={auc_dnn:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SM vs LIV ROC Curves")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/ROC_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# Score distributions
model_scores = {
    "LogisticRegression": y_log_prob,
    "XGBoost": y_xgb_prob,
    "DNN": y_dnn_prob
}

for name, scores in model_scores.items():
    plt.figure(figsize=(7,5))
    bins = np.linspace(0,1,50)
    plt.hist(scores[y_test==1], bins=bins, density=True, histtype="step", color="red", linewidth=2, label="LIV")
    plt.hist(scores[y_test==0], bins=bins, density=True, histtype="step", color="blue", linewidth=2, label="SM")
    plt.xlabel(f"{name} score")
    plt.ylabel("Normalized density")
    plt.title(f"{name} Score Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{name}_scores.png", dpi=150, bbox_inches='tight')
    plt.close()

print("All plots saved to 'plots/' directory")
print("All models saved to 'models/' directory")