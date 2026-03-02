import os
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
import lightning as L
import torchmetrics
import sklearn.metrics as m

# ==========================
# 1. Load SM and LIV ROOT files
# ==========================
def load_root(file_path):
    with uproot.open(file_path) as f:
        tree = f["LHEF"]
        df = tree.arrays(library="pd")
    return df

df_sm  = load_root("SM_LHEF.root")
df_liv = load_root("LIV_LHEF.root")

# Add labels: SM=0, LIV=1
df_sm['label']  = 0
df_liv['label'] = 1

# Concatenate
df = pd.concat([df_sm, df_liv], axis=0, ignore_index=True)

# ==========================
# 2. Select features: Z1 + Z2
# ==========================
input_columns = [col for col in df.columns if col.startswith("Z1_") or col.startswith("Z2_")]
X = df[input_columns]
y = df['label']

print(f"Total events: {len(df)}, Features: {len(input_columns)}")
print(f"Class 0 (SM): {sum(y==0)}, Class 1 (LIV): {sum(y==1)}")

# =========================================
# 3. Train/Validation/Test split (80/10/10)
# =========================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# =========================================
# 4. Standardization
# =========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# =========================================
# 5. PyTorch Dataset
# =========================================
class DataFrameDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets  = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

train_dataset = DataFrameDataset(X_train_scaled, y_train.values)
val_dataset   = DataFrameDataset(X_val_scaled, y_val.values)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# =========================================
# 6. Logistic Regression
# =========================================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

# =========================================
# 7. XGBoost
# =========================================
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="logloss",
    early_stopping_rounds=10,
    random_state=42
)
xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=False
)
y_xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]

# =========================================
# 8. Deep Neural Network
# =========================================
class SimpleMLP(L.LightningModule):
    def __init__(self, input_size=X_train_scaled.shape[1], hidden_size=32, output_size=2, dropout=0.3):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.layer3 = nn.Linear(hidden_size, output_size)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=output_size)
        self.val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.layer3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

early_stop_callback = L.pytorch.callbacks.EarlyStopping(
    monitor='val_loss', mode='min', patience=5, verbose=True
)

model_mlp = SimpleMLP()
trainer = L.Trainer(
    max_epochs=50,
    callbacks=[early_stop_callback],
    enable_progress_bar=True
)

trainer.fit(model_mlp, train_loader, val_loader)

# =========================================
# 9. DNN Predictions
# =========================================
class InferenceDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx]

test_dataset = InferenceDataset(X_test_scaled)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

predictions_mlp = trainer.predict(model_mlp, dataloaders=test_loader)
proba_list = [batch.cpu().numpy() for batch in predictions_mlp]
y_dnn_prob = np.concatenate(proba_list)[:, 1]

# =========================================
# 10. ROC & AUC
# =========================================
fpr_log, tpr_log, _ = m.roc_curve(y_test, y_log_prob)
fpr_xgb, tpr_xgb, _ = m.roc_curve(y_test, y_xgb_prob)
fpr_dnn, tpr_dnn, _ = m.roc_curve(y_test, y_dnn_prob)

auc_log = m.auc(fpr_log, tpr_log)
auc_xgb = m.auc(fpr_xgb, tpr_xgb)
auc_dnn = m.auc(fpr_dnn, tpr_dnn)

print(f"\nAUC - Logistic Regression: {auc_log:.3f}")
print(f"AUC - XGBoost:             {auc_xgb:.3f}")
print(f"AUC - DNN:                  {auc_dnn:.3f}")

# =========================================
# 11. Plot ROC curves
# =========================================
plt.figure(figsize=(7,6))
plt.plot([0,1],[0,1], 'k--', label="Random")
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC={auc_log:.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={auc_xgb:.3f})")
plt.plot(fpr_dnn, tpr_dnn, label=f"DNN (AUC={auc_dnn:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SM vs LIV Classification (Z1+Z2 features)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/ROC_Z1Z2_features.png", dpi=150)

# =========================================
# 12. Plot model score distributions (like sklearn style)
# =========================================
os.makedirs("plots", exist_ok=True)
model_scores = {
    "LogisticRegression": y_log_prob,
    "XGBoost": y_xgb_prob,
    "DNN": y_dnn_prob
}

for name, scores in model_scores.items():
    plt.figure(figsize=(6,6))
    bins = np.linspace(0,1,50)
    # signal = LIV (label=1), background = SM (label=0)
    plt.hist(scores[y_test==1], bins=bins, density=True, histtype="step", color="red", label="LIV (signal)")
    plt.hist(scores[y_test==0], bins=bins, density=True, histtype="step", color="blue", label="SM (background)")
    plt.xlabel(f"{name} score (probability)")
    plt.ylabel("Normalized events")
    plt.title(f"{name} Score Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{name}_score_distribution.png", dpi=150)
    plt.close()
    print(f"Saved {name} score plot to plots/{name}_score_distribution.png")