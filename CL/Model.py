import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ==== 自动路径适配 ====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CLDATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'Data', 'CLData')

# 1. 读取数据
train_df = pd.read_csv(os.path.join(CLDATA_DIR, 'trainData_cleaned.csv'))
val_df = pd.read_csv(os.path.join(CLDATA_DIR, 'valData_cleaned.csv'))

# 2. 特征与标签
feature_cols = [
    'patientGender', 'patientAge', 'glasgowScale', 'hematocrit', 'hemoglobin',
    'leucocitos', 'lymphocytes', 'urea', 'creatinine', 'platelets', 'diuresis',
    'SBP', 'DBP', 'glasgowScale_missing'
]
target_col = 'outcomeType'

# 3. 去除缺失值
train_df = train_df.dropna(subset=feature_cols+[target_col])
val_df = val_df.dropna(subset=feature_cols+[target_col])

# 4. 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols].values)
X_val = scaler.transform(val_df[feature_cols].values)
y_train = train_df[target_col].values.astype(np.float32)
y_val = val_df[target_col].values.astype(np.float32)

# 5. SMOTE过采样
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 6. 转为torch张量
X_train_bal = torch.tensor(X_train_bal, dtype=torch.float32)
y_train_bal = torch.tensor(y_train_bal, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# 7. 数据加载器
train_ds = TensorDataset(X_train_bal, y_train_bal)
val_ds = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)

# 8. 改进神经网络结构
class DropoutNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# 9. 初始化模型与优化器（含权重衰减L2正则）
model = DropoutNN(len(feature_cols))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCELoss()

# 10. 训练
for epoch in range(1, 51):  # 训练50轮
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    avg_loss = total_loss / len(train_loader.dataset)
    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch {epoch} - Train Loss: {avg_loss:.4f}')

# 11. 验证
model.eval()
with torch.no_grad():
    y_pred_val = model(X_val).cpu().numpy()
    y_pred_label = (y_pred_val > 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred_label)
    print('\nValidation accuracy:', acc)
    print('\nClassification report:\n', classification_report(y_val, y_pred_label, digits=3))

# 12. 如需保存模型
torch.save(model.state_dict(), os.path.join(CLDATA_DIR, 'nn_model_dropout.pth'))
