import os
import pandas as pd
import numpy as np
import joblib
import torch
from Trustscore import TrustScore  # 假设你把TrustScore类写在Trustscore/Trustscore.py中

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

class DropoutNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)
# ==== 路径自动适配 ====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../Data/CLData'))

# 1. 训练集特征名
feature_cols = [
    'patientGender', 'patientAge', 'glasgowScale', 'hematocrit', 'hemoglobin',
    'leucocitos', 'lymphocytes', 'urea', 'creatinine', 'platelets', 'diuresis',
    'SBP', 'DBP', 'glasgowScale_missing'
]
target_col = 'outcomeType'

# ===== 路径 =====
train_path = os.path.join(DATA_DIR, 'trainData_cleaned.csv')
val_path = os.path.join(DATA_DIR, 'valData_cleaned.csv')
scaler_path = os.path.join(DATA_DIR, 'scaler.pkl')
model_path = os.path.join(DATA_DIR, 'nn_model_dropout.pth')

# ===== 数据加载与标准化 =====
df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
X_train = df_train[feature_cols].dropna().values
y_train = df_train[target_col].dropna().values.astype(int)
X_val = df_val[feature_cols].dropna().values
y_val = df_val[target_col].dropna().values.astype(int)

# 标准化
scaler = joblib.load(scaler_path)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ===== 加载模型并预测 =====
input_dim = len(feature_cols)
model = DropoutNN(input_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

with torch.no_grad():
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_pred_proba = model(X_val_tensor).cpu().numpy().flatten()
    y_pred_val = (y_pred_proba > 0.5).astype(int)

# ===== TrustScore评估 =====
trustscore = TrustScore(k=10, filtering="none")  # 可调参
trustscore.fit(X_train_scaled, y_train)
val_trust_scores = trustscore.get_score(X_val_scaled, y_pred_val)

# ===== 保存结果 =====
# 新建结果只针对dropna后的index有效
val_result = df_val[feature_cols].dropna().copy()
val_result['TrustScore'] = val_trust_scores

# 合并回原始df_val
df_val.loc[val_result.index, 'TrustScore'] = val_result['TrustScore']

val_save_path = os.path.join(DATA_DIR, 'valData_trustscore.csv')
df_val.to_csv(val_save_path, index=False)
print(df_val[['TrustScore', target_col]].head(10))
