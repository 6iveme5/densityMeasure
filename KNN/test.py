import os
import pandas as pd
import numpy as np
import joblib
from KNN import KNNConfidence

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '../Data/CLData')

feature_cols = [
    'patientGender', 'patientAge', 'glasgowScale', 'hematocrit', 'hemoglobin',
    'leucocitos', 'lymphocytes', 'urea', 'creatinine', 'platelets', 'diuresis',
    'SBP', 'DBP', 'glasgowScale_missing'
]
target_col = 'outcomeType'

# 数据加载
train_df = pd.read_csv(os.path.join(DATA_DIR, 'trainData_cleaned.csv'))
val_df = pd.read_csv(os.path.join(DATA_DIR, 'valData_cleaned.csv'))
scaler = joblib.load(os.path.join(DATA_DIR, 'scaler.pkl'))

X_train = scaler.transform(train_df[feature_cols].dropna().values)
y_train = train_df[target_col].dropna().values.astype(int)
X_val = scaler.transform(val_df[feature_cols].dropna().values)
# 需要模型预测类别
# 这里假设你已有 y_pred_val（形状和X_val行数相同）

# 加载模型并预测
import torch
class DropoutNN(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(16, 1), torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)
model = DropoutNN(len(feature_cols))
model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'nn_model_dropout.pth')))
model.eval()
with torch.no_grad():
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_pred_prob = model(X_val_tensor).cpu().numpy().flatten()
    y_pred_val = (y_pred_prob > 0.5).astype(int)

# KNNConfidence打分
knn_conf = KNNConfidence(k=10)
knn_conf.fit(X_train, y_train)
knn_scores = knn_conf.get_score(X_val, y_pred_val)

# 保存分数
val_result = val_df[feature_cols].dropna().copy()
val_result['KNNConfidence'] = knn_scores
val_result.to_csv(os.path.join(DATA_DIR, 'valData_knn_conf.csv'), index=False)
print('KNNConfidence分数已保存到 valData_knn_conf.csv')
