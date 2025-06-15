import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from concurrent.futures import ThreadPoolExecutor

# ==== 自动路径 ====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FLDATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'Data', 'FLData')
CLDATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'Data', 'CLData')
VAL_PATH = os.path.join(CLDATA_DIR, 'valData_cleaned.csv')  # 通常验证集用CLData更常见

COMM_ROUNDS = 5
LOCAL_EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

feature_cols = [
    'patientGender', 'patientAge', 'glasgowScale', 'hematocrit', 'hemoglobin',
    'leucocitos', 'lymphocytes', 'urea', 'creatinine', 'platelets', 'diuresis',
    'SBP', 'DBP', 'glasgowScale_missing'
]
target_col = 'outcomeType'

# ===== 自动读取 FLData 目录下所有 csv 客户端数据 =====
# ===== 只读取指定的8个训练客户端 =====
train_files = [
    "2ª_MOSSORÓ.csv",
    "5ª_SANTA_CRUZ.csv",
    "7ª_METROPOLITANA.csv",
    "4ª_CAICÓ.csv",
    "3ª_JOÃO_CAMARA.csv",
    "8ª_AÇU.csv",
    "1ª_SÃO_JOSÉ_DE_MIPIBU.csv",
    "6ª_PAU_DOS_FERROS.csv"
]
client_csvs = [f for f in train_files if os.path.exists(os.path.join(FLDATA_DIR, f))]
print(f"用于训练的客户端数据文件：\n{client_csvs}")



client_dfs = []
all_train_df = []
for csv_file in client_csvs:
    path = os.path.join(FLDATA_DIR, csv_file)
    df = pd.read_csv(path)
    df = df.dropna(subset=feature_cols + [target_col])
    if df.shape[0] == 0:
        print(f"Warning: {csv_file} 无有效样本，跳过")
        continue
    client_dfs.append(df)
    all_train_df.append(df)
all_train_df = pd.concat(all_train_df, ignore_index=True)

NUM_CLIENTS = len(client_dfs)
print(f"最终用于训练的客户端数：{NUM_CLIENTS}")

# ===== 标准化器全局拟合 =====
scaler = StandardScaler()
scaler.fit(all_train_df[feature_cols].values)

# ===== 客户端数据准备（各自SMOTE、各自标准化） =====
client_data = []
for df in client_dfs:
    X = scaler.transform(df[feature_cols].values)
    y = df[target_col].values.astype(np.float32)
    smote = SMOTE(random_state=SEED)
    X_bal, y_bal = smote.fit_resample(X, y)
    X_tensor = torch.tensor(X_bal, dtype=torch.float32)
    y_tensor = torch.tensor(y_bal, dtype=torch.float32)
    ds = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    client_data.append((loader, len(X_bal)))  # (DataLoader, 样本数)

# ===== 验证集准备 =====
val_df = pd.read_csv(VAL_PATH)
val_df = val_df.dropna(subset=feature_cols + [target_col])
X_val = scaler.transform(val_df[feature_cols].values)
y_val = val_df[target_col].values.astype(np.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# ====== 神经网络结构 ======
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

def get_model():
    model = DropoutNN(len(feature_cols))
    return model

def get_model_params(model):
    return [param.data.cpu().clone() for param in model.parameters()]

def set_model_params(model, params):
    for p, new_p in zip(model.parameters(), params):
        p.data.copy_(new_p)

# ===== 客户端本地训练函数 =====
def local_train(global_params, client_idx):
    model = get_model()
    set_model_params(model, global_params)
    loader, num_samples = client_data[client_idx]
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()
    model.train()
    for epoch in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    new_params = get_model_params(model)
    return (new_params, num_samples)

# ===== FedAvg加权聚合 =====
def fedavg(param_list, sample_counts):
    total = sum(sample_counts)
    new_params = []
    for param_tuples in zip(*param_list):
        stacked = torch.stack(param_tuples, dim=0)
        weighted = torch.zeros_like(stacked[0])
        for i in range(len(param_list)):
            weighted += stacked[i] * (sample_counts[i] / total)
        new_params.append(weighted)
    return new_params

# ====== 训练与聚合主循环 ======
global_model = get_model()
global_params = get_model_params(global_model)

for round in range(1, COMM_ROUNDS+1):
    print(f"\n=== Communication round {round} ===")
    with ThreadPoolExecutor(max_workers=NUM_CLIENTS) as executor:
        results = list(executor.map(lambda idx: local_train(global_params, idx), range(NUM_CLIENTS)))
    client_param_list, sample_counts = zip(*results)
    global_params = fedavg(client_param_list, sample_counts)
    set_model_params(global_model, global_params)
    global_model.eval()
    with torch.no_grad():
        y_pred = global_model(X_val_tensor).cpu().numpy()
        y_pred_label = (y_pred > 0.5).astype(int)
        acc = accuracy_score(y_val_tensor, y_pred_label)
        auc = roc_auc_score(y_val_tensor, y_pred)
        print(f"验证集准确率: {acc:.4f}, AUC: {auc:.4f}")
        print("分类报告:\n", classification_report(y_val_tensor, y_pred_label, digits=3))

# ====== 最终模型保存 ======
torch.save(global_model.state_dict(), os.path.join(FLDATA_DIR, 'fedavg_nn_model_final.pth'))
print("\n最终模型已保存到:", os.path.join(FLDATA_DIR, 'fedavg_nn_model_final.pth'))
