import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. 读取数据
val_df = pd.read_csv('./Data/valData_cleaned.csv')
density_df = pd.read_csv('./Data/valData_gmm_density.csv')
train_df = pd.read_csv('./Data/trainData_cleaned.csv')

# 2. 特征和标签
feature_cols = [
    'patientGender', 'patientAge', 'glasgowScale', 'hematocrit', 'hemoglobin',
    'leucocitos', 'lymphocytes', 'urea', 'creatinine', 'platelets', 'diuresis',
    'SBP', 'DBP', 'glasgowScale_missing'
]
target_col = 'outcomeType'

# 3. 标准化
scaler = StandardScaler()
scaler.fit(train_df[feature_cols].dropna().values)
X_val = scaler.transform(val_df[feature_cols].dropna().values)
y_val = val_df[target_col].dropna().values.astype(int)

# 4. 定义和加载神经网络模型
class DropoutNN(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = DropoutNN(len(feature_cols))
model.load_state_dict(torch.load('./Data/nn_model_dropout.pth', map_location='cpu'))
model.eval()

# 5. 生成神经网络预测
with torch.no_grad():
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_pred_prob = model(X_val_tensor).cpu().numpy()
    y_pred_label = (y_pred_prob > 0.5).astype(int)

# 6. 截取与density_df一致长度（如果有缺失/顺序不一致，以短的为准）
N = min(len(density_df), len(y_pred_label), len(y_val))
density_df = density_df.iloc[:N].reset_index(drop=True)
y_pred_label = y_pred_label[:N]
y_val = y_val[:N]

# 7. 合并神经网络输出和真实标签
density_df['nn_pred'] = y_pred_label
density_df['nn_true'] = y_val

# 8. 用pd.qcut分高中低三组
density_df['density_group'] = pd.qcut(density_df['GMM_density'], q=3, labels=['低', '中', '高'])

for group in ['低', '中', '高']:
    mask = density_df['density_group'] == group
    acc = accuracy_score(density_df.loc[mask, 'nn_true'], density_df.loc[mask, 'nn_pred'])
    print(f"\n{group}密度区神经网络准确率: {acc:.3f}")
    print(f"\n{group}密度区分类报告:")
    print(classification_report(
        density_df.loc[mask, 'nn_true'],
        density_df.loc[mask, 'nn_pred'],
        digits=3
    ))