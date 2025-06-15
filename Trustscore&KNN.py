import os
import pandas as pd
import torch
import joblib
from sklearn.metrics import accuracy_score, classification_report

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'Data')
CLDATA_DIR = os.path.join(DATA_DIR, 'CLData')

feature_cols = [
    'patientGender', 'patientAge', 'glasgowScale', 'hematocrit', 'hemoglobin',
    'leucocitos', 'lymphocytes', 'urea', 'creatinine', 'platelets', 'diuresis',
    'SBP', 'DBP', 'glasgowScale_missing'
]
target_col = 'outcomeType'

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

def eval_reliability_group(DATA_DIR, model_filename, scaler_filename,
                          trustscore_filename, knnconf_filename, name='CL'):
    # 1. 加载数据
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'valData_cleaned.csv'))
    trustscore_df = pd.read_csv(os.path.join(DATA_DIR, trustscore_filename))
    knnconf_df = pd.read_csv(os.path.join(DATA_DIR, knnconf_filename))
    trustscore_df = trustscore_df.reset_index(drop=True)
    knnconf_df = knnconf_df.reset_index(drop=True)

    # 2. 标准化
    scaler_path = os.path.join(DATA_DIR, scaler_filename)
    scaler = joblib.load(scaler_path)
    X_val = scaler.transform(val_df[feature_cols].dropna().values)
    y_val = val_df[target_col].dropna().values.astype(int)

    # 3. 加载模型并预测
    model = DropoutNN(len(feature_cols))
    model_path = os.path.join(DATA_DIR, model_filename)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_pred_prob = model(X_val_tensor).cpu().numpy().flatten()
        y_pred_label = (y_pred_prob > 0.5).astype(int)

    # 4. 对齐长度
    N = min(len(trustscore_df), len(knnconf_df), len(y_pred_label), len(y_val))
    trustscore_df = trustscore_df.iloc[:N].reset_index(drop=True)
    knnconf_df = knnconf_df.iloc[:N].reset_index(drop=True)
    y_pred_label = y_pred_label[:N]
    y_val = y_val[:N]

    # 5. 合并预测和标签
    trustscore_df['nn_pred'] = y_pred_label
    trustscore_df['nn_true'] = y_val
    knnconf_df['nn_pred'] = y_pred_label
    knnconf_df['nn_true'] = y_val

    # 6. TrustScore分组评估
    trustscore_df['trustscore_group'] = pd.qcut(trustscore_df['TrustScore'], q=3, labels=['低', '中', '高'])
    print(f"\n============== {name} TrustScore 分组评估结果 ==============")
    for group in ['低', '中', '高']:
        mask = trustscore_df['trustscore_group'] == group
        acc = accuracy_score(trustscore_df.loc[mask, 'nn_true'], trustscore_df.loc[mask, 'nn_pred'])
        print(f"\n{name} TrustScore {group}分组准确率: {acc:.3f}")
        print(f"{name} TrustScore {group}分组分类报告:")
        print(classification_report(
            trustscore_df.loc[mask, 'nn_true'],
            trustscore_df.loc[mask, 'nn_pred'],
            digits=3
        ))

    # 7. KNNConfidence分组评估
    try:
        knnconf_bins = pd.qcut(knnconf_df['KNNConfidence'], q=3, labels=['低', '中', '高'], duplicates='drop')
    except ValueError as e:
        # 自动适配标签长度
        unique_bins = pd.qcut(knnconf_df['KNNConfidence'], q=3, retbins=True, duplicates='drop')[1]
        num_groups = len(unique_bins) - 1
        group_labels = ['低', '高'] if num_groups == 2 else [f'组{i + 1}' for i in range(num_groups)]
        knnconf_bins = pd.qcut(knnconf_df['KNNConfidence'], q=num_groups, labels=group_labels, duplicates='drop')
    knnconf_df['knnconf_group'] = knnconf_bins

    print(f"\n============== {name} KNNConfidence 分组评估结果 ==============")
    for group in knnconf_df['knnconf_group'].cat.categories:
        mask = knnconf_df['knnconf_group'] == group
        acc = accuracy_score(knnconf_df.loc[mask, 'nn_true'], knnconf_df.loc[mask, 'nn_pred'])
        print(f"\n{name} KNNConfidence {group}分组准确率: {acc:.3f}")
        print(f"{name} KNNConfidence {group}分组分类报告:")
        print(classification_report(
            knnconf_df.loc[mask, 'nn_true'],
            knnconf_df.loc[mask, 'nn_pred'],
            digits=3
        ))

    # 8. 保存结果
    trustscore_df.to_csv(os.path.join(DATA_DIR, f'{name}_val_result_trustscore_group.csv'), index=False)
    knnconf_df.to_csv(os.path.join(DATA_DIR, f'{name}_val_result_knnconf_group.csv'), index=False)
    print(f"{name} TrustScore/KNNConfidence 分组结果已保存。")

if __name__ == '__main__':
    eval_reliability_group(
        CLDATA_DIR,
        model_filename='nn_model_dropout.pth',
        scaler_filename='scaler.pkl',
        trustscore_filename='valData_trustscore.csv',
        knnconf_filename='valData_knn_conf.csv',
        name='CL'
    )
