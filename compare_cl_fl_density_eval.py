import os
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
# 路径与特征
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'Data')
CLDATA_DIR = os.path.join(DATA_DIR, 'CLData')
FLDATA_DIR = os.path.join(DATA_DIR, 'FLData')

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

def eval_one_setting(DATA_DIR, model_filename, scaler_filename, density_filename, name='CL'):
    # 加载数据
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'valData_cleaned.csv'))
    density_df = pd.read_csv(os.path.join(DATA_DIR, density_filename))

    # 标准化
    scaler_path = os.path.join(DATA_DIR, scaler_filename)
    scaler = joblib.load(scaler_path)
    X_val = scaler.transform(val_df[feature_cols].dropna().values)
    y_val = val_df[target_col].dropna().values.astype(int)

    # 加载模型
    model = DropoutNN(len(feature_cols))
    model_path = os.path.join(DATA_DIR, model_filename)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 预测
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_pred_prob = model(X_val_tensor).cpu().numpy()
        y_pred_label = (y_pred_prob > 0.5).astype(int)

    # 对齐索引长度
    N = min(len(density_df), len(y_pred_label), len(y_val))
    density_df = density_df.iloc[:N].reset_index(drop=True)
    y_pred_label = y_pred_label[:N]
    y_val = y_val[:N]
    y_pred_prob = y_pred_prob[:N]

    # 合并结果
    density_df['nn_pred'] = y_pred_label
    density_df['nn_true'] = y_val
    density_df['density_group'] = pd.qcut(density_df['GMM_density'], q=3, labels=['低', '中', '高'])

    print(f"\n============== {name} 结果 ==============")
    for group in ['低', '中', '高']:
        mask = density_df['density_group'] == group
        acc = accuracy_score(density_df.loc[mask, 'nn_true'], density_df.loc[mask, 'nn_pred'])
        print(f"\n{name} {group}密度区准确率: {acc:.3f}")
        print(f"{name} {group}密度区分类报告:")
        print(classification_report(
            density_df.loc[mask, 'nn_true'],
            density_df.loc[mask, 'nn_pred'],
            digits=3
        ))

    # 整体评估
    print(f"\n{name} 整体评估：")
    print(classification_report(y_val, y_pred_label, digits=3))

    # ROC曲线和AUC
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # PR曲线和AP
    precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
    ap = average_precision_score(y_val, y_pred_prob)

    # 绘图，不保存，仅在科学模式/Plots窗口显示
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    axs[0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
    axs[0].plot([0, 1], [0, 1], 'k--', label='Random')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title(f'ROC Curve - {name}')
    axs[0].legend(loc='lower right')

    # PR
    axs[1].plot(recall, precision, label=f'{name} (AP={ap:.3f})')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].set_title(f'Precision-Recall Curve - {name}')
    axs[1].legend(loc='best')

    plt.tight_layout()
    plt.show()  # 在PyCharm科学模式下图像自动在侧边栏显示

if __name__ == '__main__':
    # 评估 CL
    eval_one_setting(
        CLDATA_DIR,
        model_filename='nn_model_dropout.pth',
        scaler_filename='scaler.pkl',
        density_filename='valData_gmm_density.csv',
        name='CL'
    )
    # 评估 FL
    eval_one_setting(
        FLDATA_DIR,
        model_filename='fedavg_nn_model_final.pth',
        scaler_filename='scaler.pkl',
        density_filename='valData_gmm_density.csv',
        name='FL'
    )


import matplotlib.pyplot as plt
import numpy as np

labels = ['lowdensity', 'middensity', 'highdensity']
CL_acc = [0.690, 0.742, 0.777]
FL_acc = [0.737, 0.790, 0.819]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, CL_acc, width, label='CL')
rects2 = ax.bar(x + width/2, FL_acc, width, label='FL')

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy in Different Density Groups')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

CL_overall = 0.736
FL_overall = 0.782
plt.bar(['CL', 'FL'], [CL_overall, FL_overall])
plt.ylabel('Overall Accuracy')
plt.title('Overall Accuracy Comparison')
plt.ylim(0, 1)
plt.show()

# 假设每组数据已手动填写（如上表格中直接复制）
CL_prec = [0.521, 0.377, 0.215]
FL_prec = [0.603, 0.425, 0.244]
labels = ['lowdensity', 'middensity', 'highdensity']
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, CL_prec, width, label='CL')
plt.bar(x + width/2, FL_prec, width, label='FL')
plt.ylabel('Precision')
plt.title('Precision in Different Density Groups')
plt.xticks(x, labels)
plt.legend()
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

plt.plot(labels, CL_acc, marker='o', label='CL')
plt.plot(labels, FL_acc, marker='o', label='FL')
plt.ylabel('Accuracy')
plt.title('Accuracy by Density Group')
plt.legend()
plt.ylim(0, 1)
plt.show()
