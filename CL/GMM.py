import os
import pandas as pd
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ==== 路径自动适配 ====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../Data/CLData'))

# 可选：调试路径
print("数据目录：", DATA_DIR)

# 1. 训练集特征名
feature_cols = [
    'patientGender', 'patientAge', 'glasgowScale', 'hematocrit', 'hemoglobin',
    'leucocitos', 'lymphocytes', 'urea', 'creatinine', 'platelets', 'diuresis',
    'SBP', 'DBP', 'glasgowScale_missing'
]

# ===== 是否第一次训练并保存模型 =====
train_path = os.path.join(DATA_DIR, 'trainData_cleaned.csv')
gmm_path = os.path.join(DATA_DIR, 'gmm_model.pkl')
scaler_path = os.path.join(DATA_DIR, 'scaler.pkl')

# ----------- 只需运行一次的模型训练和保存 -----------
if not os.path.exists(gmm_path) or not os.path.exists(scaler_path):
    print("首次拟合密度模型与标准化器，并保存...")
    df_train = pd.read_csv(train_path)
    X_train = df_train[feature_cols].dropna().values

    # 先做标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # 保存标准化器
    joblib.dump(scaler, scaler_path)

    # GMM建模
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm.fit(X_train_scaled)
    joblib.dump(gmm, gmm_path)
else:
    print("检测到已有模型文件，跳过训练。")

# ----------- 加载密度模型与标准化器 -----------
scaler = joblib.load(scaler_path)
gmm = joblib.load(gmm_path)

# ----------- 密度评估并保存结果 -----------
val_path = os.path.join(DATA_DIR, 'valData_cleaned.csv')
df_val = pd.read_csv(val_path)

X_val = df_val[feature_cols].dropna().values
X_val_scaled = scaler.transform(X_val)

# 使用加载好的GMM进行密度评估
val_densities = np.exp(gmm.score_samples(X_val_scaled))

# 用训练集密度阈值判定高低密度
# （注意：如果首次拟合，下面代码要先拟合阈值并保存，可扩展为独立保存。这里直接用训练集再跑一遍阈值流程也可以）
df_train = pd.read_csv(train_path)
X_train = df_train[feature_cols].dropna().values
X_train_scaled = scaler.transform(X_train)
train_densities = np.exp(gmm.score_samples(X_train_scaled))
threshold = np.quantile(train_densities, 0.3)

is_high_density_val = (val_densities >= threshold).astype(int)

# 新建密度和高密度列，只在dropna()后的行有效
val_result = df_val[feature_cols].dropna().copy()
val_result['GMM_density'] = val_densities
val_result['is_high_density'] = is_high_density_val

# 合并到原始df_val里（按dropna后的index匹配）
df_val.loc[val_result.index, 'GMM_density'] = val_result['GMM_density']
df_val.loc[val_result.index, 'is_high_density'] = val_result['is_high_density']

# 保存结果
val_save_path = os.path.join(DATA_DIR, 'valData_gmm_density.csv')
df_val.to_csv(val_save_path, index=False)
print(df_val[['GMM_density', 'is_high_density']].head(10))