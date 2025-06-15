import os
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib

# ==== 路径自动适配 ====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FLDATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'Data', 'FLData')

feature_cols = [
    'patientGender', 'patientAge', 'glasgowScale', 'hematocrit', 'hemoglobin',
    'leucocitos', 'lymphocytes', 'urea', 'creatinine', 'platelets', 'diuresis',
    'SBP', 'DBP', 'glasgowScale_missing'
]

client_csvs = [
    "2ª_MOSSORÓ.csv", "5ª_SANTA_CRUZ.csv", "7ª_METROPOLITANA.csv", "4ª_CAICÓ.csv",
    "3ª_JOÃO_CAMARA.csv", "8ª_AÇU.csv", "1ª_SÃO_JOSÉ_DE_MIPIBU.csv", "6ª_PAU_DOS_FERROS.csv"
]
all_train_X = []
gmms = []
client_ns = []

# 1. 读取所有客户端数据，收集所有特征用于拟合scaler
for csv_file in client_csvs:
    df = pd.read_csv(os.path.join(FLDATA_DIR, csv_file))
    X = df[feature_cols].dropna().values
    if len(X) == 0:
        continue
    all_train_X.append(X)

# 2. 用全量数据拟合scaler，并保存
all_train_X_concat = np.concatenate(all_train_X, axis=0)
scaler = StandardScaler()
scaler.fit(all_train_X_concat)
joblib.dump(scaler, os.path.join(FLDATA_DIR, "scaler.pkl"))
print("全局scaler已保存到:", os.path.join(FLDATA_DIR, "scaler.pkl"))

# 3. 分别对各客户端标准化后建GMM
for X in all_train_X:
    X_scaled = scaler.transform(X)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm.fit(X_scaled)
    gmms.append(gmm)
    client_ns.append(X.shape[0])
print(f"聚合 {len(gmms)} 个客户端 GMM")

# 4. FedAvg 聚合参数
n_components = 2
weights = np.array([gmm.weights_ * n for gmm, n in zip(gmms, client_ns)])
means = np.array([gmm.means_ * (gmm.weights_ * n)[:, None] for gmm, n in zip(gmms, client_ns)])
covariances = np.array([gmm.covariances_ * (gmm.weights_ * n)[:, None, None] for gmm, n in zip(gmms, client_ns)])

total_samples = np.sum(client_ns)
agg_weights = np.sum(weights, axis=0) / total_samples
agg_means = np.sum(means, axis=0) / np.sum(weights, axis=0)[:, None]
agg_covariances = np.sum(covariances, axis=0) / np.sum(weights, axis=0)[:, None, None]

agg_gmm = GaussianMixture(n_components=n_components, covariance_type='full')
agg_gmm.weights_ = agg_weights
agg_gmm.means_ = agg_means
agg_gmm.covariances_ = agg_covariances
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
agg_gmm.precisions_cholesky_ = _compute_precision_cholesky(agg_covariances, 'full')
joblib.dump(agg_gmm, os.path.join(FLDATA_DIR, 'fed_gmm_model.pkl'))
print(f"全局GMM模型已保存到: {os.path.join(FLDATA_DIR, 'fed_gmm_model.pkl')}")

# 5. 验证集：标准化、密度评分、保存
val_path = os.path.join(FLDATA_DIR, 'valData_cleaned.csv')
df_val = pd.read_csv(val_path)
X_val = df_val[feature_cols].dropna().values
X_val_scaled = scaler.transform(X_val)
val_densities = np.exp(agg_gmm.score_samples(X_val_scaled))
threshold = np.quantile(val_densities, 0.3)
is_high_density_val = (val_densities >= threshold).astype(int)

val_result = df_val[feature_cols].dropna().copy()
val_result['GMM_density'] = val_densities
val_result['is_high_density'] = is_high_density_val
df_val.loc[val_result.index, 'GMM_density'] = val_result['GMM_density']
df_val.loc[val_result.index, 'is_high_density'] = val_result['is_high_density']
save_path = os.path.join(FLDATA_DIR, 'valData_gmm_density.csv')
df_val.to_csv(save_path, index=False)
print(df_val[['GMM_density', 'is_high_density']].head(10))
