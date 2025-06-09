import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

# 1. 训练集特征名
feature_cols = [
    'patientGender', 'patientAge', 'glasgowScale', 'hematocrit', 'hemoglobin',
    'leucocitos', 'lymphocytes', 'urea', 'creatinine', 'platelets', 'diuresis',
    'SBP', 'DBP', 'glasgowScale_missing'
]

# 2. 载入训练好的 GMM
train_path = './Data/trainData_cleaned.csv'
df_train = pd.read_csv(train_path)
X_train = df_train[feature_cols].dropna().values
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
gmm.fit(X_train)
train_densities = np.exp(gmm.score_samples(X_train))
threshold = np.quantile(train_densities, 0.3)  # 保持和训练集一致

# 3. 载入新样本（验证集）
val_path = './Data/valData_cleaned.csv'
df_val = pd.read_csv(val_path)

# 4. 只对特征列做密度评估（丢弃有缺失的行）
X_val = df_val[feature_cols].dropna().values

# 5. 计算验证集密度分数
val_densities = np.exp(gmm.score_samples(X_val))
is_high_density_val = (val_densities >= threshold).astype(int)

# 6. 保存/输出结果
# 新建密度和高密度列，注意：只在dropna()后的行有效
val_result = df_val[feature_cols].dropna().copy()
val_result['GMM_density'] = val_densities
val_result['is_high_density'] = is_high_density_val

# 你也可以把这两列合并到原始df_val里（按dropna后的index匹配即可）：
df_val.loc[val_result.index, 'GMM_density'] = val_result['GMM_density']
df_val.loc[val_result.index, 'is_high_density'] = val_result['is_high_density']

# 7. 保存结果
df_val.to_csv('./Data/valData_gmm_density.csv', index=False)
print(df_val[['GMM_density', 'is_high_density']].head(10))
