import numpy as np
import os
import re
import pandas as pd

def load_and_clean_data(data_dir='Data'):
    def _read_csv(path):
        return pd.read_csv(path)

    def _clean_dataframe(df):
        df = df.copy()
        # 日期解析
        for col in ['requestDate', 'admissionDate']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

        # 血压字段解析
        def extract_bp(bp_str):
            match = re.match(r'(\d{2,3})[xX](\d{2,3})', str(bp_str).replace(" ", ""))
            if match:
                sbp, dbp = int(match.group(1)), int(match.group(2))
                if 60 <= sbp <= 250 and 40 <= dbp <= 150:
                    return sbp, dbp
            return np.nan, np.nan

        if 'blodPressure' in df.columns:
            df['SBP'], df['DBP'] = zip(*df['blodPressure'].apply(extract_bp))
            df.drop(columns=['blodPressure'], inplace=True)

        if 'glasgowScale' in df.columns:
            df['glasgowScale'] = pd.to_numeric(df['glasgowScale'], errors='coerce')
            df['glasgowScale_missing'] = df['glasgowScale'].isna().astype(int)
            median_val = df['glasgowScale'].median(skipna=True)
            df['glasgowScale'] = df['glasgowScale'].fillna(median_val)

        # 数值字段范围限制
        for col, (low, high) in {
            'creatinine': (0, 10),
            'urea': (0, 100),
            'platelets': (1, 1000),
            'diuresis': (1, 5000),
        }.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].apply(lambda x: x if low <= x <= high else np.nan)

        # 住院时长过滤
        if 'lengthofStay' in df.columns:
            df['lengthofStay'] = pd.to_numeric(df['lengthofStay'], errors='coerce')
            df = df[(df['lengthofStay'] > 0) & (df['lengthofStay'] <= 365)]

        # outcomeType转二值
        if 'outcomeType' in df.columns:
            df['outcomeType'] = df['outcomeType'].astype(str).str.lower().map({'survival': 1, 'death': 0})

        # # 年龄异常标志
        # if 'patientAge' in df.columns:
        #     df['age_outlier'] = df['patientAge'].apply(lambda x: 1 if x < 5 or x > 90 else 0)

        # 删除无关列
        drop_cols = [
            'patientFfederalUnit', 'icdCode', 'requestDate', 'admissionDate',
            'requestType', 'requestBedType', 'admissionBedType', 'admissionHealthUnit'
        ]
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

        # 字符串列编码
        for col in df.select_dtypes(include='object').columns:
            df[col] = pd.factorize(df[col])[0]

        # 清洗结束后，调整列顺序
        cols = list(df.columns)
        for col in ['lengthofStay', 'outcomeType']:
            if col in cols:
                cols.remove(col)
                cols.append(col)
        df = df[cols]

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df



    train_path = os.path.join(data_dir, 'trainData.csv')
    val_path = os.path.join(data_dir, 'valData.csv')

    train_df = _read_csv(train_path)
    val_df = _read_csv(val_path)

    train_df = _clean_dataframe(train_df)
    val_df = _clean_dataframe(val_df)

    os.makedirs(data_dir, exist_ok=True)
    train_df.to_csv(os.path.join(data_dir, 'trainData_cleaned.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'valData_cleaned.csv'), index=False)



    return train_df, val_df

load_and_clean_data()