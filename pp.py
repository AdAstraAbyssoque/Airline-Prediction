# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

# 设置学生信息
student_name = 'Bowen_LIU'
student_id = '50012962'

# 1. 读取并合并航班数据
csv_files = glob.glob('flights-data/*.csv')
data_list = []
for file in csv_files:
	df = pd.read_csv(file)
	data_list.append(df)
data = pd.concat(data_list, ignore_index=True)

# 2. 数据清洗
data.dropna(inplace=True)

# 编码目标变量
status_le = LabelEncoder()
data['Status'] = status_le.fit_transform(data['Status'])

# 编码分类特征
label_encoder_dep = LabelEncoder()
data['DepTimeBlk'] = label_encoder_dep.fit_transform(data['DepTimeBlk'])

label_encoder_arr = LabelEncoder()
data['ArrTimeBlk'] = label_encoder_arr.fit_transform(data['ArrTimeBlk'])

# 3. 选择指定特征
selected_features = [
	'DayofMonth', 'DayOfWeek', 'CRSArrTime', 'CRSDepTime', 'Month', 'Quarter',
	'FlightNum', 'Distance', 'DepTimeBlk', 'ArrTimeBlk'
]
data = data[selected_features + ['Status']]

# 4. 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop(['Status'], axis=1))
X = pd.DataFrame(data_scaled, columns=selected_features)
y = data['Status']

# 5. 拆分训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(
	X, y, test_size=0.2, stratify=y, random_state=42
)

# 6. 训练模型
clf = RandomForestClassifier(
	n_estimators=454,
	max_depth=20,
	min_samples_split=5,
	min_samples_leaf=4,
	bootstrap=False,
	max_features='sqrt',
	random_state=42,
	n_jobs=-1
)
clf.fit(X_train, y_train)

# 在验证集上进行预测
y_pred = clf.predict(X_valid)

# 计算F1分数
f1 = f1_score(y_valid, y_pred, average='macro')
print(f"验证集 F1 分数: {f1:.4f}")

# 输出分类报告
print("分类报告：")
print(classification_report(y_valid, y_pred, target_names=status_le.classes_))
