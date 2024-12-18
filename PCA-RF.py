# 导入必要的库
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler

# 设置学生信息
student_name = 'Bowen_LIU'
student_id = '50012962'

# 1. 读取并合并2016-2017年的航班数据
csv_files = glob.glob('flights-data/*.csv')
data_list = []
print("开始读取CSV文件...")
for file in tqdm(csv_files, desc="读取CSV文件"):
    df = pd.read_csv(file)
    data_list.append(df)
data = pd.concat(data_list, ignore_index=True)
print(f"数据合并完成，共{data.shape[0]}行，{data.shape[1]}列。")

# 2. 数据清洗和预处理
print("开始处理缺失值...")
initial_rows = data.shape[0]
data.dropna(inplace=True)
removed_rows = initial_rows - data.shape[0]
print(f"缺失值处理完成，删除了{removed_rows}行，剩余{data.shape[0]}行。")

# 检查目标变量的分布
print("Status列的唯一值及其计数：")
print(data['Status'].value_counts())

# 将Status列编码为数字
status_le = LabelEncoder()
data['Status'] = status_le.fit_transform(data['Status'])

# 3. 编码分类变量
print("开始编码分类变量...")
# 添加所有需要编码的分类特征
categorical_features = ['UniqueCarrier', 'Carrier', 'Origin', 'Dest', 'OriginState', 'DestState', 'DepTimeBlk', 'ArrTimeBlk']
# 删除高基数或不必要的特征
data.drop(['TailNum', 'OriginCityName', 'OriginStateName', 'DestCityName', 'DestStateName'], axis=1, inplace=True, errors='ignore')

label_encoders = {}
for col in tqdm(categorical_features, desc="编码分类变量"):
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 4. 处理时间特征
# CRSDepTime和CRSArrTime转换为分钟数
def convert_time(x):
    try:
        x = int(x)
    except:
        x = 0
    if x == 2400:
        x = 0
    hours = x // 100
    minutes = x % 100
    return hours * 60 + minutes

data['CRSDepTime'] = data['CRSDepTime'].apply(convert_time)
data['CRSArrTime'] = data['CRSArrTime'].apply(convert_time)

# 5. 特征选择

# 5.1 删除FlightDate列，因为相关时间特征已提取
if 'FlightDate' in data.columns:
    data.drop('FlightDate', axis=1, inplace=True)

# 5.2 计算相关系数矩阵
corr_matrix = data.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('特征相关性热力图')
plt.savefig('correlation_heatmap.png')

# 5.3 基于特征重要性进行初步筛选
print("计算初始特征重要性...")
initial_rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
initial_rfc.fit(data.drop(['Status'], axis=1), data['Status'])

feature_importances = pd.Series(initial_rfc.feature_importances_, index=data.drop(['Status'], axis=1).columns)
feature_importances.sort_values(ascending=False, inplace=True)
print("初始特征重要性：")
print(feature_importances)

# 可视化特征重要性
plt.figure(figsize=(12,8))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
plt.title('初始特征重要性')
plt.xlabel('重要性分数')
plt.ylabel('特征')
plt.tight_layout()
plt.savefig('initial_feature_importance.png')

# 选择重要性排名前 15 的特征（可根据情况调整）
selected_features = feature_importances.index[:15].tolist()
print(f"选择的特征：{selected_features}")

# 5.4 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[selected_features])

# 5.5 使用PCA进行降维（保留 95% 以上的方差）
pca = PCA(n_components=0.95, random_state=42)
principal_components = pca.fit_transform(data_scaled)
print(f"PCA后特征数量：{principal_components.shape[1]}")

# 将主成分转为DataFrame
pca_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
data_pca = pd.DataFrame(principal_components, columns=pca_columns)

# 将目标变量加入
data_pca['Status'] = data['Status'].values

# 6. 拆分训练集和验证集
print("拆分训练集和验证集...")
X = data_pca.drop(['Status'], axis=1)
y = data_pca['Status']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"训练集大小：{X_train.shape[0]}，验证集大小：{X_valid.shape[0]}")

# 7. 定义Optuna的目标函数
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 5, 50, step=5),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample'])
    }
    
    clf = RandomForestClassifier(
        **param,
        random_state=42,
        n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    accuracies = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        
        f1 = f1_score(y_val, y_pred, average='macro')
        acc = accuracy_score(y_val, y_pred)
        
        f1_scores.append(f1)
        accuracies.append(acc)
    
    trial.set_user_attr("accuracy", np.mean(accuracies))
    return np.mean(f1_scores)

# 8. 创建Optuna的study并优化，不更改数据库
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='random_forest_f1_optimization_pca',
    storage='sqlite:///optuna_rf_new_study.db',
    load_if_exists=True
)
study.optimize(objective, n_trials=100, show_progress_bar=True)

# 9. 输出最佳参数和对应的F1分数及准确率
print("最佳参数：", study.best_params)
print("最佳F1分数：", study.best_value)
print("对应的准确率：", study.best_trial.user_attrs["accuracy"])

# 10. 使用最佳参数训练最终模型
best_params = study.best_params
best_params['random_state'] = 42
best_params['n_jobs'] = -1

best_rfc = RandomForestClassifier(**best_params)
best_rfc.fit(X_train, y_train)

# 在验证集上进行预测
y_pred = best_rfc.predict(X_valid)

# 计算F1分数和准确率
f1 = f1_score(y_valid, y_pred, average='macro')
acc = accuracy_score(y_valid, y_pred)
print(f"验证集 F1 分数: {f1:.4f}")
print(f"验证集 准确率: {acc:.4f}")

# 输出分类报告
print("分类报告：")
print(classification_report(y_valid, y_pred, target_names=status_le.classes_))

# 11. 绘制特征重要性分析并可视化
importances = best_rfc.feature_importances_
feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("特征重要性：")
print(feature_importance)

# 可视化特征重要性
plt.figure(figsize=(12,8))
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette='viridis')
plt.title('特征重要性')
plt.xlabel('重要性分数')
plt.ylabel('主成分')
plt.tight_layout()
plt.savefig('final_feature_importance.png')

# 12. 绘制混淆矩阵
cm = confusion_matrix(y_valid, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=status_le.classes_,
            yticklabels=status_le.classes_)
plt.ylabel('实际')
plt.xlabel('预测')
plt.title('混淆矩阵')
plt.savefig('confusion_matrix.png')

# 13. 对测试数据进行同样的预处理，并进行预测
print("开始处理测试数据并进行预测...")
test_data = pd.read_csv('test_data.csv')
print(f"测试数据共{test_data.shape[0]}行，{test_data.shape[1]}列。")
test_data.dropna(inplace=True)
print(f"处理缺失值后，测试数据剩余{test_data.shape[0]}行。")

# 编码测试数据的分类变量
for col in categorical_features:
    le = label_encoders[col]
    # 处理未见过的类别
    test_data[col] = test_data[col].apply(lambda x: x if x in le.classes_ else '<unknown>')
    if '<unknown>' not in le.classes_:
        le.classes_ = np.append(le.classes_, '<unknown>')
    test_data[col] = le.transform(test_data[col])

# 处理时间特征
test_data['CRSDepTime'] = test_data['CRSDepTime'].apply(convert_time)
test_data['CRSArrTime'] = test_data['CRSArrTime'].apply(convert_time)

# 选择与训练集相同的特征
test_selected_features = test_data[selected_features]

# 数据标准化
test_scaled = scaler.transform(test_selected_features)

# PCA转换
test_pca = pca.transform(test_scaled)

# 转换为DataFrame
test_pca_df = pd.DataFrame(test_pca, columns=pca_columns)

# 对测试数据进行预测
print("对测试数据进行预测...")
y_test_pred = best_rfc.predict(test_pca_df)

# 将预测结果转换回原始标签
test_data['Status'] = status_le.inverse_transform(y_test_pred)

# 将预测结果保存为规定格式的csv文件
test_data[['Status']].to_csv('prediction_results.csv', index=False)
print("预测结果已保存至 prediction_results.csv")

# 14. 可视化Optuna的F1分数和准确率
print("开始可视化Optuna的F1分数和准确率...")
# 获取所有试验的F1分数和准确率
f1_scores = [trial.value for trial in study.trials if trial.value is not None]
accuracies = [trial.user_attrs["accuracy"] for trial in study.trials if "accuracy" in trial.user_attrs]

# 绘制F1分数分布
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
sns.histplot(f1_scores, bins=20, kde=True, color='skyblue')
plt.title('Optuna试验的F1分数分布')
plt.xlabel('F1分数')
plt.ylabel('频数')

# 绘制准确率分布
plt.subplot(1, 2, 2)
sns.histplot(accuracies, bins=20, kde=True, color='salmon')
plt.title('Optuna试验的准确率分布')
plt.xlabel('准确率')
plt.ylabel('频数')

plt.tight_layout()
plt.savefig('optuna_scores_distribution.png')

# 15. F1分数与准确率的关系图
plt.figure(figsize=(8,6))
sns.scatterplot(x=f1_scores, y=accuracies, alpha=0.6)
plt.title('Optuna试验的F1分数与准确率关系图')
plt.xlabel('F1分数')
plt.ylabel('准确率')
plt.grid(True)
plt.savefig('optuna_f1_vs_accuracy.png')