# 1. 导入必要的库
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score
import optuna
from optuna.samplers import TPESampler
import multiprocessing

student_id = "50012962"
student_name = "Bowen_LIU"

# 2. 读取并合并数据
csv_files = glob.glob('flights-data/*.csv')
data_list = []
for file in tqdm(csv_files, desc="读取CSV文件"):
    df = pd.read_csv(file)
    data_list.append(df)
data = pd.concat(data_list, ignore_index=True)
data.dropna(inplace=True)

# 3. 编码分类变量
categorical_features = ['UniqueCarrier', 'Origin', 'Dest', 'DepTimeBlk', 'ArrTimeBlk', 'Status']
label_encoders = {}
for col in tqdm(categorical_features, desc="编码分类变量"):
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 4. 处理时间特征
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

# 5. 选择特征和目标变量
features = ['Month', 'DayofMonth', 'DayOfWeek',
            'CRSDepTime', 'CRSArrTime', 'UniqueCarrier', 'Origin', 'Dest',
            'DepTimeBlk', 'ArrTimeBlk', 'Distance']
X = data[features]
y = data['Status']

# 6. 拆分训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 7. 定义Optuna的目标函数
def objective(trial):
    param = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'max_depth': trial.suggest_int('max_depth', 5, 50, step=5),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    accuracies = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            **param,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        
        f1 = f1_score(y_val, y_pred, average='macro')
        acc = accuracy_score(y_val, y_pred)
        
        f1_scores.append(f1)
        accuracies.append(acc)
    
    trial.set_user_attr("accuracy", np.mean(accuracies))
    return np.mean(f1_scores)

# 8. 创建Optuna的study并优化
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='lightgbm_f1_optimization',
    storage='sqlite:///optuna_lgbm_study.db',
    load_if_exists=True
)

if __name__ == '__main__':
    n_trials = 100
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

# 9. 输出最佳参数
print("最佳参数：", study.best_params)

# 10. 使用最佳参数训练最终模型
best_params = study.best_params
best_params['objective'] = 'multiclass'
best_params['num_class'] = len(np.unique(y_train))
best_params['random_state'] = 42
best_params['n_jobs'] = -1

model = lgb.LGBMClassifier(**best_params)
model.fit(X_train, y_train)

# 11. 在验证集上进行预测
y_pred = model.predict(X_valid)

# 12. 计算F1分数和准确率
f1 = f1_score(y_valid, y_pred, average='macro')
acc = accuracy_score(y_valid, y_pred)
print(f"验证集 F1 分数: {f1:.4f}")
print(f"验证集 准确率: {acc:.4f}")

# 13. 使用optuna-dashboard可视化优化过程
# 在命令行中运行以下命令启动dashboard（需提前安装optuna-dashboard库）
# optuna-dashboard sqlite:///optuna_lgbm_study.db

# 14. 处理测试数据并进行预测
print("开始处理测试数据并进行预测...")
test_data = pd.read_csv('test_data.csv')

# 编码测试数据的分类变量
for col in categorical_features:
    if col != 'Status':  # 测试数据中没有 'Status' 列
        le = label_encoders[col]
        test_data[col] = le.transform(test_data[col])

# 处理时间特征
test_data['CRSDepTime'] = test_data['CRSDepTime'].apply(convert_time)
test_data['CRSArrTime'] = test_data['CRSArrTime'].apply(convert_time)

# 提取特征
X_test = test_data[features]

# 对测试数据进行预测
print("对测试数据进行预测...")
y_test_pred = model.predict(X_test)

# 将预测结果转换回原始标签并添加为新列
test_data['Status'] = label_encoders['Status'].inverse_transform(y_test_pred)

# 保存预测结果
test_data[['Status']].to_csv(f"{student_id}_{student_name}.csv", index=False)
print(f"预测结果已保存至 {student_id}_{student_name}.csv")