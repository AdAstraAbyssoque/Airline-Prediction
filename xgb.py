# Python
# 优化代码以加快运行速度，提高F1分数，并将准确率保存到数据库

import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from optuna.integration import XGBoostPruningCallback
from sqlalchemy import create_engine

# 自定义TqdmCallback
from xgboost.callback import TrainingCallback

class TqdmCallback(TrainingCallback):
    def __init__(self, pbar):
        self.pbar = pbar

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False  # 继续训练

# 加速数据读取
csv_files = glob.glob('flights-data/*.csv')
data = pd.concat((pd.read_csv(file) for file in tqdm(csv_files, desc="读取CSV文件")), ignore_index=True)
print(f"数据合并完成，共{data.shape[0]}行，{data.shape[1]}列。")

# 数据清洗和预处理
data.dropna(inplace=True)
print(f"缺失值处理完成，剩余{data.shape[0]}行。")

# 编码分类变量
categorical_features = ['UniqueCarrier', 'Origin', 'Dest', 'DepTimeBlk', 'ArrTimeBlk']
label_encoders = {col: LabelEncoder().fit(data[col]) for col in categorical_features}
for col in categorical_features:
    data[col] = label_encoders[col].transform(data[col])

# 处理时间特征
def convert_time(x):
    try:
        x = int(x)
    except:
        x = 0
    if x == 2400:
        x = 0
    return (x // 100) * 60 + (x % 100)

data['CRSDepTime'] = data['CRSDepTime'].apply(convert_time)
data['CRSArrTime'] = data['CRSArrTime'].apply(convert_time)

# 选择特征和目标变量
features = ['Month', 'DayofMonth', 'DayOfWeek',
            'CRSDepTime', 'CRSArrTime', 'UniqueCarrier', 'Origin', 'Dest',
            'DepTimeBlk', 'ArrTimeBlk', 'Distance']
X = data[features]
y = data['Status']

# 编码目标变量
status_le = LabelEncoder().fit(y)
y = status_le.transform(y)

# 拆分训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 定义Optuna的目标函数
def objective(trial):
    param = {
        'objective': 'multi:softprob',
        'num_class': len(status_le.classes_),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'verbosity': 0,
        'random_state': 42,
        'n_jobs': -1,
        'max_depth': trial.suggest_int('max_depth', 3, 20, step=1),  # 扩大搜索范围
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),  # 使用suggest_float
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100),  # 增加上限
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # 使用suggest_float
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # 使用suggest_float
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),  # 使用suggest_float
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),  # 使用suggest_float
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True)  # 使用suggest_float
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    
    evals_result = {}
    num_boost_round = 2000
    with tqdm(total=num_boost_round, desc='XGBoost Training') as pbar:
        callbacks = [
            XGBoostPruningCallback(trial, 'validation-mlogloss'),
            TqdmCallback(pbar)
        ]
        bst = xgb.train(
            params=param,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dvalid, 'validation')],
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=False,
            callbacks=callbacks
        )
    preds = bst.predict(dvalid)
    y_pred = np.argmax(preds, axis=1)
    f1 = f1_score(y_valid, y_pred, average='macro')
    return f1

# 创建Optuna的study并优化
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    storage='sqlite:///xgb.db',
    study_name='xgb_optimization',
    load_if_exists=True
)
study.optimize(objective, n_trials=200, show_progress_bar=True)  # 增加 n_trials

# 输出最佳参数和对应的F1分数
print("最佳参数：", study.best_params)
print("最佳F1分数：", study.best_value)

# 使用最佳参数训练XGBoost模型
best_params = study.best_params
best_params.update({
    'objective': 'multi:softprob',
    'num_class': len(status_le.classes_),
    'eval_metric': 'mlogloss',
    'use_label_encoder': False,
    'verbosity': 0,
    'random_state': 42,
    'n_jobs': -1
})

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# 训练模型时使用 tqdm 进度条
evals_result = {}
num_boost_round = best_params.get('n_estimators', 2000)

with tqdm(total=num_boost_round, desc='Final Model Training') as pbar:
    callbacks = [
        TqdmCallback(pbar)
    ]
    bst = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dvalid, 'validation')],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=False,
        callbacks=callbacks
    )

# 在验证集上进行预测
preds = bst.predict(dvalid)
y_pred = np.argmax(preds, axis=1)

# 计算F1分数和准确率
f1 = f1_score(y_valid, y_pred, average='macro')
acc = accuracy_score(y_valid, y_pred)
print(f"验证集 F1 分数: {f1:.4f}")
print(f"验证集 准确率: {acc:.4f}")

# 保存准确率到数据库
engine = create_engine('sqlite:///xgb.db')
acc_df = pd.DataFrame({'accuracy': [acc]})
acc_df.to_sql('accuracy_scores', engine, if_exists='append', index=False)

# 输出分类报告
print("分类报告：")
print(classification_report(y_valid, y_pred, target_names=status_le.classes_))

# 特征重要性分析并可视化
feature_importance = pd.Series(bst.get_score(importance_type='weight')).sort_values(ascending=False)
print("特征重要性：")
print(feature_importance)

plt.figure(figsize=(12,8))
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette='viridis')
plt.title('特征重要性')
plt.xlabel('重要性分数')
plt.ylabel('特征')
plt.tight_layout()
plt.show()

# 绘制混淆矩阵
cm = confusion_matrix(y_valid, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=status_le.classes_,
            yticklabels=status_le.classes_)
plt.ylabel('实际')
plt.xlabel('预测')
plt.title('混淆矩阵')
plt.show()

# 处理测试数据并进行预测
test_data = pd.read_csv('test_data.csv')
for col in categorical_features:
    test_data[col] = label_encoders[col].transform(test_data[col])
test_data['CRSDepTime'] = test_data['CRSDepTime'].apply(convert_time)
test_data['CRSArrTime'] = test_data['CRSArrTime'].apply(convert_time)

X_test = test_data[features]
dtest = xgb.DMatrix(X_test)

y_test_pred = bst.predict(dtest)
y_test_pred = np.argmax(y_test_pred, axis=1)
test_data['Status'] = status_le.inverse_transform(y_test_pred)
test_data.to_csv('prediction_results_xgb_optuna.csv', index=False)
print("预测结果已保存至 prediction_results_xgb_optuna.csv")