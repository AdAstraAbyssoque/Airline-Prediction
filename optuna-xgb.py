# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import warnings
import xgboost as xgb
import optuna

warnings.filterwarnings('ignore')

# 1. Read and merge flight data
print("Starting to read data...")
csv_files = glob.glob('flights-data/*.csv')
data_list = []
for file in csv_files:
    df = pd.read_csv(file)
    data_list.append(df)
data = pd.concat(data_list, ignore_index=True)
print(f"Original data size: {data.shape}")

# 2. Data cleaning
columns_to_keep = [
    'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier',
    'TailNum', 'FlightNum', 'OriginAirportID', 'OriginCityMarketID',
    'OriginStateFips', 'OriginWac', 'DestAirportID', 'DestCityMarketID',
    'DestStateFips', 'DestWac', 'CRSDepTime', 'CRSArrTime', 'Flights',
    'Distance', 'Status'
]
data = data[columns_to_keep]
categorical_features = ['UniqueCarrier', 'TailNum', 'Status']
label_encoders = {}

for feature in categorical_features:
    label_encoders[feature] = LabelEncoder()
    data[feature] = label_encoders[feature].fit_transform(data[feature])

# 4. Data standardization
scaler = StandardScaler()
features_to_scale = [col for col in data.columns if col != 'Status']
data_scaled = data.copy()
data_scaled[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# 5. Prepare training data
X = data_scaled.drop('Status', axis=1)
y = data_scaled['Status']

print(f"\nFinal processed data size: {data.shape}")

# Use train_test_split instead of cross-validation
train_X, valid_X, train_y, valid_y = train_test_split(X, y, 
                                                      test_size=0.2, 
                                                      random_state=3407, 
                                                      stratify=y)

def objective(trial):
    params = {
        "max_depth": 30,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators": 500,
        "objective": "multi:softmax",
        "random_state": 3407,
        "n_jobs": -1
    }
    model = xgb.XGBClassifier(**params)
    # Remove early_stopping_rounds and tqdm callback
    model.fit(train_X, train_y, eval_set=[(valid_X, valid_y)], verbose=False)
    y_pred = model.predict(valid_X)
    return f1_score(valid_y, y_pred, average="macro")

study = optuna.create_study(direction="maximize", storage="sqlite:///optuna-xgb.db", study_name="xgboost")
study.optimize(objective, n_trials=100, timeout=1200, show_progress_bar=False)

print("Best parameters:", study.best_params)
print("Best score:", study.best_value)

# Retrain and evaluate with the best parameters
best_params = study.best_params
best_clf = xgb.XGBClassifier(**best_params, n_estimators=200, objective='multi:softmax',
                             random_state=3407, n_jobs=-1)
best_clf.fit(train_X, train_y)
final_pred = best_clf.predict(valid_X)
final_f1 = f1_score(valid_y, final_pred, average='macro')
print(f"\nFinal model F1 score: {final_f1:.4f}")
print("Classification report:")
print(classification_report(valid_y, final_pred, target_names=label_encoders['Status'].classes_))