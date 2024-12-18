# Python
# 使用GRU优化代码以提高F1分数，并将准确率保存到数据库
# 实现ResNet、BatchNorm、Dropout、PReLU、Nadam优化器、学习率衰减和早停

import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch_optimizer as optim_  # pip install torch_optimizer
from torch.utils.data import Dataset, DataLoader
from sqlalchemy import create_engine
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 数据读取与预处理
csv_files = glob.glob('flights-data/*.csv')
data = pd.concat((pd.read_csv(file) for file in tqdm(csv_files, desc="读取CSV文件")), ignore_index=True)
print(f"数据合并完成，共{data.shape[0]}行，{data.shape[1]}列。")

data.dropna(inplace=True)
print(f"缺失值处理完成，剩余{data.shape[0]}行。")

categorical_features = ['UniqueCarrier', 'Origin', 'Dest', 'DepTimeBlk', 'ArrTimeBlk']
label_encoders = {col: LabelEncoder().fit(data[col]) for col in categorical_features}
for col in categorical_features:
    data[col] = label_encoders[col].transform(data[col])

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

features = ['Month', 'DayofMonth', 'DayOfWeek',
            'CRSDepTime', 'CRSArrTime', 'UniqueCarrier', 'Origin', 'Dest',
            'DepTimeBlk', 'ArrTimeBlk', 'Distance']
X = data[features].values
y = data['Status'].values

status_le = LabelEncoder().fit(y)
y = status_le.transform(y)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"训练集大小：{X_train.shape[0]}，验证集大小：{X_valid.shape[0]}")

# 自定义Dataset
class FlightDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = FlightDataset(X_train, y_train)
valid_dataset = FlightDataset(X_valid, y_valid)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.batch_norm1(out)
        out = self.prelu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.prelu2(out)
        out = self.dropout2(out)
        
        out += residual
        return out

# 定义带有残差连接和BatchNorm的GRU模型
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate, num_residual_blocks=2, temperature=1.0):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.temperature = temperature  # 温度参数
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Residual Blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(hidden_size, dropout_rate) for _ in range(num_residual_blocks)
        ])
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # 转换为 (batch_size, 1, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)  # out: (batch_size, seq_length=1, hidden_size)
        out = out[:, -1, :]  # (batch_size, hidden_size)
        out = self.batch_norm(out)
        out = self.prelu(out)
        out = self.dropout(out)
        
        out = self.residual_blocks(out)  # Residual Blocks
        
        out = self.fc(out)
        out = out / self.temperature  # 温度缩放
        return out  # CrossEntropyLoss 包含 Softmax

input_size = X.shape[1]
hidden_size = 128
num_layers = 2
num_classes = len(status_le.classes_)
dropout_rate = 0.5
num_residual_blocks = 2
temperature = 1.0  # 可以根据需要调整

model = GRUClassifier(input_size, hidden_size, num_layers, num_classes, dropout_rate, num_residual_blocks, temperature)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 使用标签平滑
optimizer = torch.optim.NAdam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # Nadam 优化器
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)  # 学习率衰减

# 训练与验证
num_epochs = 100
best_f1 = 0
patience = 10
counter = 0

engine = create_engine('sqlite:///gru.db')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + len(train_loader) * batch_size)
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    model.eval()
    valid_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            valid_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_valid_loss = valid_loss / len(valid_loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), 'best_gru_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("早停。")
            break
    
    # 保存准确率到数据库
    acc_df = pd.DataFrame({'epoch': [epoch+1], 'accuracy': [acc], 'f1_score': [f1]})
    acc_df.to_sql('accuracy_scores', engine, if_exists='append', index=False)

# 加载最佳模型
model.load_state_dict(torch.load('best_gru_model.pth'))

# 在验证集上进行预测
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in valid_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

f1 = f1_score(all_labels, all_preds, average='macro')
acc = accuracy_score(all_labels, all_preds)
print(f"验证集 F1 分数: {f1:.4f}")
print(f"验证集 准确率: {acc:.4f}")

# 输出分类报告
print("分类报告：")
print(classification_report(all_labels, all_preds, target_names=status_le.classes_))

# 特征重要性分析（基于权重）
feature_importance = pd.Series(model.fc.weight.data.cpu().numpy().mean(axis=0), index=features).sort_values(ascending=False)
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
cm = confusion_matrix(all_labels, all_preds)
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

X_test = test_data[features].values
test_dataset = FlightDataset(X_test, np.zeros(X_test.shape[0]))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
y_test_pred = []
with torch.no_grad():
    for X_batch, _ in tqdm(test_loader, desc="预测测试数据"):
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_test_pred.extend(preds)

test_data['Status'] = status_le.inverse_transform(y_test_pred)
test_data.to_csv('prediction_results_gru_optuna.csv', index=False)
print("预测结果已保存至 prediction_results_gru_optuna.csv")