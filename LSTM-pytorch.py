import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 加载数据
data = pd.read_csv('measures_v2.csv')

# 2. 定义输入特征和目标变量
features = ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'torque', 'ambient']
targets = ['stator_winding', 'stator_tooth', 'pm', 'stator_yoke']

# 3. 定义要排除的profile_id
exclude_profiles_train = [2, 3, 4]  # 不参与训练的profile
test_profiles = [5, 6, 7]  # 专门用于测试的profile


# 4. 数据预处理函数
def create_sequences(group, time_steps=10):
    X_group = group[features].values
    y_group = group[targets].values
    X_seq, y_seq = [], []
    for i in range(len(X_group) - time_steps):
        X_seq.append(X_group[i:i + time_steps])
        y_seq.append(y_group[i + time_steps])
    return np.array(X_seq), np.array(y_seq)


# 5. 按profile_id划分数据集
def prepare_datasets(data, exclude=[], test_only=[]):
    X_train, y_train = np.empty((0, 10, len(features))), np.empty((0, len(targets)))
    X_test, y_test = np.empty((0, 10, len(features))), np.empty((0, len(targets)))

    for profile_id, group in data.groupby('profile_id'):
        X_seq, y_seq = create_sequences(group)
        if len(X_seq) == 0:
            continue

        if profile_id in test_only:  # 测试专用数据
            X_test = np.concatenate((X_test, X_seq))
            y_test = np.concatenate((y_test, y_seq))
        elif profile_id not in exclude:  # 训练数据
            X_train = np.concatenate((X_train, X_seq))
            y_train = np.concatenate((y_train, y_seq))

    return X_train, y_train, X_test, y_test


# 准备数据集
X_train, y_train, X_test, y_test = prepare_datasets(
    data, exclude=exclude_profiles_train, test_only=test_profiles
)

print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

# 6. 数据标准化
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# 重塑用于缩放
X_train_reshaped = X_train.reshape(-1, len(features))
X_train_scaled = feature_scaler.fit_transform(X_train_reshaped)
X_train = X_train_scaled.reshape(X_train.shape)

y_train = target_scaler.fit_transform(y_train)

# 测试集使用训练集的scaler转换
X_test_reshaped = X_test.reshape(-1, len(features))
X_test_scaled = feature_scaler.transform(X_test_reshaped)
X_test = X_test_scaled.reshape(X_test.shape)

y_test_scaled = target_scaler.transform(y_test)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)

# 创建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 7. 构建PyTorch LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 只取最后一个时间步的输出
        out = out[:, -1, :]

        # 全连接层
        out = self.bn1(out)
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)

        return out


# 初始化模型
input_size = len(features)
hidden_size = 128
num_layers = 2
output_size = len(targets)

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
print(model)

# 8. 训练配置
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# 早停机制
early_stopping_patience = 15
best_loss = float('inf')
patience_counter = 0


# 训练函数
def train_model(model, train_loader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

        for batch_X, batch_y in progress_bar:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

        # 早停检查
        global best_loss, patience_counter
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # 学习率调整
        scheduler.step(avg_loss)


# 开始训练
train_model(model, train_loader, criterion, optimizer, epochs=100)

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))


# 9. 评估模型
def evaluate_model(model, test_loader, target_scaler):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            # 收集预测和真实值
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    # 合并所有batch的结果
    y_pred_scaled = np.concatenate(all_preds)
    y_true_scaled = np.concatenate(all_targets)

    # 反标准化
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_true = target_scaler.inverse_transform(y_true_scaled)

    # 计算指标
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))

    print(f"\n测试集 MSE (原始尺度): {mse:.4f}")
    print(f"测试集 MAE (原始尺度): {mae:.4f}")

    return y_true, y_pred


# 执行评估
y_true, y_pred = evaluate_model(model, test_loader, target_scaler)


# 10. 可视化结果
def plot_results(y_true, y_pred, targets, n_samples=100):
    plt.figure(figsize=(15, 10))
    for i, target in enumerate(targets):
        plt.subplot(2, 2, i + 1)
        plt.plot(y_true[:n_samples, i], label='True', alpha=0.7)
        plt.plot(y_pred[:n_samples, i], label='Predicted', alpha=0.7)
        plt.title(f'{target} Prediction')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_results(y_true, y_pred, targets)

# 11. 保存模型和scaler
torch.save(model.state_dict(), 'pytorch_lstm_model.pth')
import joblib

joblib.dump(feature_scaler, 'feature_scaler.save')
joblib.dump(target_scaler, 'target_scaler.save')