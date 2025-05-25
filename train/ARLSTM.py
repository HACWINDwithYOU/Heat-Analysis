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
data = pd.read_csv('../datasets/measures_v2.csv')

# 2. 定义输入特征和目标变量
features = ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'torque', 'ambient']
targets = ['stator_winding', 'stator_tooth', 'pm', 'stator_yoke']

# 3. 定义要排除的profile_id
exclude_profiles_train = [2, 4, 20]  # 不参与训练的profile
test_profiles = [2, 4, 20]  # 专门用于测试的profile


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

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,  # 20%训练数据作为验证集
    random_state=42,
    shuffle=False    # 保持时间序列顺序
)

print(f"最终数据集形状 - 训练: {X_train_final.shape}, 验证: {X_val.shape}, 测试: {X_test.shape}")


# 6. 数据标准化
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# 训练集标准化
X_train_final_reshaped = X_train_final.reshape(-1, len(features))
feature_scaler.fit(X_train_final_reshaped)  # 仅用训练集拟合
X_train_final_scaled = feature_scaler.transform(X_train_final_reshaped)
X_train_final = X_train_final_scaled.reshape(X_train_final.shape)

# 验证集标准化（使用训练集的scaler）
X_val_reshaped = X_val.reshape(-1, len(features))
X_val_scaled = feature_scaler.transform(X_val_reshaped)
X_val = X_val_scaled.reshape(X_val.shape)

# 测试集标准化（同上）
X_test_reshaped = X_test.reshape(-1, len(features))
X_test_scaled = feature_scaler.transform(X_test_reshaped)
X_test = X_test_scaled.reshape(X_test.shape)

# 目标变量标准化（仅用训练集拟合）
y_train_final_reshaped = y_train_final.reshape(-1, len(targets))
target_scaler.fit(y_train_final_reshaped)  # 关键修改：仅用训练集
y_train_final_scaled = target_scaler.transform(y_train_final_reshaped)
y_train_final = y_train_final_scaled.reshape(y_train_final.shape)

# 验证集和测试集转换
y_val_scaled = target_scaler.transform(y_val.reshape(-1, len(targets))).reshape(y_val.shape)
y_test_scaled = target_scaler.transform(y_test.reshape(-1, len(targets))).reshape(y_test.shape)


# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_final).to(device)
y_train_tensor = torch.FloatTensor(y_train_final).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)

# 创建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, torch.FloatTensor(y_val_scaled).to(device))
test_dataset = TensorDataset(X_test_tensor, torch.FloatTensor(y_test_scaled).to(device))

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 验证集不shuffle
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("训练目标变量范围：", y_train_final_scaled.min(), y_train_final_scaled.max())  # 应接近[0,1]
print("验证目标变量范围：", y_val_scaled.min(), y_val_scaled.max())  # 可能略微超出[0,1]

# 7. 构建PyTorch LSTM模型
class AutoregressiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, prediction_steps=1):
        super(AutoregressiveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_steps = prediction_steps

        # 主LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

        # 自回归反馈层
        self.feedback = nn.Linear(output_size, input_size)

    def forward(self, x, future=0):
        # x shape: (batch, seq_len, input_size)
        batch_size = x.size(0)
        outputs = []

        # 初始化隐藏状态
        h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # 处理输入序列
        for t in range(x.size(1)):
            _, (h_t, c_t) = self.lstm(x[:, t:t + 1, :], (h_t, c_t))

        # 自回归预测
        last_output = x[:, -1:, :]
        for _ in range(self.prediction_steps if future == 0 else future):
            # 通过LSTM
            _, (h_t, c_t) = self.lstm(last_output, (h_t, c_t))

            # 通过全连接层
            out = self.fc(h_t[-1])  # 取最后一层的隐藏状态
            outputs.append(out.unsqueeze(1))

            # 反馈到输入
            if self.feedback is not None:
                last_output = self.feedback(out).unsqueeze(1)

        return torch.cat(outputs, dim=1)  # (batch, prediction_steps, output_size)


# 初始化模型
input_size = len(features)
hidden_size = 128
num_layers = 2
output_size = len(targets)

model = AutoregressiveLSTM(input_size, hidden_size, num_layers, output_size).to(device)
print(model)


# 8. 训练配置
criterion = nn.MSELoss()
# criterion = TemporalSmoothLoss(base_loss=nn.MSELoss(), alpha=0.1, lookback=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# 早停机制
early_stopping_patience = 999
best_loss = float('inf')
patience_counter = 0


# 训练函数
# 在训练前初始化记录器
train_losses = []
val_losses = []
learning_rates = []


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100):
    model.train()
    global best_loss, patience_counter

    for epoch in range(epochs):
        # 训练阶段
        model.train()
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

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 记录数据
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # 改进的早停机制
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Validation loss improved to {best_loss:.4f}, saving model...")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve for {patience_counter}/{early_stopping_patience} epochs")
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

        # 学习率调整
        scheduler.step(avg_val_loss)  # 根据验证损失调整



# 开始训练
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20)

torch.save(model.state_dict(), 'last_model.pth')

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))


def plot_training_curves():
    plt.figure(figsize=(15, 10))

    # Loss曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # MSE曲线（假设记录了MSE）
    plt.subplot(2, 2, 2)
    plt.plot(val_losses, label='Validation MSE', color='red')  # 如果是MSE损失
    plt.title('Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)

    # 学习率曲线
    plt.subplot(2, 2, 3)
    plt.plot(learning_rates, label='Learning Rate', color='green')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.grid(True)
    plt.yscale('log')  # 对数尺度更清晰

    # 损失差值曲线
    plt.subplot(2, 2, 4)
    loss_diff = [t - v for t, v in zip(train_losses, val_losses)]
    plt.plot(loss_diff, label='Train-Val Difference', color='purple')
    plt.title('Generalization Gap (Train - Val)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png')  # 保存图像
    plt.show()


# 训练后调用
plot_training_curves()

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
    y_true = y_true_scaled  # 不再反标准化真实值
    # y_true = y_true_scaled

    # 计算指标
    scaled_mse = np.mean((y_true_scaled - y_pred_scaled) ** 2)

    # 2. 原始尺度下的误差（用于业务报告）
    original_mse = np.mean((target_scaler.inverse_transform(y_true_scaled) - y_pred) ** 2)

    print(f"标准化尺度 MSE: {scaled_mse:.4f}")
    print(f"原始尺度 MSE: {original_mse:.4f}")

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