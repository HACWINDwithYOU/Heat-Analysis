import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. 加载数据
data = pd.read_csv('measures_v2.csv')  # 替换为你的文件路径

# 2. 定义输入特征和目标变量
features = ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'torque', 'ambient']
targets = ['stator_winding', 'stator_tooth', 'pm', 'stator_yoke']


# 3. 按profile_id分组处理
def create_sequences(group, time_steps=10):
    """为单个profile创建时间序列样本"""
    X_group = group[features].values
    y_group = group[targets].values

    X_seq, y_seq = [], []
    for i in range(len(X_group) - time_steps):
        X_seq.append(X_group[i:i + time_steps])
        y_seq.append(y_group[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

# 初始化空数组
X_all, y_all = np.empty((0, 10, len(features))), np.empty((0, len(targets)))

# 按profile分组并创建序列
for _, group in data.groupby('profile_id'):
    X_seq, y_seq = create_sequences(group)
    if len(X_seq) > 0:  # 确保有足够数据创建序列
        X_all = np.concatenate((X_all, X_seq))
        y_all = np.concatenate((y_all, y_seq))

# 4. 数据标准化 (每个特征单独缩放)
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# 重塑用于缩放 (n_samples * time_steps, n_features)
X_reshaped = X_all.reshape(-1, len(features))
X_reshaped_scaled = feature_scaler.fit_transform(X_reshaped)
X_all_scaled = X_reshaped_scaled.reshape(X_all.shape)

y_all_scaled = target_scaler.fit_transform(y_all)

# 5. 划分训练集和测试集 (保持profile完整性)
X_train, X_test, y_train, y_test = train_test_split(
    X_all_scaled, y_all_scaled, test_size=0.2, random_state=42, shuffle=True
)

# 6. 构建LSTM模型 (与之前相同)
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(targets))
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# 7. 训练模型
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# 8. 评估与预测
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# 测试集评估
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {test_loss}')

# 预测示例
sample_idx = 0
sample_input = X_test[sample_idx][np.newaxis, ...]  # 添加批次维度
prediction_scaled = model.predict(sample_input)

# 反标准化
prediction_actual = target_scaler.inverse_transform(prediction_scaled)
true_value_actual = target_scaler.inverse_transform(y_test[sample_idx][np.newaxis, ...])

print("Predicted Values:", prediction_actual)
print("True Values:", true_value_actual)

# 9. 保存模型
model.save('lstm_multi_profile_model.h5')