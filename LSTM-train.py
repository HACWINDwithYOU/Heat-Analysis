import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import tensorflow as tf

# 检查GPU可用性
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print("GPU Details:")
    for gpu in tf.config.list_physical_devices('GPU'):
        print(tf.config.experimental.get_device_details(gpu))

# 1. 加载数据
data = pd.read_csv('measures_v2.csv')

# 2. 定义输入特征和目标变量
features = ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'torque', 'ambient']
targets = ['stator_winding', 'stator_tooth', 'pm', 'stator_yoke']

# 3. 定义要排除的profile_id（示例值，请根据实际情况修改）
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


# 7. 构建复杂LSTM模型
def build_industrial_lstm(input_shape, output_dim):
    """
    工业级单向LSTM，严格保证仅使用历史数据
    结构说明：
    - 3层单向LSTM逐步降维
    - 每层后接BatchNorm和Dropout防止过拟合
    - 最终通过稠密层输出
    """
    model = Sequential([
        # 第一层LSTM（返回序列供下层使用）
        LSTM(128,
             input_shape=input_shape,
             return_sequences=True,
             kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),

        # 第二层LSTM
        LSTM(64,
             return_sequences=False,  # 最后一层LSTM不返回序列
             kernel_regularizer=l2(0.005)),
        BatchNormalization(),

        # 输出分支
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])  # 确保包含metrics
    return model


from tensorflow.keras.layers import Conv1D, Input
from tensorflow.keras.models import Model


def build_causal_lstm(input_shape, output_dim):
    """
    因果卷积+LSTM混合结构
    关键特性：
    - 使用因果卷积（padding='causal'）预处理时序
    - 保证卷积操作不泄露未来信息
    - LSTM进一步捕捉长期依赖
    """
    inputs = Input(shape=input_shape)

    # 因果卷积层（kernel_size=3表示只看过去3步）
    x = Conv1D(64, kernel_size=3, padding='causal', activation='relu')(inputs)
    x = BatchNormalization()(x)

    # LSTM处理
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64)(x)

    # 输出
    outputs = Dense(output_dim, dtype='float32')(x)  # 关键修改

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])  # 确保包含metrics
    return model


from tensorflow.keras.layers import Dropout


def build_standard_lstm(input_shape, output_dim):
    """平衡结构和性能的通用LSTM"""
    model = Sequential([
        LSTM(64,
             input_shape=input_shape,
             return_sequences=True,
             kernel_regularizer=l2(0.001)),
        Dropout(0.2),

        LSTM(32),
        Dropout(0.1),

        Dense(32, activation='relu'),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_complex_lstm(input_shape, output_dim):
    model = Sequential([
        # 第一LSTM层（带双向和返回序列）
        LSTM(128,
             input_shape=input_shape,
             return_sequences=True,
             kernel_regularizer=l2(0.01),
             recurrent_dropout=0.2),
        BatchNormalization(),
        Dropout(0.3),

        # 第二LSTM层
        LSTM(64,
             return_sequences=True,
             kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        Dropout(0.2),

        # 第三LSTM层
        LSTM(32,
             return_sequences=False),
        BatchNormalization(),

        # 密集层
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.1),
        Dense(32, activation='relu'),

        # 输出层
        Dense(output_dim)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# # GPU优化配置
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         # 设置GPU内存按需增长
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         # 设置混合精度训练（进一步提高速度）
#         policy = tf.keras.mixed_precision.Policy('mixed_float16')
#         tf.keras.mixed_precision.set_global_policy(policy)
#         print("GPU配置完成，启用混合精度训练")
#     except RuntimeError as e:
#         print(e)


# 初始化模型
model = build_causal_lstm(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    output_dim=len(targets)
)
# model = build_simple_lstm(X_train, targets)
model.summary()

# 8. 训练配置
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

# if tf.config.list_physical_devices('GPU'):
#     # 自动分配GPU内存
#     physical_devices = tf.config.list_physical_devices('GPU')
#     try:
#         tf.config.experimental.set_memory_growth(physical_devices[0], True)
#         # 启用混合精度
#         tf.keras.mixed_precision.set_global_policy('mixed_float16')
#     except:
#         pass

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)


# 9. 评估模型
def evaluate_model(model, X_test, y_test):
    results = model.evaluate(X_test, y_test, verbose=0)

    if isinstance(results, list):
        print(f"Test Loss: {results[0]:.4f}, Test MAE: {results[1]:.4f}")
    else:
        print(f"Test Loss: {results:.4f}")

    y_pred = model.predict(X_test)
    return y_test, y_pred


# 执行评估
y_true, y_pred = evaluate_model(model, X_test, y_test_scaled)


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
import joblib

model.save('complex_lstm_model.keras')
joblib.dump(feature_scaler, 'feature_scaler.save')
joblib.dump(target_scaler, 'target_scaler.save')