import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import os

# 配置设置
MODEL_PATH = '../models/casual_lstm_e234.keras'
FEATURE_SCALER_PATH = '../models/feature_scaler.save'
TARGET_SCALER_PATH = '../models/target_scaler.save'
DATA_PATH = '../datasets/measures_v2.csv'  # 原始数据路径
PROFILE_ID_TO_PREDICT = 2  # 要预测的profile_id
TIME_STEPS = 10  # 与训练时一致


# 1. 加载模型和scaler
def load_resources():
    # 确保文件存在
    if not all(os.path.exists(f) for f in [MODEL_PATH, FEATURE_SCALER_PATH, TARGET_SCALER_PATH]):
        raise FileNotFoundError("模型或scaler文件未找到")

    # 加载模型和scaler
    model = load_model(MODEL_PATH)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    return model, feature_scaler, target_scaler


# 2. 数据预处理函数
def prepare_profile_data(data, profile_id, feature_scaler, time_steps=10):
    # 筛选特定profile的数据
    profile_data = data[data['profile_id'] == profile_id]
    if len(profile_data) == 0:
        raise ValueError(f"profile_id {profile_id} 不存在")

    # 定义特征和目标列（需与训练时一致）
    features = ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'torque', 'ambient']
    targets = ['stator_winding', 'stator_tooth', 'pm', 'stator_yoke']

    # 创建序列
    X = profile_data[features].values
    y = profile_data[targets].values

    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])

    if len(X_seq) == 0:
        raise ValueError("数据不足以创建序列")

    # 转换为numpy数组并标准化
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # 标准化特征
    X_reshaped = X_seq.reshape(-1, len(features))
    X_scaled = feature_scaler.transform(X_reshaped)
    X_seq_scaled = X_scaled.reshape(X_seq.shape)

    return X_seq_scaled, y_seq, features, targets


# 3. 推理和可视化函数
def predict_and_visualize(model, X, y_true, target_scaler, targets):
    # 预测
    y_pred_scaled = model.predict(X, verbose=0)

    # 反标准化
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    # y_true_original = target_scaler.inverse_transform(y_true) if y_true.shape[1] == len(targets) else y_true
    y_true_original = y_true

    # 计算指标
    mse = np.mean((y_true_original - y_pred) ** 2)
    mae = np.mean(np.abs(y_true_original - y_pred))

    print(f"\n预测结果指标 (profile_id: {PROFILE_ID_TO_PREDICT})")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")

    # 可视化
    plt.figure(figsize=(15, 10))
    for i, target in enumerate(targets):
        plt.subplot(2, 2, i + 1)
        plt.plot(y_true_original[:, i], label='True', alpha=0.7)
        plt.plot(y_pred[:, i], label='Predicted', alpha=0.7)
        plt.title(f'{target} Prediction')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

    return y_true_original, y_pred


# 主函数
def main():
    print("正在加载模型和预处理工具...")
    model, feature_scaler, target_scaler = load_resources()

    print("正在加载数据...")
    data = pd.read_csv(DATA_PATH)

    print(f"正在预处理 profile_id={PROFILE_ID_TO_PREDICT} 的数据...")
    X, y_true, features, targets = prepare_profile_data(
        data, PROFILE_ID_TO_PREDICT, feature_scaler, TIME_STEPS
    )

    print("进行预测和结果评估...")
    y_true_original, y_pred = predict_and_visualize(
        model, X, y_true, target_scaler, targets
    )

    # 保存预测结果
    results = pd.DataFrame({
        **{f'true_{target}': y_true_original[:, i] for i, target in enumerate(targets)},
        **{f'pred_{target}': y_pred[:, i] for i, target in enumerate(targets)}
    })
    # results.to_csv(f'profile_{PROFILE_ID_TO_PREDICT}_predictions.csv', index=False)
    # print(f"\n预测结果已保存到 profile_{PROFILE_ID_TO_PREDICT}_predictions.csv")


if __name__ == "__main__":
    main()