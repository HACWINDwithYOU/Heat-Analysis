import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import MeanSquaredError

# 加载模型和scaler
model = load_model('models/lstm_multi_profile_model.h5',
                  custom_objects={'mse': MeanSquaredError()})
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# 假设这些是之前训练时使用的特征和目标列名（需与训练时完全一致）
features = ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'torque', 'ambient']
targets = ['stator_winding', 'stator_tooth', 'pm', 'stator_yoke']


def preprocess_data(data, time_steps=10):
    """处理新数据，生成预测所需的序列"""
    # 标准化特征（假设scaler已拟合过）
    X_scaled = feature_scaler.transform(data[features])

    # 创建序列
    X_seq = []
    for i in range(len(X_scaled) - time_steps):
        X_seq.append(X_scaled[i:i + time_steps])
    return np.array(X_seq)


def predict_all_profiles(data, profile_col='profile_id'):
    """处理包含多个profile的数据集"""
    results = {}
    for profile_id, group in data.groupby(profile_col):
        if len(group) >= 10:  # 确保足够创建序列
            X_seq = preprocess_data(group)
            predictions = model.predict(X_seq)
            pred_actual = target_scaler.inverse_transform(predictions)

            # 获取对应的真实值（注意对齐时间点）
            y_true = group[targets].iloc[10:].values  # 跳过前time_steps个点
            results[profile_id] = {'pred': pred_actual, 'true': y_true}
    return results

# 在加载新数据前，先加载训练数据来拟合scaler
train_data = pd.read_csv('measures_v2.csv')  # 替换为原始训练数据路径

# 使用训练数据拟合scaler
feature_scaler.fit(train_data[features])
target_scaler.fit(train_data[targets])

# 然后再处理新数据
new_data = pd.read_csv('measures_v2.csv')
predictions = predict_all_profiles(new_data)


# 加载新数据（假设包含profile_id列）
new_data = pd.read_csv('measures_v2.csv')

# 进行预测（确保scaler已加载或重新拟合）
# 如果scaler未保存，需要重新拟合（使用训练数据的统计量）
# feature_scaler.fit(training_data[features])
# target_scaler.fit(training_data[targets])

predictions = predict_all_profiles(new_data)


def plot_comparison(results, n_profiles=3):
    """绘制前n个profile的预测对比图"""
    plt.figure(figsize=(15, 4 * len(targets)))

    for i, profile_id in enumerate(list(results.keys())[:n_profiles]):
        res = results[profile_id]

        for j, target in enumerate(targets):
            ax = plt.subplot(len(targets), n_profiles, j * n_profiles + i + 1)

            # 绘制曲线
            ax.plot(res['true'][:, j], label='True', color='blue', alpha=0.6)
            ax.plot(res['pred'][:, j], label='Predicted', color='red', linestyle='--')

            # 装饰图形
            ax.set_title(f'Profile {profile_id} - {target}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    plt.show()


# 绘制前3个profile的对比
plot_comparison(predictions, n_profiles=3)

# 将预测结果保存到CSV
output_dfs = []
for profile_id, res in predictions.items():
    df = pd.DataFrame({
        'profile_id': profile_id,
        'time_step': np.arange(len(res['pred'])) + 10,  # 从time_steps开始计数
        **{f'pred_{t}': res['pred'][:, i] for i, t in enumerate(targets)},
        **{f'true_{t}': res['true'][:, i] for i, t in enumerate(targets)}
    })
    output_dfs.append(df)

pd.concat(output_dfs).to_csv('predictions_vs_actual.csv', index=False)
print("预测结果已保存到 predictions_vs_actual.csv")