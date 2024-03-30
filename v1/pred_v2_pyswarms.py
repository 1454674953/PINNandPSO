import numpy as np
import torch
import torch.nn as nn
from pyswarms.single.global_best import GlobalBestPSO
import joblib

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


scaler = joblib.load('model/scaler_1.pkl')  # 假设 scaler 的状态被保存在 'scaler.pkl'
# 加载模型
model = SimpleNet()
model.load_state_dict(torch.load('model/model_1.pth'))
model.eval()

# 加载 StandardScaler 实例
# 注意：需要从保存 scaler 状态的地方加载它，这里假设它被保存到 'scaler.pkl'

# 固定特征值
fixed_feature1 = 199.8  # 示例值
fixed_feature2 = 2.6    # 示例值

# 目标输出值
target_value = 10.6  # 示例值

# 确保fixed_features_scaled是全局变量或者以其他方式可以在函数中使用
fixed_features_scaled = torch.tensor(scaler.transform([[fixed_feature1, fixed_feature2, 0]]), dtype=torch.float32).squeeze(0)[:2]
target_output = torch.tensor([target_value], dtype=torch.float32)

# PSO 参数设置
n_particles = 30  # 粒子数量
n_dimensions = 1  # 我们只优化一个变量特征
n_iterations = 100  # 迭代次数
# bounds = np.array([[0, 100]])  # 变量特征的边界，范围在0到100之间


# 设置原始尺度的边界
original_bounds = np.array([[0,50]])

# 转换为标准化尺度的边界
# 使用scaler.transform来进行转换，但要注意我们需要为其他特征也提供值
# 这里假设其他特征的平均值可以代表一个“典型”点
mean_values_for_other_features = scaler.mean_[:-1]  # 获取除了我们关注的特征以外的其他特征的均值
lower_bound = np.hstack((mean_values_for_other_features, original_bounds[:, 0]))
upper_bound = np.hstack((mean_values_for_other_features, original_bounds[:, 1]))

# 转换边界
scaled_lower_bound = scaler.transform([lower_bound])
scaled_upper_bound = scaler.transform([upper_bound])

# 我们只关心最后一个特征（即我们想要优化的特征）的标准化后的边界
bounds = np.array([[scaled_lower_bound[0, -1], scaled_upper_bound[0, -1]]])

# 打印标准化后的边界
print(f"标准化后的边界: {bounds}")


# 定义目标函数，该函数计算所有粒子的损失值
def f_per_particle(m, fixed_features_scaled, target_output):
    """计算每个粒子的损失值"""
    # m 是一个1维数组，代表单个粒子的位置
    m_tensor = torch.tensor(m, dtype=torch.float32)
    # fixed_features_scaled 是一个1维数组，包含所有固定特征
    # 我们需要将m_tensor变形为[1, n_dimensions]，然后与fixed_features_scaled拼接
    combined_input = torch.cat((fixed_features_scaled.unsqueeze(0), m_tensor.view(1, -1)), dim=1)
    model.eval()
    with torch.no_grad():
        output = model(combined_input)
    loss = (output - target_output).pow(2).sum().item()
    return loss


# 将目标函数转化为PSO可以优化的形式
def f(x, fixed_features_scaled=fixed_features_scaled, target_output=target_output):
    return np.array([f_per_particle(x[i], fixed_features_scaled, target_output) for i in range(x.shape[0])])

# 创建边界
max_bound = scaler.transform([[fixed_feature1, fixed_feature2, 50]])[0][-1]
min_bound = scaler.transform([[fixed_feature1, fixed_feature2, 0]])[0][-1]
bounds = (np.ones(n_dimensions) * min_bound, np.ones(n_dimensions) * max_bound)

# 创建粒子群优化器
optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=n_dimensions, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}, bounds=bounds)

# 执行优化
cost, pos = optimizer.optimize(f, iters=n_iterations)

# 输出全局最佳位置（反向标准化以得到原始尺度的值）
optimized_feature_original_scale = scaler.inverse_transform([[0, 0, pos[0]]])[0, 2]
print(f'优化后的第三个特征值（原始尺度）: {optimized_feature_original_scale}')
