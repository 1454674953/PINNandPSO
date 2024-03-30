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
fixed_feature1 = 232.1  # 示例值
fixed_feature2 = 8.6  # 示例值
# 使用标准化器转换固定特征
fixed_features_scaled = torch.tensor(scaler.transform([[fixed_feature1, fixed_feature2, 0]]),
                                     dtype=torch.float32).squeeze(0)[:2]

# PSO 参数设置
n_particles = 30  # 粒子数量
n_dimensions = 1  # 我们只优化一个变量特征
n_iterations = 100  # 迭代次数

# 设置原始尺度的边界
original_bounds = np.array([[0, 100]])

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
# 初始化粒子群
particles = np.random.rand(n_particles, n_dimensions) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
velocities = np.zeros_like(particles)  # 粒子的速度初始化为0
pbest_positions = particles.copy()  # 个体最佳位置
pbest_scores = np.full(n_particles, np.inf)  # 个体最佳得分初始化为无穷大

# 初始化全局最佳
gbest_score = np.inf  # 全局最佳得分初始化为无穷大
gbest_position = np.zeros(n_dimensions)  # 全局最佳位置

# 定义速度边界（这里假设为位置边界的10%）
velocity_bounds = 0.1 * (bounds[:, 1] - bounds[:, 0])
max_velocity = velocity_bounds
min_velocity = -velocity_bounds

# 目标函数：计算模型在每个粒子位置上的输出
def objective_function(particles):
    # particles是一个数组，其中每一行代表一个粒子的位置
    model_outputs = np.zeros(particles.shape[0])
    for i, particle in enumerate(particles):
        variable_feature_tensor = torch.tensor([particle], dtype=torch.float32)
        combined_input = torch.cat((fixed_features_scaled.unsqueeze(0), variable_feature_tensor), dim=1)
        model.eval()
        with torch.no_grad():
            output = model(combined_input)
        model_outputs[i] = output.item()
    return model_outputs

# 初始化粒子群优化器
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
bounds = (np.ones(n_dimensions) * bounds[0, 0], np.ones(n_dimensions) * bounds[0, 1])  # 设置边界
optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=n_dimensions, options=options, bounds=bounds)

# 执行优化
cost, pos = optimizer.optimize(objective_function, iters=n_iterations)

# 输出全局最佳位置（反向标准化以得到原始尺度的值）
optimized_feature_original_scale = scaler.inverse_transform([[0, 0, pos[0]]])[0, 2]
print(f'优化后的第三个特征值（原始尺度）: {optimized_feature_original_scale}')
print(f'模型输出的最小值（原始尺度）: {cost}')
