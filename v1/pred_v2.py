import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 定义神经网络结构
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
# 使用标准化器转换固定特征
fixed_features_scaled = torch.tensor(scaler.transform([[fixed_feature1, fixed_feature2, 0]]), dtype=torch.float32).squeeze(0)[:2]

# 将目标值转换为Tensor
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

# 初始化粒子群
particles = np.random.rand(n_particles, n_dimensions) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
velocities = np.zeros_like(particles)  # 粒子的速度初始化为0
pbest_positions = particles.copy()  # 个体最佳位置
pbest_scores = np.full(n_particles, np.inf)  # 个体最佳得分初始化为无穷大

# 初始化全局最佳
gbest_score = np.inf  # 全局最佳得分初始化为无穷大
gbest_position = np.zeros(n_dimensions)  # 全局最佳位置

# 计算损失函数的函数
def calculate_loss(model, fixed_features_scaled, variable_feature, target_output):
    # 确保variable_feature是一维张量
    variable_feature_tensor = torch.tensor(variable_feature, dtype=torch.float32)
    # 组合输入张量
    combined_input = torch.cat((fixed_features_scaled, variable_feature_tensor)).unsqueeze(0)
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        output = model(combined_input)
    loss = (output - target_output).pow(2).sum().item()
    return loss


# 更新粒子群
for i in range(n_iterations):
    for j in range(n_particles):
        # 计算当前粒子的得分

        current_score = calculate_loss(model, fixed_features_scaled, particles[j], target_output)

        # 更新个体最佳
        if current_score < pbest_scores[j]:
            pbest_scores[j] = current_score
            pbest_positions[j] = particles[j].copy()

        # 更新全局最佳
        if current_score < gbest_score:
            gbest_score = current_score
            gbest_position = particles[j].copy()

    # 更新粒子的速度和位置
    w = 0.5  # 惯性权重
    c1 = 1.0  # 认知系数
    c2 = 1.5  # 社会系数
    for j in range(n_particles):
        velocities[j] = (w * velocities[j] +
                         c1 * np.random.rand(n_dimensions) * (pbest_positions[j] - particles[j]) +
                         c2 * np.random.rand(n_dimensions) * (gbest_position - particles[j]))
        particles[j] += velocities[j]
        # 将粒子限制在搜索空间边界内
        particles[j] = np.maximum(particles[j], bounds[:, 0])
        particles[j] = np.minimum(particles[j], bounds[:, 1])

    # 打印当前迭代的最佳得分
    print(f'迭代 {i+1}/{n_iterations}, 最佳得分: {gbest_score}')

# 输出全局最佳位置
optimized_feature = gbest_position[0]
# 反向标准化以得到原始尺度的值
optimized_feature_original_scale = scaler.inverse_transform([[0, 0, optimized_feature]])[0, 2]
print(f'优化后的第三个特征值（原始尺度）: {optimized_feature_original_scale}')
