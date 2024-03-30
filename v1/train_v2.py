import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

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


# 读取数据
data = pd.read_csv('Advertising.csv')
X = data.iloc[:, 1:-1].values  # 前三列作为特征
y = data.iloc[:, -1].values  # 第四列作为标签

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = y.reshape(-1, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.05, random_state=42)

# 转换为torch张量
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)


# 适配scaler之后
# scaler.fit(X_train)


# 保存模型和scaler的目录
model_dir = 'model_2'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
joblib.dump(scaler, os.path.join(model_dir, 'scaler_2.pkl'))

# 初始化网络
net = SimpleNet()

# PSO的参数设置
n_particles = 100  # 粒子数量
n_iterations = 1000  # 迭代次数
n_dimensions = sum(p.numel() for p in net.parameters())  # 网络参数的数量

# 初始化粒子群的位置和速度
particles = np.random.randn(n_particles, n_dimensions)
velocities = np.zeros_like(particles)

# 初始化个体最佳和全局最佳
pbest_positions = particles.copy()
pbest_scores = np.full(n_particles, np.inf)
gbest_score = np.inf
gbest_position = None

# 定义损失函数
criterion = nn.MSELoss()


def set_weights(net, weights_vector):
    """将扁平化的权重向量应用到网络的权重和偏置上。"""
    offset = 0
    for param in net.parameters():
        param_shape = param.data.shape
        param_size = torch.numel(param)
        param.data = torch.FloatTensor(weights_vector[offset:offset + param_size]).view(param_shape)
        offset += param_size


def calculate_loss(net, criterion, X, y):
    """计算当前网络权重下的损失。"""
    outputs = net(X)
    loss = criterion(outputs, y)
    return loss.item()


# 开始PSO优化
for i in range(n_iterations):
    for j in range(n_particles):
        # 设置网络权重
        set_weights(net, particles[j])

        # 计算损失
        current_loss = calculate_loss(net, criterion, X_train_tensor, y_train_tensor)

        # 更新个体最佳
        if current_loss < pbest_scores[j]:
            pbest_scores[j] = current_loss
            pbest_positions[j] = particles[j].copy()

        # 更新全局最佳
        if current_loss < gbest_score:
            gbest_score = current_loss
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

    # 打印当前迭代的最佳损失
    print(f'迭代 {i + 1}/{n_iterations}, 最佳损失: {gbest_score}')

# 设置为全局最优权重
set_weights(net, gbest_position)

# 保存模型权重
torch.save(net.state_dict(), os.path.join(model_dir, 'model_2.pth'))

# ... 以下代码不变 ...
'''
测试
'''
# 假设 new_data 是你的新输入数据，例如：
new_data = [[199.8 ,  2.6 ,  21.2]]

# 如果你之前对训练数据进行了标准化，同样的转换也需要应用到新数据上
new_data = scaler.transform(new_data)  # 使用之前的scaler来转换新数据

# 将数据转换为torch张量
new_data_tensor = torch.FloatTensor(new_data)

# 使用模型进行预测
net.eval()  # 将模型设置为评估模式
with torch.no_grad():
    predicted_output = net(new_data_tensor)
    print("Predicted Output:", predicted_output.item())  # 如果你预期的输出是单个数值，可以使用 .item()
##############################################
