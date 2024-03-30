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


'''
反向查找
'''
# 固定特征值
fixed_feature1 = 199.8
fixed_feature2 = 2.6

# 标准化固定特征值
fixed_features_scaled = torch.tensor(scaler.transform([[fixed_feature1, fixed_feature2, 0]]), dtype=torch.float32).squeeze(0)[
                 :2]

# 初始化第三个特征的值
variable_feature = torch.randn(1, requires_grad=True)

# 选择优化器
optimizer = optim.Adam([variable_feature], lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

# 目标输出值
target_output = torch.tensor([10.6])  # 目标值需要根据具体情况设置

# 优化循环
for step in range(10000):
    optimizer.zero_grad()

    # 组合输入张量
    combined_input = torch.cat((fixed_features_scaled, variable_feature)).unsqueeze(0)

    model.eval()  # 确保网络处于评估模式
    current_output = model(combined_input)

    # 计算损失
    loss = (current_output - target_output).pow(2).sum()

    # 反向传播
    loss.backward()
    optimizer.step()

    scheduler.step()
    if step % 1000 == 0:
        print(f"Step {step}: Loss = {loss.item()}")

# 获取优化后的特征值
optimized_feature = variable_feature.detach().item()

# 反向标准化以得到原始尺度的值
optimized_feature_original_scale = scaler.inverse_transform([[0, 0, optimized_feature]])[0,2]

print(f"Optimized third feature value (original scale): {optimized_feature_original_scale}")
