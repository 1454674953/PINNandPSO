import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 神经网络定义
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
y = data.iloc[:, -1].values   # 第四列作为标签

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# 转换为torch张量
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# 初始化网络、损失函数和优化器
net = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 训练过程
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = net(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

# 测试
with torch.no_grad():
    predicted = net(X_test)
    loss = criterion(predicted, y_test)
    print(f'Test Loss: {loss.item()}')

# 保存模型（如果需要）
# torch.save(net.state_dict(), 'model.pth')

# 以后可以使用这样的方式来加载模型（如果需要）
# model = SimpleNet()
# model.load_state_dict(torch.load('model.pth'))
# model.eval()

# test_input = torch.tensor([232.1, 8.6, 8.7])
# predicted_output = net(test_input)
# print(predicted_output)

# 假设 new_data 是你的新输入数据，例如：
new_data = [[237.54047 ,  37.51869 ,  31.828905]]

# 如果之前对训练数据进行了标准化，同样的转换也需要应用到新数据上
new_data = scaler.transform(new_data)  # 使用之前的scaler来转换新数据

# 将数据转换为torch张量
new_data_tensor = torch.FloatTensor(new_data)

# 使用模型进行预测
net.eval()  # 将模型设置为评估模式
with torch.no_grad():
    predicted_output = net(new_data_tensor)
    print("Predicted Output:", predicted_output.item())  # 如果你预期的输出是单个数值，可以使用 .item()

'''
反向查找
'''
# 假设目标输出值是 some_target_value
some_target_value = 22.1
target_output = torch.tensor([some_target_value])  # 替换为你的目标输出值

# 初始化一组随机输入特征，设置 requires_grad=True 以便进行梯度下降
input_features = torch.randn(3, requires_grad=True)

# 选择优化器
optimizer = torch.optim.Adam([input_features], lr=0.01)

# 进行优化以寻找接近目标输出的输入特征
for step in range(10000):  # 迭代次数可以根据需要调整
    optimizer.zero_grad()
    net.eval()  # 确保网络处于评估模式
    current_output = net(input_features.unsqueeze(0))

    # 计算当前输出与目标输出的差异
    loss = (current_output - target_output).pow(2).sum()

    # 反向传播优化输入特征
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item()}")

# 输出优化后的输入特征
optimized_input = input_features.detach().numpy()
print("Optimized Input Features (before scaling):", optimized_input)

# 反标准化以得到原始数据的尺度
optimized_input_original_scale = scaler.inverse_transform(optimized_input.reshape(1, -1))
print("Optimized Input Features (original scale):", optimized_input_original_scale)

# 256.07324   35.818188  68.8326