# 导入库
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
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

# 现在保存它
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
joblib.dump(scaler, os.path.join(model_dir, 'scaler_1.pkl'))



# 初始化网络、损失函数和优化器
net = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 训练过程
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = net(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

# 测试
with torch.no_grad():
    predicted = net(X_test_tensor)
    loss = criterion(predicted, y_test_tensor)
    print(f'Test Loss: {loss.item()}')

'''
测试
'''
# 假设 new_data 是你的新输入数据，例如：
new_data = [[199.8 ,  2.6 ,  47.17090584317077]]

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

# 保存模型
torch.save(net.state_dict(), 'model/model_1.pth')



