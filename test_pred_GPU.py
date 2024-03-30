import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import csv

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


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
y = data.iloc[:, -1].values  # 第四列作为标签

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

# 创建网络实例并移动到设备上
net = SimpleNet().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 训练过程
epochs = 10000
batch_size = 512

# 创建TensorDataset和DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

loss_history = []  # 用于记录损失的列表

for epoch in range(epochs):
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = net(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * X_batch.size(0)

    epoch_loss /= len(train_loader.dataset)
    loss_history.append(epoch_loss)

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {epoch_loss}')

# 测试
net.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    predicted = net(X_test)
    loss = criterion(predicted, y_test.to(device))
    print(f'Test Loss: {loss.item()}')

# 绘制损失曲线
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss During Training')
plt.legend()
plt.show()

# 将损失数据保存到CSV文件中
with open('data/loss_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Loss'])
    for epoch, loss in enumerate(loss_history):
        writer.writerow([epoch, loss])

# 预测新数据
# 请确保new_data是你想要预测的新数据
new_data = [[256.07324, 35.818188, 68.8326]]
new_data = scaler.transform(new_data)  # 使用之前的scaler来转换新数据
new_data_tensor = torch.FloatTensor(new_data).to(device)  # 转换为torch张量并移到GPU

net.eval()  # 将模型设置为评估模式
with torch.no_grad():
    predicted_output = net(new_data_tensor)
    print("Predicted Output:", predicted_output.item())
