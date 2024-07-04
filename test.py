import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import numpy as np

# 定义基础模型，具有可配置的层数
class BaseModel(nn.Module):
    def __init__(self, input_size, hidden_size, feature_size, output_size, num_layers):
        super(BaseModel, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.feature_layer = nn.Linear(hidden_size, feature_size)
        self.output_layer = nn.Linear(feature_size, output_size)
    
    def forward(self, x):
        x = self.network(x)
        features = torch.relu(self.feature_layer(x))  # 倒数第二层的特征
        output = self.output_layer(features)  # 最后输出
        return output, features

# 定义Epinet模型，包含n个固定模型
class EpinetModel(nn.Module):
    def __init__(self, n_fixed_models, feature_size, prior_mu=0.0, prior_sigma=1.0, prior_scale=1.0):
        super(EpinetModel, self).__init__()
        self.n_fixed_models = n_fixed_models
        self.prior_scale = prior_scale
        self.prior_mu = torch.full((n_fixed_models,), prior_mu)
        self.prior_sigma = torch.full((n_fixed_models,), prior_sigma)
        input_dim = 1 + n_fixed_models + feature_size  # x, z, and base_features
        self.fixed_models = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        ) for _ in range(n_fixed_models)])
        
        self.network_trainable = nn.Sequential(
            nn.Linear(input_dim, 15),
            nn.ReLU(),
            nn.Linear(15, 15),
            nn.ReLU(),
            nn.Linear(15, 1)
        )
        
        # 初始化固定模型参数并将requires_grad设置为False
        for model in self.fixed_models:
            for param in model.parameters():
                param.requires_grad = False
    
    def forward(self, x, z, base_features):
        input_combined = torch.cat([x, z, base_features.detach()], dim=1)
        fixed_outputs = [model(input_combined) for model in self.fixed_models]
        fixed_output = torch.stack(fixed_outputs, dim=-1)
        
        # 直接使用z作为权重并将它们加权求和到一起
        weights = z.unsqueeze(-1)
        weighted_fixed_output = (fixed_output * weights).sum(dim=-1) * self.prior_scale
        
        output_trainable = self.network_trainable(input_combined)
        return (weighted_fixed_output + output_trainable).sum(dim=-1, keepdim=True)
    
    def sample_prior(self, batch_size):
        # 为每个数据点从贝叶斯先验中采样，并确保scale为非负
        return Normal(self.prior_mu.unsqueeze(0).expand(batch_size, -1), torch.abs(self.prior_sigma.unsqueeze(0).expand(batch_size, -1))).rsample()

# 定义包含基础模型和Epinet模型的完整Epinet
class Epinet(nn.Module):
    def __init__(self, base_model, n_fixed_models, feature_size, prior_mu=0.0, prior_sigma=1.0, prior_scale=1.0):
        super(Epinet, self).__init__()
        self.base_model = base_model
        self.epinet_model = EpinetModel(n_fixed_models, feature_size, prior_mu, prior_sigma, prior_scale)
    
    def forward(self, x, z):
        base_output, base_features = self.base_model(x)
        output = self.epinet_model(x, z, base_features)
        return base_output + output
    
    def predict_with_uncertainty(self, x, n_samples=10):
        base_output, base_features = self.base_model(x)
        predictions = []
        for _ in range(n_samples):
            z_prior = self.epinet_model.sample_prior(x.size(0))
            output = self.epinet_model(x, z_prior, base_features)
            predictions.append(output)
        predictions = torch.stack(predictions, dim=0)
        return predictions

# 生成训练集和测试集数据
def create_sine_data():
    # 训练集：从y=sin(x)函数中随机采样1000个点，有的地方密，有的地方疏
    x_train = np.concatenate([
        np.random.uniform(-50, -10, 100),
        np.random.uniform(-10, -5, 100),
        np.random.uniform(-5, 0, 10),
        np.random.uniform(0, 5, 100),
        np.random.uniform(5, 10, 1),
        np.random.uniform(5, 100, 30),
        np.random.uniform(100, 105, 1)
    ])
    y_train = np.sin(x_train)
    
    # 测试集：从y=sin(x)函数中均匀采样300个点
    x_test = np.linspace(-50, 105, 800)
    y_test = np.sin(x_test)
    
    return (torch.tensor(x_train).unsqueeze(1).float(), torch.tensor(y_train).unsqueeze(1).float()), \
           (torch.tensor(x_test).unsqueeze(1).float(), torch.tensor(y_test).unsqueeze(1).float())

train_data, test_data = create_sine_data()

# 可视化训练和测试数据
plt.figure(figsize=(10, 5))
plt.scatter(train_data[0].numpy(), train_data[1].numpy(), label='Train Data', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Train Data')
plt.legend()
plt.show()

# 初始化模型、损失函数和优化器
input_size = 1
hidden_size = 30
output_size = 1
num_layers = 10  # 可配置的层数
feature_size = 10  # 特征层大小
n_fixed_models = 8  # 修改后的固定模型数量
prior_scale = 0.1  # 你可以根据需要调整这个参数
n_samples = 100  # 你可以根据需要调整这个参数
weight_decay = 1e-4  # L2正则化参数

base_model = BaseModel(input_size, hidden_size, feature_size, output_size, num_layers)
model = Epinet(base_model, n_fixed_models, feature_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=weight_decay)

# 记录训练和测试损失
train_losses = []
test_losses = []

# 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 从贝叶斯先验中采样
    z_prior = model.epinet_model.sample_prior(train_data[0].size(0))
    
    outputs = model(train_data[0], z_prior)
    loss = criterion(outputs, train_data[1])
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())

    if (epoch+1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            z_prior = model.epinet_model.sample_prior(test_data[0].size(0))
            test_outputs = model(test_data[0], z_prior)
            test_loss = criterion(test_outputs, test_data[1])
            test_losses.append(test_loss.item())
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    z_prior = model.epinet_model.sample_prior(test_data[0].size(0))
    test_outputs = model(test_data[0], z_prior)
    test_loss = criterion(test_outputs, test_data[1])
    print(f'Test Loss: {test_loss.item():.4f}')

# 使用新的预测函数进行预测
with torch.no_grad():
    predictions = []
    for i in range(n_samples):
        z_prior = model.epinet_model.sample_prior(test_data[0].size(0))
        test_outputs = model(test_data[0], z_prior)
        predictions.append(test_outputs)

# 可视化训练和测试损失
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(range(0, num_epochs, num_epochs // len(test_losses)), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.show()

# 修正可视化代码，确保 x 和 y 的尺寸一致
# 可视化模型在测试集上的预测结果与真实值
plt.figure(figsize=(10, 5))
plt.scatter(train_data[0].numpy(), train_data[1].numpy(), label='Train Data', alpha=0.5)
plt.plot(test_data[0].numpy(), test_outputs.numpy(), label='Model Output', color='black')
for i in range(n_samples):
    plt.scatter(test_data[0].numpy(), predictions[i].numpy(), label=f'Prediction {i+1}', alpha=0.3)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Model Predictions vs True Values')
plt.legend()
plt.show()