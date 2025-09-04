# 测试NumPy版本的GCN代码
import numpy as np
import sys

print("开始测试NumPy版本的GCN实现...")

# 设置随机种子
np.random.seed(42)

print("初始化Pyodide GCN环境...")

# 创建示例图数据
num_nodes = 10
num_features = 5
num_classes = 3

# 生成节点特征 (随机)
features = np.random.randn(num_nodes, num_features).astype(np.float32)
print(f"生成特征矩阵: {features.shape}")

# 生成邻接矩阵 (小世界网络)
adj_matrix = np.eye(num_nodes, dtype=np.float32)
for i in range(num_nodes):
    for j in range(i+1, min(i+3, num_nodes)):
        if np.random.random() > 0.3:
            adj_matrix[i, j] = 1.0
            adj_matrix[j, i] = 1.0

# 度归一化
degree = np.sum(adj_matrix, axis=1)
degree_inv_sqrt = np.power(degree, -0.5)
degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
D_inv_sqrt = np.diag(degree_inv_sqrt)
adj_norm = D_inv_sqrt @ adj_matrix @ D_inv_sqrt

print(f"归一化邻接矩阵: {adj_norm.shape}")

# 生成标签
labels = np.random.randint(0, num_classes, num_nodes)
print(f"生成标签: {labels}")

# 简化的GCN模型类
class SimpleGCN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 初始化权重
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.b2 = np.zeros((1, output_dim))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x, adj):
        # 第一层GCN
        h1 = adj @ x @ self.W1 + self.b1
        h1 = self.relu(h1)
        
        # 第二层GCN
        h2 = adj @ h1 @ self.W2 + self.b2
        
        return self.softmax(h2)
    
    def cross_entropy_loss(self, y_pred, y_true):
        # 转换为one-hot
        y_one_hot = np.eye(self.output_dim)[y_true]
        return -np.mean(np.sum(y_one_hot * np.log(y_pred + 1e-8), axis=1))
    
    def train_step(self, x, adj, y_true, lr=0.01):
        # 前向传播
        h1 = adj @ x @ self.W1 + self.b1
        h1_relu = self.relu(h1)
        h2 = adj @ h1_relu @ self.W2 + self.b2
        y_pred = self.softmax(h2)
        
        # 计算损失
        loss = self.cross_entropy_loss(y_pred, y_true)
        
        # 计算准确率
        pred_labels = np.argmax(y_pred, axis=1)
        accuracy = np.mean(pred_labels == y_true)
        
        # 反向传播(简化版)
        batch_size = x.shape[0]
        y_one_hot = np.eye(self.output_dim)[y_true]
        
        # 输出层梯度
        dL_dh2 = (y_pred - y_one_hot) / batch_size
        dL_dW2 = (adj @ h1_relu).T @ dL_dh2
        dL_db2 = np.sum(dL_dh2, axis=0, keepdims=True)
        
        # 隐藏层梯度
        dL_dh1_relu = dL_dh2 @ self.W2.T @ adj.T
        dL_dh1 = dL_dh1_relu * (h1 > 0)  # ReLU导数
        dL_dW1 = (adj @ x).T @ dL_dh1
        dL_db1 = np.sum(dL_dh1, axis=0, keepdims=True)
        
        # 更新参数
        self.W1 -= lr * dL_dW1
        self.W2 -= lr * dL_dW2
        self.b1 -= lr * dL_db1
        self.b2 -= lr * dL_db2
        
        return loss, accuracy

# 创建模型
model = SimpleGCN(num_features, 8, num_classes)
print(f"创建GCN模型: {num_features} -> 8 -> {num_classes}")

# 训练参数
epochs = 20
learning_rate = 0.01

# 存储训练历史
losses = []
accuracies = []

print("开始训练...")
for epoch in range(epochs):
    loss, acc = model.train_step(features, adj_norm, labels, learning_rate)
    
    losses.append(float(loss))
    accuracies.append(float(acc))
    
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:2d}: Loss={loss:.4f}, Accuracy={acc:.4f}")

print(f"\n训练完成! 最终准确率: {accuracies[-1]:.4f}")
print(f"最终损失: {losses[-1]:.4f}")

# 最终预测
final_pred = model.forward(features, adj_norm)
pred_labels = np.argmax(final_pred, axis=1)

print(f"\n预测结果: {pred_labels}")
print(f"真实标签: {labels}")
print(f"准确预测: {np.sum(pred_labels == labels)}/{len(labels)}")

print("\n✅ NumPy版本GCN测试成功完成!")