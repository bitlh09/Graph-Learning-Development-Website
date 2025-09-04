import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict
import random

# 设置随机种子以确保结果可重现
np.random.seed(42)
random.seed(42)

class GraphSAGE:
    """GraphSAGE模型实现"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, aggregator='mean'):
        """
        初始化GraphSAGE模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（类别数）
            num_layers: 层数
            aggregator: 聚合函数类型 ('mean', 'max', 'lstm')
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.aggregator = aggregator
        
        # 初始化权重矩阵
        self.weights = []
        self.biases = []
        
        # 第一层
        self.weights.append(self._xavier_init(input_dim * 2, hidden_dim))
        self.biases.append(np.zeros(hidden_dim))
        
        # 中间层
        for i in range(num_layers - 2):
            self.weights.append(self._xavier_init(hidden_dim * 2, hidden_dim))
            self.biases.append(np.zeros(hidden_dim))
        
        # 输出层
        if num_layers > 1:
            self.weights.append(self._xavier_init(hidden_dim * 2, output_dim))
        else:
            self.weights.append(self._xavier_init(input_dim * 2, output_dim))
        self.biases.append(np.zeros(output_dim))
        
    def _xavier_init(self, fan_in, fan_out):
        """Xavier初始化"""
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))
    
    def _relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """ReLU激活函数的导数"""
        return (x > 0).astype(float)
    
    def _softmax(self, x):
        """Softmax激活函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _aggregate_neighbors(self, node_features, adjacency_list, nodes):
        """聚合邻居节点特征"""
        aggregated_features = []
        
        for node in nodes:
            neighbors = adjacency_list.get(node, [])
            if len(neighbors) == 0:
                # 如果没有邻居，使用零向量
                aggregated_features.append(np.zeros(node_features.shape[1]))
            else:
                neighbor_features = node_features[neighbors]
                
                if self.aggregator == 'mean':
                    agg_feature = np.mean(neighbor_features, axis=0)
                elif self.aggregator == 'max':
                    agg_feature = np.max(neighbor_features, axis=0)
                else:  # 默认使用mean
                    agg_feature = np.mean(neighbor_features, axis=0)
                
                aggregated_features.append(agg_feature)
        
        return np.array(aggregated_features)
    
    def forward(self, node_features, adjacency_list, target_nodes):
        """前向传播"""
        self.activations = []  # 存储激活值用于反向传播
        
        current_features = node_features.copy()
        self.activations.append(current_features)
        
        for layer in range(self.num_layers):
            # 聚合邻居特征
            neighbor_features = self._aggregate_neighbors(
                current_features, adjacency_list, target_nodes
            )
            
            # 获取目标节点的自身特征
            self_features = current_features[target_nodes]
            
            # 连接自身特征和聚合的邻居特征
            combined_features = np.concatenate([self_features, neighbor_features], axis=1)
            
            # 线性变换
            z = np.dot(combined_features, self.weights[layer]) + self.biases[layer]
            
            # 激活函数
            if layer < self.num_layers - 1:
                current_features = self._relu(z)
            else:
                current_features = z  # 最后一层不使用激活函数
            
            self.activations.append(current_features)
        
        return current_features
    
    def backward(self, predictions, true_labels, node_features, adjacency_list, target_nodes, learning_rate=0.01):
        """反向传播"""
        batch_size = len(target_nodes)
        
        # 计算输出层的梯度
        predictions_softmax = self._softmax(predictions)
        
        # 创建one-hot编码的真实标签
        y_true_onehot = np.zeros_like(predictions_softmax)
        y_true_onehot[np.arange(batch_size), true_labels] = 1
        
        # 输出层梯度
        delta = predictions_softmax - y_true_onehot
        
        # 反向传播更新权重
        for layer in reversed(range(self.num_layers)):
            # 计算当前层的输入
            if layer == 0:
                # 第一层的输入是原始特征的连接
                neighbor_features = self._aggregate_neighbors(
                    node_features, adjacency_list, target_nodes
                )
                self_features = node_features[target_nodes]
                layer_input = np.concatenate([self_features, neighbor_features], axis=1)
            else:
                # 其他层的输入来自前一层的输出
                neighbor_features = self._aggregate_neighbors(
                    self.activations[layer], adjacency_list, target_nodes
                )
                self_features = self.activations[layer][target_nodes]
                layer_input = np.concatenate([self_features, neighbor_features], axis=1)
            
            # 计算权重梯度
            weight_grad = np.dot(layer_input.T, delta) / batch_size
            bias_grad = np.mean(delta, axis=0)
            
            # 更新权重和偏置
            self.weights[layer] -= learning_rate * weight_grad
            self.biases[layer] -= learning_rate * bias_grad
            
            # 计算下一层的梯度（如果不是第一层）
            if layer > 0:
                # 计算传递给前一层的梯度
                delta_next = np.dot(delta, self.weights[layer].T)
                
                # 分离自身特征和邻居特征的梯度
                self_grad = delta_next[:, :self.activations[layer].shape[1]]
                
                # 应用激活函数的导数
                if layer > 0:
                    delta = self_grad * self._relu_derivative(self.activations[layer][target_nodes])
    
    def predict(self, node_features, adjacency_list, target_nodes):
        """预测"""
        logits = self.forward(node_features, adjacency_list, target_nodes)
        probabilities = self._softmax(logits)
        return np.argmax(probabilities, axis=1)


class CiteseerDataset:
    """Citeseer数据集加载器"""
    
    def __init__(self, data_path="data/citeseer"):
        self.data_path = data_path
        self.node_features = None
        self.node_labels = None
        self.adjacency_list = None
        self.label_to_idx = {}
        self.idx_to_label = {}
        
    def load_data(self):
        """加载Citeseer数据集"""
        print("正在加载Citeseer数据集...")
        
        # 读取节点内容文件
        content_file = os.path.join(self.data_path, "citeseer.content")
        cites_file = os.path.join(self.data_path, "citeseer.cites")
        
        # 存储节点ID到索引的映射
        node_id_to_idx = {}
        node_features_list = []
        node_labels_list = []
        
        # 读取节点特征和标签
        with open(content_file, 'r') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split('\t')
                node_id = parts[0]
                features = [float(x) for x in parts[1:-1]]
                label = parts[-1]
                
                node_id_to_idx[node_id] = idx
                node_features_list.append(features)
                node_labels_list.append(label)
        
        # 创建标签映射
        unique_labels = list(set(node_labels_list))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # 转换标签为数值
        numeric_labels = [self.label_to_idx[label] for label in node_labels_list]
        
        # 转换为numpy数组
        self.node_features = np.array(node_features_list, dtype=np.float32)
        self.node_labels = np.array(numeric_labels, dtype=np.int32)
        
        # 读取边信息
        self.adjacency_list = defaultdict(list)
        with open(cites_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    cited_id, citing_id = parts
                    if cited_id in node_id_to_idx and citing_id in node_id_to_idx:
                        cited_idx = node_id_to_idx[cited_id]
                        citing_idx = node_id_to_idx[citing_id]
                        # 创建无向图
                        self.adjacency_list[cited_idx].append(citing_idx)
                        self.adjacency_list[citing_idx].append(cited_idx)
        
        # 去重
        for node in self.adjacency_list:
            self.adjacency_list[node] = list(set(self.adjacency_list[node]))
        
        print(f"数据加载完成!")
        print(f"节点数量: {len(self.node_features)}")
        print(f"特征维度: {self.node_features.shape[1]}")
        print(f"类别数量: {len(self.label_to_idx)}")
        print(f"边数量: {sum(len(neighbors) for neighbors in self.adjacency_list.values()) // 2}")
        print(f"类别: {list(self.label_to_idx.keys())}")
        
        return self.node_features, self.node_labels, self.adjacency_list
    
    def split_data(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """分割数据集"""
        if self.node_features is None:
            raise ValueError("数据尚未加载，请先调用load_data()方法")
        num_nodes = len(self.node_features)
        indices = np.random.permutation(num_nodes)
        
        train_end = int(train_ratio * num_nodes)
        val_end = int((train_ratio + val_ratio) * num_nodes)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        return train_indices, val_indices, test_indices


def cross_entropy_loss(predictions, true_labels):
    """交叉熵损失函数"""
    predictions_softmax = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    predictions_softmax = predictions_softmax / np.sum(predictions_softmax, axis=1, keepdims=True)
    
    batch_size = predictions.shape[0]
    log_probs = -np.log(predictions_softmax[np.arange(batch_size), true_labels] + 1e-8)
    return np.mean(log_probs)


def calculate_accuracy(predictions, true_labels):
    """计算准确率"""
    return np.mean(predictions == true_labels)


def train_graphsage():
    """训练GraphSAGE模型"""
    print("=" * 50)
    print("开始训练GraphSAGE模型")
    print("=" * 50)
    
    # 加载数据
    dataset = CiteseerDataset()
    node_features, node_labels, adjacency_list = dataset.load_data()
    
    # 分割数据
    train_indices, val_indices, test_indices = dataset.split_data()
    
    print(f"\n数据分割:")
    print(f"训练集: {len(train_indices)} 个节点")
    print(f"验证集: {len(val_indices)} 个节点")
    print(f"测试集: {len(test_indices)} 个节点")
    
    # 初始化模型
    input_dim = node_features.shape[1]
    hidden_dim = 128
    output_dim = len(dataset.label_to_idx)
    
    model = GraphSAGE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        aggregator='mean'
    )
    
    print(f"\n模型参数:")
    print(f"输入维度: {input_dim}")
    print(f"隐藏维度: {hidden_dim}")
    print(f"输出维度: {output_dim}")
    print(f"层数: 2")
    print(f"聚合函数: mean")
    
    # 训练参数
    epochs = 100
    learning_rate = 0.01
    batch_size = 256
    
    # 记录训练过程
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"\n开始训练...")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {learning_rate}")
    print(f"批次大小: {batch_size}")
    
    for epoch in range(epochs):
        # 随机打乱训练数据
        np.random.shuffle(train_indices)
        
        epoch_losses = []
        epoch_train_acc = []
        
        # 批次训练
        for i in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[i:i+batch_size]
            
            # 前向传播
            predictions = model.forward(node_features, adjacency_list, batch_indices)
            batch_labels = node_labels[batch_indices]
            
            # 计算损失
            loss = cross_entropy_loss(predictions, batch_labels)
            epoch_losses.append(loss)
            
            # 计算训练准确率
            pred_labels = model.predict(node_features, adjacency_list, batch_indices)
            acc = calculate_accuracy(pred_labels, batch_labels)
            epoch_train_acc.append(acc)
            
            # 反向传播
            model.backward(predictions, batch_labels, node_features, adjacency_list, 
                         batch_indices, learning_rate)
        
        # 计算验证准确率
        val_pred_labels = model.predict(node_features, adjacency_list, val_indices)
        val_acc = calculate_accuracy(val_pred_labels, node_labels[val_indices])
        
        # 记录指标
        avg_loss = np.mean(epoch_losses)
        avg_train_acc = np.mean(epoch_train_acc)
        
        train_losses.append(avg_loss)
        train_accuracies.append(avg_train_acc)
        val_accuracies.append(val_acc)
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Loss: {avg_loss:.4f}, "
                  f"Train Acc: {avg_train_acc:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
    
    # 测试集评估
    print("\n" + "=" * 30)
    print("训练完成，开始测试...")
    
    test_pred_labels = model.predict(node_features, adjacency_list, test_indices)
    test_acc = calculate_accuracy(test_pred_labels, node_labels[test_indices])
    
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 计算每个类别的准确率
    print("\n各类别测试准确率:")
    for class_idx, class_name in dataset.idx_to_label.items():
        class_mask = node_labels[test_indices] == class_idx
        if np.sum(class_mask) > 0:
            class_acc = calculate_accuracy(test_pred_labels[class_mask], 
                                         node_labels[test_indices][class_mask])
            print(f"{class_name}: {class_acc:.4f} ({np.sum(class_mask)} 个样本)")
    
    # 绘制训练曲线
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='训练准确率')
        plt.plot(val_accuracies, label='验证准确率')
        plt.title('准确率曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('graphsage_training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n训练曲线已保存为 'graphsage_training_curves.png'")
        
    except Exception as e:
        print(f"绘图时出现错误: {e}")
    
    print("\n" + "=" * 50)
    print("GraphSAGE训练完成！")
    print("=" * 50)
    
    return model, dataset, test_acc


if __name__ == "__main__":
    # 训练模型
    model, dataset, test_accuracy = train_graphsage()
    
    print(f"\n最终测试准确率: {test_accuracy:.4f}")