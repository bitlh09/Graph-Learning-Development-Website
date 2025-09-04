import numpy as np
import os
from collections import defaultdict
import time

print("GraphSAGE完整训练实现")
print("=" * 50)

class GraphSAGETrainer:
    """GraphSAGE模型训练器"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # 权重初始化 (Xavier初始化)
        self.W1 = np.random.randn(input_dim * 2, hidden_dim) * np.sqrt(2.0 / (input_dim * 2))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
        
        # 用于存储前向传播的中间结果
        self.cache = {}
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_backward(self, dout, x):
        """ReLU的反向传播"""
        dx = dout.copy()
        dx[x <= 0] = 0
        return dx
    
    def softmax(self, x):
        """Softmax激活函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_pred, y_true):
        """交叉熵损失"""
        m = y_pred.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)
        
        # 创建one-hot编码
        y_true_onehot = np.zeros_like(y_pred)
        y_true_onehot[np.arange(m), y_true] = 1
        
        # 计算损失
        loss = -np.sum(y_true_onehot * np.log(y_pred_clipped)) / m
        return loss
    
    def aggregate_neighbors(self, features, adj_list, nodes):
        """聚合邻居特征"""
        agg_features = []
        for node in nodes:
            neighbors = adj_list.get(node, [])
            if len(neighbors) == 0:
                agg_features.append(np.zeros(features.shape[1]))
            else:
                valid_neighbors = [n for n in neighbors if n < len(features)]
                if len(valid_neighbors) == 0:
                    agg_features.append(np.zeros(features.shape[1]))
                else:
                    neighbor_feat = features[valid_neighbors]
                    agg_features.append(np.mean(neighbor_feat, axis=0))
        return np.array(agg_features)
    
    def forward(self, features, adj_list, nodes):
        """前向传播"""
        batch_size = len(nodes)
        
        # 聚合邻居特征
        neighbor_feat = self.aggregate_neighbors(features, adj_list, nodes)
        self_feat = features[nodes]
        
        # 连接自身和邻居特征
        combined = np.concatenate([self_feat, neighbor_feat], axis=1)
        
        # 第一层
        z1 = np.dot(combined, self.W1) + self.b1
        h1 = self.relu(z1)
        
        # 输出层
        z2 = np.dot(h1, self.W2) + self.b2
        y_pred = self.softmax(z2)
        
        # 保存中间结果用于反向传播
        self.cache = {
            'combined': combined,
            'z1': z1,
            'h1': h1,
            'z2': z2,
            'y_pred': y_pred
        }
        
        return y_pred
    
    def backward(self, y_true):
        """反向传播"""
        # 获取缓存的值
        combined = self.cache['combined']
        z1 = self.cache['z1']
        h1 = self.cache['h1']
        z2 = self.cache['z2']
        y_pred = self.cache['y_pred']
        
        batch_size = y_pred.shape[0]
        
        # 创建one-hot编码的真实标签
        y_true_onehot = np.zeros_like(y_pred)
        y_true_onehot[np.arange(batch_size), y_true] = 1
        
        # 输出层的梯度
        dz2 = (y_pred - y_true_onehot) / batch_size
        
        # W2和b2的梯度
        dW2 = np.dot(h1.T, dz2)
        db2 = np.sum(dz2, axis=0)
        
        # 隐藏层的梯度
        dh1 = np.dot(dz2, self.W2.T)
        dz1 = self.relu_backward(dh1, z1)
        
        # W1和b1的梯度
        dW1 = np.dot(combined.T, dz1)
        db1 = np.sum(dz1, axis=0)
        
        # 更新权重
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def predict(self, features, adj_list, nodes):
        """预测"""
        y_pred = self.forward(features, adj_list, nodes)
        return np.argmax(y_pred, axis=1)
    
    def train_step(self, features, adj_list, nodes, labels):
        """单步训练"""
        # 前向传播
        y_pred = self.forward(features, adj_list, nodes)
        
        # 计算损失
        loss = self.cross_entropy_loss(y_pred, labels)
        
        # 反向传播
        self.backward(labels)
        
        # 计算准确率
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == labels)
        
        return loss, accuracy


def load_citeseer_data():
    """加载citeseer数据"""
    print("加载Citeseer数据集...")
    
    content_file = "data/citeseer/citeseer.content"
    cites_file = "data/citeseer/citeseer.cites"
    
    # 读取节点特征和标签
    node_id_to_idx = {}
    features_list = []
    labels_list = []
    
    with open(content_file, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            node_id = parts[0]
            features = [float(x) for x in parts[1:-1]]
            label = parts[-1]
            
            node_id_to_idx[node_id] = idx
            features_list.append(features)
            labels_list.append(label)
    
    # 创建标签映射
    unique_labels = list(set(labels_list))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_idx[label] for label in labels_list]
    
    # 转换为numpy数组
    features = np.array(features_list, dtype=np.float32)
    labels = np.array(numeric_labels, dtype=np.int32)
    
    # 特征归一化
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
    
    # 读取边信息
    adj_list = defaultdict(list)
    edge_count = 0
    
    with open(cites_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                cited_id, citing_id = parts
                if cited_id in node_id_to_idx and citing_id in node_id_to_idx:
                    cited_idx = node_id_to_idx[cited_id]
                    citing_idx = node_id_to_idx[citing_id]
                    adj_list[cited_idx].append(citing_idx)
                    adj_list[citing_idx].append(cited_idx)
                    edge_count += 1
    
    # 去重
    for node in adj_list:
        adj_list[node] = list(set(adj_list[node]))
    
    print(f"数据加载完成:")
    print(f"  节点数: {len(features)}")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  类别数: {len(unique_labels)}")
    print(f"  边数: {edge_count}")
    print(f"  类别: {unique_labels}")
    
    return features, labels, adj_list, label_to_idx, unique_labels


def main():
    """主训练函数"""
    # 设置随机种子
    np.random.seed(42)
    
    # 加载数据
    features, labels, adj_list, label_to_idx, unique_labels = load_citeseer_data()
    
    # 数据分割
    num_nodes = len(features)
    indices = np.random.permutation(num_nodes)
    
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"\n数据分割:")
    print(f"  训练集: {len(train_indices)}")
    print(f"  验证集: {len(val_indices)}")
    print(f"  测试集: {len(test_indices)}")
    
    # 初始化模型
    input_dim = features.shape[1]
    hidden_dim = 128
    output_dim = len(label_to_idx)
    learning_rate = 0.01
    
    model = GraphSAGETrainer(input_dim, hidden_dim, output_dim, learning_rate)
    
    print(f"\n模型参数:")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏维度: {hidden_dim}")
    print(f"  输出维度: {output_dim}")
    print(f"  学习率: {learning_rate}")
    
    # 训练参数
    epochs = 50
    batch_size = 32
    
    print(f"\n开始训练...")
    print(f"  轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    print("-" * 50)
    
    # 训练记录
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # 打乱训练数据
        np.random.shuffle(train_indices)
        
        epoch_losses = []
        epoch_accuracies = []
        
        # 批次训练
        for i in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[i:i+batch_size]
            batch_labels = labels[batch_indices]
            
            # 训练步骤
            loss, accuracy = model.train_step(features, adj_list, batch_indices, batch_labels)
            
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
        
        # 计算验证准确率
        val_predictions = model.predict(features, adj_list, val_indices)
        val_accuracy = np.mean(val_predictions == labels[val_indices])
        
        # 记录指标
        avg_loss = np.mean(epoch_losses)
        avg_train_accuracy = np.mean(epoch_accuracies)
        
        train_losses.append(avg_loss)
        train_accuracies.append(avg_train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1:2d}/{epochs}: Loss: {avg_loss:.4f}, "
                  f"Train Acc: {avg_train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, "
                  f"Time: {elapsed_time:.1f}s")
    
    print("-" * 50)
    print("训练完成！")
    
    # 测试集评估
    print("\n最终评估:")
    test_predictions = model.predict(features, adj_list, test_indices)
    test_accuracy = np.mean(test_predictions == labels[test_indices])
    
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 各类别准确率
    print("\n各类别测试准确率:")
    for class_idx, class_name in enumerate(unique_labels):
        class_mask = labels[test_indices] == class_idx
        if np.sum(class_mask) > 0:
            class_predictions = test_predictions[class_mask]
            class_labels = labels[test_indices][class_mask]
            class_accuracy = np.mean(class_predictions == class_labels)
            print(f"  {class_name}: {class_accuracy:.4f} ({np.sum(class_mask)} samples)")
    
    # 显示训练曲线数据
    print("\n训练曲线数据:")
    print(f"最终训练准确率: {train_accuracies[-1]:.4f}")
    print(f"最终验证准确率: {val_accuracies[-1]:.4f}")
    print(f"最佳验证准确率: {max(val_accuracies):.4f} (Epoch {np.argmax(val_accuracies)+1})")
    
    total_time = time.time() - start_time
    print(f"总训练时间: {total_time:.1f}秒")
    
    print("\n" + "=" * 50)
    print("GraphSAGE训练完成！")
    print("实现了完整的前向传播、反向传播和梯度下降")
    print("==" * 25)
    
    return model, test_accuracy, val_accuracies


if __name__ == "__main__":
    model, test_acc, val_accs = main()
    print(f"\n🎉 最终测试准确率: {test_acc:.4f}")