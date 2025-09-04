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
        return np.array(agg_features)\n    \n    def forward(self, features, adj_list, nodes):\n        \"\"\"前向传播\"\"\"\n        batch_size = len(nodes)\n        \n        # 聚合邻居特征\n        neighbor_feat = self.aggregate_neighbors(features, adj_list, nodes)\n        self_feat = features[nodes]\n        \n        # 连接自身和邻居特征\n        combined = np.concatenate([self_feat, neighbor_feat], axis=1)\n        \n        # 第一层\n        z1 = np.dot(combined, self.W1) + self.b1\n        h1 = self.relu(z1)\n        \n        # 输出层\n        z2 = np.dot(h1, self.W2) + self.b2\n        y_pred = self.softmax(z2)\n        \n        # 保存中间结果用于反向传播\n        self.cache = {\n            'combined': combined,\n            'z1': z1,\n            'h1': h1,\n            'z2': z2,\n            'y_pred': y_pred\n        }\n        \n        return y_pred\n    \n    def backward(self, y_true):\n        \"\"\"反向传播\"\"\"\n        # 获取缓存的值\n        combined = self.cache['combined']\n        z1 = self.cache['z1']\n        h1 = self.cache['h1']\n        z2 = self.cache['z2']\n        y_pred = self.cache['y_pred']\n        \n        batch_size = y_pred.shape[0]\n        \n        # 创建one-hot编码的真实标签\n        y_true_onehot = np.zeros_like(y_pred)\n        y_true_onehot[np.arange(batch_size), y_true] = 1\n        \n        # 输出层的梯度\n        dz2 = (y_pred - y_true_onehot) / batch_size\n        \n        # W2和b2的梯度\n        dW2 = np.dot(h1.T, dz2)\n        db2 = np.sum(dz2, axis=0)\n        \n        # 隐藏层的梯度\n        dh1 = np.dot(dz2, self.W2.T)\n        dz1 = self.relu_backward(dh1, z1)\n        \n        # W1和b1的梯度\n        dW1 = np.dot(combined.T, dz1)\n        db1 = np.sum(dz1, axis=0)\n        \n        # 更新权重\n        self.W2 -= self.learning_rate * dW2\n        self.b2 -= self.learning_rate * db2\n        self.W1 -= self.learning_rate * dW1\n        self.b1 -= self.learning_rate * db1\n    \n    def predict(self, features, adj_list, nodes):\n        \"\"\"预测\"\"\"\n        y_pred = self.forward(features, adj_list, nodes)\n        return np.argmax(y_pred, axis=1)\n    \n    def train_step(self, features, adj_list, nodes, labels):\n        \"\"\"单步训练\"\"\"\n        # 前向传播\n        y_pred = self.forward(features, adj_list, nodes)\n        \n        # 计算损失\n        loss = self.cross_entropy_loss(y_pred, labels)\n        \n        # 反向传播\n        self.backward(labels)\n        \n        # 计算准确率\n        predictions = np.argmax(y_pred, axis=1)\n        accuracy = np.mean(predictions == labels)\n        \n        return loss, accuracy


def load_citeseer_data():\n    \"\"\"加载citeseer数据\"\"\"\n    print(\"加载Citeseer数据集...\")\n    \n    content_file = \"data/citeseer/citeseer.content\"\n    cites_file = \"data/citeseer/citeseer.cites\"\n    \n    # 读取节点特征和标签\n    node_id_to_idx = {}\n    features_list = []\n    labels_list = []\n    \n    with open(content_file, 'r') as f:\n        for idx, line in enumerate(f):\n            parts = line.strip().split('\\t')\n            node_id = parts[0]\n            features = [float(x) for x in parts[1:-1]]\n            label = parts[-1]\n            \n            node_id_to_idx[node_id] = idx\n            features_list.append(features)\n            labels_list.append(label)\n    \n    # 创建标签映射\n    unique_labels = list(set(labels_list))\n    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}\n    numeric_labels = [label_to_idx[label] for label in labels_list]\n    \n    # 转换为numpy数组\n    features = np.array(features_list, dtype=np.float32)\n    labels = np.array(numeric_labels, dtype=np.int32)\n    \n    # 特征归一化\n    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)\n    \n    # 读取边信息\n    adj_list = defaultdict(list)\n    edge_count = 0\n    \n    with open(cites_file, 'r') as f:\n        for line in f:\n            parts = line.strip().split('\\t')\n            if len(parts) == 2:\n                cited_id, citing_id = parts\n                if cited_id in node_id_to_idx and citing_id in node_id_to_idx:\n                    cited_idx = node_id_to_idx[cited_id]\n                    citing_idx = node_id_to_idx[citing_id]\n                    adj_list[cited_idx].append(citing_idx)\n                    adj_list[citing_idx].append(cited_idx)\n                    edge_count += 1\n    \n    # 去重\n    for node in adj_list:\n        adj_list[node] = list(set(adj_list[node]))\n    \n    print(f\"数据加载完成:\")\n    print(f\"  节点数: {len(features)}\")\n    print(f\"  特征维度: {features.shape[1]}\")\n    print(f\"  类别数: {len(unique_labels)}\")\n    print(f\"  边数: {edge_count}\")\n    print(f\"  类别: {unique_labels}\")\n    \n    return features, labels, adj_list, label_to_idx, unique_labels


def main():\n    \"\"\"主训练函数\"\"\"\n    # 设置随机种子\n    np.random.seed(42)\n    \n    # 加载数据\n    features, labels, adj_list, label_to_idx, unique_labels = load_citeseer_data()\n    \n    # 数据分割\n    num_nodes = len(features)\n    indices = np.random.permutation(num_nodes)\n    \n    train_size = int(0.6 * num_nodes)\n    val_size = int(0.2 * num_nodes)\n    \n    train_indices = indices[:train_size]\n    val_indices = indices[train_size:train_size + val_size]\n    test_indices = indices[train_size + val_size:]\n    \n    print(f\"\\n数据分割:\")\n    print(f\"  训练集: {len(train_indices)}\")\n    print(f\"  验证集: {len(val_indices)}\")\n    print(f\"  测试集: {len(test_indices)}\")\n    \n    # 初始化模型\n    input_dim = features.shape[1]\n    hidden_dim = 128\n    output_dim = len(label_to_idx)\n    learning_rate = 0.01\n    \n    model = GraphSAGETrainer(input_dim, hidden_dim, output_dim, learning_rate)\n    \n    print(f\"\\n模型参数:\")\n    print(f\"  输入维度: {input_dim}\")\n    print(f\"  隐藏维度: {hidden_dim}\")\n    print(f\"  输出维度: {output_dim}\")\n    print(f\"  学习率: {learning_rate}\")\n    \n    # 训练参数\n    epochs = 50\n    batch_size = 32\n    \n    print(f\"\\n开始训练...\")\n    print(f\"  轮数: {epochs}\")\n    print(f\"  批次大小: {batch_size}\")\n    print(\"-\" * 50)\n    \n    # 训练记录\n    train_losses = []\n    train_accuracies = []\n    val_accuracies = []\n    \n    start_time = time.time()\n    \n    for epoch in range(epochs):\n        # 打乱训练数据\n        np.random.shuffle(train_indices)\n        \n        epoch_losses = []\n        epoch_accuracies = []\n        \n        # 批次训练\n        for i in range(0, len(train_indices), batch_size):\n            batch_indices = train_indices[i:i+batch_size]\n            batch_labels = labels[batch_indices]\n            \n            # 训练步骤\n            loss, accuracy = model.train_step(features, adj_list, batch_indices, batch_labels)\n            \n            epoch_losses.append(loss)\n            epoch_accuracies.append(accuracy)\n        \n        # 计算验证准确率\n        val_predictions = model.predict(features, adj_list, val_indices)\n        val_accuracy = np.mean(val_predictions == labels[val_indices])\n        \n        # 记录指标\n        avg_loss = np.mean(epoch_losses)\n        avg_train_accuracy = np.mean(epoch_accuracies)\n        \n        train_losses.append(avg_loss)\n        train_accuracies.append(avg_train_accuracy)\n        val_accuracies.append(val_accuracy)\n        \n        # 打印进度\n        if (epoch + 1) % 5 == 0 or epoch == 0:\n            elapsed_time = time.time() - start_time\n            print(f\"Epoch {epoch+1:2d}/{epochs}: Loss: {avg_loss:.4f}, \"\n                  f\"Train Acc: {avg_train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, \"\n                  f\"Time: {elapsed_time:.1f}s\")\n    \n    print(\"-\" * 50)\n    print(\"训练完成！\")\n    \n    # 测试集评估\n    print(\"\\n最终评估:\")\n    test_predictions = model.predict(features, adj_list, test_indices)\n    test_accuracy = np.mean(test_predictions == labels[test_indices])\n    \n    print(f\"测试集准确率: {test_accuracy:.4f}\")\n    \n    # 各类别准确率\n    print(\"\\n各类别测试准确率:\")\n    for class_idx, class_name in enumerate(unique_labels):\n        class_mask = labels[test_indices] == class_idx\n        if np.sum(class_mask) > 0:\n            class_predictions = test_predictions[class_mask]\n            class_labels = labels[test_indices][class_mask]\n            class_accuracy = np.mean(class_predictions == class_labels)\n            print(f\"  {class_name}: {class_accuracy:.4f} ({np.sum(class_mask)} samples)\")\n    \n    # 显示训练曲线数据\n    print(\"\\n训练曲线数据:\")\n    print(f\"最终训练准确率: {train_accuracies[-1]:.4f}\")\n    print(f\"最终验证准确率: {val_accuracies[-1]:.4f}\")\n    print(f\"最佳验证准确率: {max(val_accuracies):.4f} (Epoch {np.argmax(val_accuracies)+1})\")\n    \n    total_time = time.time() - start_time\n    print(f\"总训练时间: {total_time:.1f}秒\")\n    \n    print(\"\\n\" + \"=\" * 50)\n    print(\"GraphSAGE训练完成！\")\n    print(\"实现了完整的前向传播、反向传播和梯度下降\")\n    print(\"=\" * 50)\n    \n    return model, test_accuracy, val_accuracies\n\n\nif __name__ == \"__main__\":\n    model, test_acc, val_accs = main()\n    print(f\"\\n🎉 最终测试准确率: {test_acc:.4f}\")