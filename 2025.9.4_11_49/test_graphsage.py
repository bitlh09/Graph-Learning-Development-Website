import numpy as np
import os
from collections import defaultdict

print("开始测试GraphSAGE实现...")

# 简化的GraphSAGE类
class SimpleGraphSAGE:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 简单的权重初始化
        self.W1 = np.random.randn(input_dim * 2, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def aggregate_neighbors(self, features, adj_list, nodes):
        """聚合邻居特征"""
        agg_features = []
        for node in nodes:
            neighbors = adj_list.get(node, [])
            if len(neighbors) == 0:
                agg_features.append(np.zeros(features.shape[1]))
            else:
                # 只选择在features范围内的邻居
                valid_neighbors = [n for n in neighbors if n < len(features)]
                if len(valid_neighbors) == 0:
                    agg_features.append(np.zeros(features.shape[1]))
                else:
                    neighbor_feat = features[valid_neighbors]
                    agg_features.append(np.mean(neighbor_feat, axis=0))
        return np.array(agg_features)
    
    def forward(self, features, adj_list, nodes):
        """前向传播 - 简化为单层"""
        # 聚合邻居特征
        neighbor_feat = self.aggregate_neighbors(features, adj_list, nodes)
        self_feat = features[nodes]
        
        # 连接自身和邻居特征
        combined = np.concatenate([self_feat, neighbor_feat], axis=1)
        
        # 第一层
        h1 = self.relu(np.dot(combined, self.W1) + self.b1)
        
        # 输出层
        output = np.dot(h1, self.W2) + self.b2
        
        return output
    
    def predict(self, features, adj_list, nodes):
        """预测"""
        logits = self.forward(features, adj_list, nodes)
        probs = self.softmax(logits)
        return np.argmax(probs, axis=1)

# 加载citeseer数据的简化版本
def load_citeseer_simple():
    """简化的citeseer数据加载"""
    print("加载Citeseer数据...")
    
    content_file = "data/citeseer/citeseer.content"
    cites_file = "data/citeseer/citeseer.cites"
    
    # 检查文件是否存在
    if not os.path.exists(content_file):
        print(f"错误: 找不到文件 {content_file}")
        return None, None, None
    
    if not os.path.exists(cites_file):
        print(f"错误: 找不到文件 {cites_file}")
        return None, None, None
    
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
    
    return features, labels, adj_list, label_to_idx

# 主测试函数
def main():
    # 加载数据
    features, labels, adj_list, label_to_idx = load_citeseer_simple()
    
    if features is None:
        print("数据加载失败!")
        return
    
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
    hidden_dim = 64
    output_dim = len(label_to_idx)
    
    model = SimpleGraphSAGE(input_dim, hidden_dim, output_dim)
    
    print(f"\n模型初始化:")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏维度: {hidden_dim}")
    print(f"  输出维度: {output_dim}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    
    # 选择一小批节点进行测试
    test_nodes = train_indices[:10]
    
    try:
        logits = model.forward(features, adj_list, test_nodes)
        predictions = model.predict(features, adj_list, test_nodes)
        
        print(f"前向传播成功!")
        print(f"  输出logits形状: {logits.shape}")
        print(f"  预测标签: {predictions}")
        print(f"  真实标签: {labels[test_nodes]}")
        
        # 计算准确率
        accuracy = np.mean(predictions == labels[test_nodes])
        print(f"  随机初始化准确率: {accuracy:.4f}")
        
    except Exception as e:
        print(f"前向传播失败: {e}")
        return
    
    # 测试更大的批次
    print("\n测试更大批次...")
    try:
        large_batch = train_indices[:100]
        large_predictions = model.predict(features, adj_list, large_batch)
        large_accuracy = np.mean(large_predictions == labels[large_batch])
        print(f"  大批次准确率: {large_accuracy:.4f}")
        
        # 测试验证集
        val_predictions = model.predict(features, adj_list, val_indices)
        val_accuracy = np.mean(val_predictions == labels[val_indices])
        print(f"  验证集准确率: {val_accuracy:.4f}")
        
    except Exception as e:
        print(f"批次测试失败: {e}")
    
    print("\nGraphSAGE基础功能测试完成!")
    print("模型可以成功进行前向传播和预测")
    print("下一步可以添加反向传播训练过程")

if __name__ == "__main__":
    main()