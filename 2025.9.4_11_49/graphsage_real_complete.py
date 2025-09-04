#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的 GraphSAGE 实现，支持真实 Citeseer 数据集
包含完整的前向传播、反向传播和梯度下降
"""

import numpy as np
import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import time

# 设置随机种子
np.random.seed(42)
random.seed(42)

class GraphSAGEComplete:
    """完整的 GraphSAGE 实现，包含反向传播"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, aggregator='mean'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.aggregator = aggregator
        
        # 权重初始化
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
        
        # 用于存储前向传播的中间结果
        self.cache = {}
    
    def _xavier_init(self, fan_in, fan_out):
        """Xavier初始化"""
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))
    
    def _relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """ReLU的导数"""
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
                aggregated_features.append(np.zeros(node_features.shape[1]))
            else:
                neighbor_features = node_features[neighbors]
                if self.aggregator == 'mean':
                    agg_feature = np.mean(neighbor_features, axis=0)
                elif self.aggregator == 'max':
                    agg_feature = np.max(neighbor_features, axis=0)
                else:
                    agg_feature = np.mean(neighbor_features, axis=0)
                aggregated_features.append(agg_feature)
        
        return np.array(aggregated_features)
    
    def forward(self, node_features, adjacency_list, target_nodes):
        """前向传播"""
        self.cache = {'activations': [], 'z_values': [], 'combined_features': []}
        
        current_features = node_features.copy()
        self.cache['activations'].append(current_features)
        
        for layer in range(self.num_layers):
            # 聚合邻居特征
            neighbor_features = self._aggregate_neighbors(
                current_features, adjacency_list, target_nodes
            )
            
            # 获取目标节点的自身特征
            self_features = current_features[target_nodes]
            
            # 连接自身特征和聚合的邻居特征
            combined_features = np.concatenate([self_features, neighbor_features], axis=1)
            self.cache['combined_features'].append(combined_features)
            
            # 线性变换
            z = np.dot(combined_features, self.weights[layer]) + self.biases[layer]
            self.cache['z_values'].append(z)
            
            # 激活函数
            if layer < self.num_layers - 1:
                current_features = self._relu(z)
            else:
                current_features = z  # 最后一层不使用激活函数
            
            self.cache['activations'].append(current_features)
        
        return current_features
    
    def backward(self, predictions, true_labels, learning_rate=0.01):
        """反向传播"""
        batch_size = len(true_labels)
        
        # 计算输出层的梯度
        predictions_softmax = self._softmax(predictions)
        
        # 创建one-hot编码的真实标签
        y_true_onehot = np.zeros_like(predictions_softmax)
        y_true_onehot[np.arange(batch_size), true_labels] = 1
        
        # 输出层梯度
        delta = predictions_softmax - y_true_onehot
        
        # 反向传播更新权重
        for layer in reversed(range(self.num_layers)):
            # 获取当前层的输入
            layer_input = self.cache['combined_features'][layer]
            
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
                
                # 应用激活函数的导数
                z_prev = self.cache['z_values'][layer - 1]
                delta = delta_next * self._relu_derivative(z_prev)
    
    def predict(self, node_features, adjacency_list, target_nodes):
        """预测"""
        logits = self.forward(node_features, adjacency_list, target_nodes)
        probabilities = self._softmax(logits)
        return np.argmax(probabilities, axis=1)
    
    def compute_loss(self, predictions, true_labels):
        """计算交叉熵损失"""
        predictions_softmax = self._softmax(predictions)
        batch_size = predictions.shape[0]
        log_probs = -np.log(predictions_softmax[np.arange(batch_size), true_labels] + 1e-8)
        return np.mean(log_probs)


def load_citeseer_data(data_path="data/citeseer", use_subset=False, subset_size=1000):
    """加载 Citeseer 数据集"""
    print("=" * 60)
    print("📚 正在加载真实 Citeseer 数据集...")
    print("=" * 60)
    
    content_file = os.path.join(data_path, "citeseer.content")
    cites_file = os.path.join(data_path, "citeseer.cites")
    
    if not os.path.exists(content_file) or not os.path.exists(cites_file):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    # 存储节点信息
    node_ids = []
    node_features_list = []
    node_labels_list = []
    
    print("🔄 读取节点特征和标签...")
    
    # 读取节点内容
    with open(content_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            node_id = parts[0]
            features = [float(x) for x in parts[1:-1]]
            label = parts[-1]
            
            node_ids.append(node_id)
            node_features_list.append(features)
            node_labels_list.append(label)
    
    print(f"✅ 原始数据集大小: {len(node_ids)} 个节点")
    
    # 如果使用子集，进行采样
    if use_subset and len(node_ids) > subset_size:
        print(f"🎯 采样 {subset_size} 个节点作为训练子集...")
        
        # 分层采样确保每个类别都有代表
        indices_by_label = defaultdict(list)
        for i, label in enumerate(node_labels_list):
            indices_by_label[label].append(i)
        
        selected_indices = []
        samples_per_class = subset_size // len(indices_by_label)
        
        for label, indices in indices_by_label.items():
            if len(indices) < samples_per_class:
                selected_indices.extend(indices)
            else:
                selected_indices.extend(random.sample(indices, samples_per_class))
        
        # 如果还需要更多样本，随机添加
        if len(selected_indices) < subset_size:
            remaining = subset_size - len(selected_indices)
            all_indices = set(range(len(node_ids)))
            available = list(all_indices - set(selected_indices))
            selected_indices.extend(random.sample(available, min(remaining, len(available))))
        
        # 重新映射
        selected_indices = sorted(selected_indices[:subset_size])
        node_ids = [node_ids[i] for i in selected_indices]
        node_features_list = [node_features_list[i] for i in selected_indices]
        node_labels_list = [node_labels_list[i] for i in selected_indices]
        
        print(f"✅ 子集采样完成: {len(node_ids)} 个节点")
    
    # 创建节点ID到索引的映射
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # 创建标签映射
    unique_labels = sorted(list(set(node_labels_list)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # 转换标签为数值
    numeric_labels = [label_to_idx[label] for label in node_labels_list]
    
    # 转换为numpy数组
    node_features = np.array(node_features_list, dtype=np.float32)
    node_labels = np.array(numeric_labels, dtype=np.int32)
    
    print("🔄 读取图结构...")
    
    # 读取边信息
    adjacency_list = defaultdict(list)
    edge_count = 0
    
    with open(cites_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            
            cited_id, citing_id = parts
            
            # 只保留在当前节点集合中的边
            if cited_id in node_id_to_idx and citing_id in node_id_to_idx:
                cited_idx = node_id_to_idx[cited_id]
                citing_idx = node_id_to_idx[citing_id]
                
                # 创建无向图
                adjacency_list[cited_idx].append(citing_idx)
                adjacency_list[citing_idx].append(cited_idx)
                edge_count += 1
    
    # 去重
    for node in adjacency_list:
        adjacency_list[node] = list(set(adjacency_list[node]))
    
    # 特征归一化
    feature_mean = np.mean(node_features, axis=0)
    feature_std = np.std(node_features, axis=0)
    feature_std[feature_std == 0] = 1  # 避免除零
    node_features = (node_features - feature_mean) / feature_std
    
    print("=" * 60)
    print("✅ 数据加载完成!")
    print(f"📊 节点数量: {len(node_features)}")
    print(f"📊 特征维度: {node_features.shape[1]}")
    print(f"📊 类别数量: {len(label_to_idx)}")
    print(f"📊 边数量: {sum(len(neighbors) for neighbors in adjacency_list.values()) // 2}")
    print(f"📊 类别标签: {list(label_to_idx.keys())}")
    print("=" * 60)
    
    return node_features, node_labels, adjacency_list, label_to_idx, idx_to_label


def train_graphsage_complete(use_subset=True, subset_size=800):
    """完整的 GraphSAGE 训练流程"""
    
    print("🚀 GraphSAGE 完整训练流程")
    print("=" * 70)
    
    # 加载数据
    node_features, node_labels, adjacency_list, label_to_idx, idx_to_label = load_citeseer_data(
        use_subset=use_subset, subset_size=subset_size
    )
    
    # 分割数据
    num_nodes = len(node_features)
    indices = np.random.permutation(num_nodes)
    
    train_end = int(0.6 * num_nodes)
    val_end = int(0.8 * num_nodes)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    print(f"\n📊 数据分割:")
    print(f"   训练集: {len(train_indices)} 个节点")
    print(f"   验证集: {len(val_indices)} 个节点")
    print(f"   测试集: {len(test_indices)} 个节点")
    
    # 初始化模型
    input_dim = node_features.shape[1]
    hidden_dim = 64
    output_dim = len(label_to_idx)
    
    model = GraphSAGEComplete(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        aggregator='mean'
    )
    
    print(f"\n🧠 模型参数:")
    print(f"   输入维度: {input_dim}")
    print(f"   隐藏维度: {hidden_dim}")
    print(f"   输出维度: {output_dim}")
    print(f"   层数: 2")
    print(f"   聚合函数: mean")
    
    # 训练参数
    epochs = 100
    learning_rate = 0.01
    batch_size = 32
    
    print(f"\n⚙️ 训练参数:")
    print(f"   训练轮数: {epochs}")
    print(f"   学习率: {learning_rate}")
    print(f"   批次大小: {batch_size}")
    
    # 训练循环
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"\n🎯 开始训练...")
    print("=" * 70)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # 随机打乱训练数据
        np.random.shuffle(train_indices)
        
        epoch_losses = []
        epoch_preds = []
        epoch_labels = []
        
        # 批次训练
        for i in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[i:i+batch_size]
            
            # 前向传播
            predictions = model.forward(node_features, adjacency_list, batch_indices)
            batch_labels = node_labels[batch_indices]
            
            # 计算损失
            loss = model.compute_loss(predictions, batch_labels)
            epoch_losses.append(loss)
            
            # 计算预测
            pred_labels = np.argmax(model._softmax(predictions), axis=1)
            epoch_preds.extend(pred_labels)
            epoch_labels.extend(batch_labels)
            
            # 反向传播
            model.backward(predictions, batch_labels, learning_rate)
        
        # 计算训练准确率
        train_acc = np.mean(np.array(epoch_preds) == np.array(epoch_labels))
        
        # 验证集评估
        val_predictions = model.forward(node_features, adjacency_list, val_indices)
        val_pred_labels = np.argmax(model._softmax(val_predictions), axis=1)
        val_acc = np.mean(val_pred_labels == node_labels[val_indices])
        
        # 记录指标
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs}: Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} (用时: {elapsed:.1f}s)")
    
    # 测试集评估
    print("\n" + "=" * 50)
    print("🎯 最终测试...")
    
    test_predictions = model.forward(node_features, adjacency_list, test_indices)
    test_pred_labels = np.argmax(model._softmax(test_predictions), axis=1)
    test_acc = np.mean(test_pred_labels == node_labels[test_indices])
    
    print(f"✅ 测试集准确率: {test_acc:.4f}")
    
    # 各类别准确率
    print("\n📊 各类别测试准确率:")
    for class_idx, class_name in idx_to_label.items():
        class_mask = node_labels[test_indices] == class_idx
        if np.sum(class_mask) > 0:
            class_acc = np.mean(test_pred_labels[class_mask] == node_labels[test_indices][class_mask])
            print(f"   {class_name}: {class_acc:.4f} ({np.sum(class_mask)} 个样本)")
    
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
        plt.savefig('graphsage_real_data_training.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 训练曲线已保存为 'graphsage_real_data_training.png'")
        
    except Exception as e:
        print(f"绘图时出现错误: {e}")
    
    total_time = time.time() - start_time
    print("=" * 70)
    print(f"🎉 GraphSAGE 训练完成！总用时: {total_time:.1f}秒")
    print(f"🎯 最终测试准确率: {test_acc:.4f}")
    print("=" * 70)
    
    return model, test_acc


if __name__ == "__main__":
    print("🎯 GraphSAGE 在真实 Citeseer 数据集上的完整实现")
    print("💡 这个版本包含完整的前向传播、反向传播和梯度下降")
    print()
    
    print("🎯 选择训练模式:")
    print("1. 完整数据集训练 (3312 个节点，时间较长)")
    print("2. 小子集训练 (800 个节点，推荐)")
    print("3. 中等子集训练 (1500 个节点)")
    
    choice = input("请选择 (1/2/3，默认2): ").strip()
    
    if choice == "1":
        print("🚀 开始完整数据集训练...")
        model, acc = train_graphsage_complete(use_subset=False)
    elif choice == "3":
        print("🚀 开始中等子集训练...")
        model, acc = train_graphsage_complete(use_subset=True, subset_size=1500)
    else:
        print("🚀 开始小子集训练...")
        model, acc = train_graphsage_complete(use_subset=True, subset_size=800)
    
    print(f"\n🏆 最终结果: 测试准确率 {acc:.4f}")
    print("✅ 这次真的使用了真实的 Citeseer 数据集！")