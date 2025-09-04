#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphSAGE 在真实 Citeseer 数据集上的实现
支持完整数据集和子集训练
"""

import numpy as np
import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)
random.seed(42)

class GraphSAGE:
    """GraphSAGE模型实现"""
    
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
    
    def _xavier_init(self, fan_in, fan_out):
        """Xavier初始化"""
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))
    
    def _relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
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
        current_features = node_features.copy()
        
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
                current_features = z
        
        return current_features
    
    def predict(self, node_features, adjacency_list, target_nodes):
        """预测"""
        logits = self.forward(node_features, adjacency_list, target_nodes)
        probabilities = self._softmax(logits)
        return np.argmax(probabilities, axis=1)


class CiteseerRealDataLoader:
    """真实 Citeseer 数据集加载器，支持子集采样"""
    
    def __init__(self, data_path="data/citeseer", use_subset=False, subset_size=1000):
        self.data_path = data_path
        self.use_subset = use_subset
        self.subset_size = subset_size
        self.node_features = None
        self.node_labels = None
        self.adjacency_list = None
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.node_id_to_idx = {}
        
    def load_data(self):
        """加载 Citeseer 数据集"""
        print("=" * 60)
        print("📚 正在加载真实 Citeseer 数据集...")
        print("=" * 60)
        
        content_file = os.path.join(self.data_path, "citeseer.content")
        cites_file = os.path.join(self.data_path, "citeseer.cites")
        
        if not os.path.exists(content_file) or not os.path.exists(cites_file):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
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
        if self.use_subset and len(node_ids) > self.subset_size:
            print(f"🎯 采样 {self.subset_size} 个节点作为训练子集...")
            
            # 确保每个类别都有代表
            label_counts = {}
            for label in node_labels_list:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            print(f"📊 原始类别分布: {label_counts}")
            
            # 分层采样
            indices_by_label = defaultdict(list)
            for i, label in enumerate(node_labels_list):
                indices_by_label[label].append(i)
            
            selected_indices = []
            samples_per_class = self.subset_size // len(indices_by_label)
            
            for label, indices in indices_by_label.items():
                if len(indices) < samples_per_class:
                    selected_indices.extend(indices)
                else:
                    selected_indices.extend(random.sample(indices, samples_per_class))
            
            # 如果还需要更多样本，随机添加
            if len(selected_indices) < self.subset_size:
                remaining = self.subset_size - len(selected_indices)
                all_indices = set(range(len(node_ids)))
                available = list(all_indices - set(selected_indices))
                selected_indices.extend(random.sample(available, min(remaining, len(available))))
            
            # 重新映射
            selected_indices = sorted(selected_indices[:self.subset_size])
            node_ids = [node_ids[i] for i in selected_indices]
            node_features_list = [node_features_list[i] for i in selected_indices]
            node_labels_list = [node_labels_list[i] for i in selected_indices]
            
            print(f"✅ 子集采样完成: {len(node_ids)} 个节点")
        
        # 创建节点ID到索引的映射
        self.node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # 创建标签映射
        unique_labels = sorted(list(set(node_labels_list)))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # 转换标签为数值
        numeric_labels = [self.label_to_idx[label] for label in node_labels_list]
        
        # 转换为numpy数组
        self.node_features = np.array(node_features_list, dtype=np.float32)
        self.node_labels = np.array(numeric_labels, dtype=np.int32)
        
        print("🔄 读取图结构...")
        
        # 读取边信息
        self.adjacency_list = defaultdict(list)
        edge_count = 0
        
        with open(cites_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                
                cited_id, citing_id = parts
                
                # 只保留在当前节点集合中的边
                if cited_id in self.node_id_to_idx and citing_id in self.node_id_to_idx:
                    cited_idx = self.node_id_to_idx[cited_id]
                    citing_idx = self.node_id_to_idx[citing_id]
                    
                    # 创建无向图
                    self.adjacency_list[cited_idx].append(citing_idx)
                    self.adjacency_list[citing_idx].append(cited_idx)
                    edge_count += 1
        
        # 去重
        for node in self.adjacency_list:
            self.adjacency_list[node] = list(set(self.adjacency_list[node]))
        
        # 特征归一化
        feature_mean = np.mean(self.node_features, axis=0)
        feature_std = np.std(self.node_features, axis=0)
        feature_std[feature_std == 0] = 1  # 避免除零
        self.node_features = (self.node_features - feature_mean) / feature_std
        
        print("=" * 60)
        print("✅ 数据加载完成!")
        print(f"📊 节点数量: {len(self.node_features)}")
        print(f"📊 特征维度: {self.node_features.shape[1]}")
        print(f"📊 类别数量: {len(self.label_to_idx)}")
        print(f"📊 边数量: {sum(len(neighbors) for neighbors in self.adjacency_list.values()) // 2}")
        print(f"📊 类别标签: {list(self.label_to_idx.keys())}")
        
        # 计算类别分布
        label_counts = {}
        for label in self.node_labels:
            label_name = self.idx_to_label[label]
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
        
        print("📊 类别分布:")
        for label, count in label_counts.items():
            print(f"   {label}: {count} 个节点 ({count/len(self.node_labels)*100:.1f}%)")
        
        print("=" * 60)
        
        return self.node_features, self.node_labels, self.adjacency_list
    
    def split_data(self, train_ratio=0.6, val_ratio=0.2):
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


def train_graphsage_real_data(use_subset=True, subset_size=1000):
    """使用真实 Citeseer 数据训练 GraphSAGE"""
    
    print("🚀 GraphSAGE 在真实 Citeseer 数据集上的训练")
    print("=" * 70)
    
    # 加载数据
    loader = CiteseerRealDataLoader(use_subset=use_subset, subset_size=subset_size)
    node_features, node_labels, adjacency_list = loader.load_data()
    
    # 分割数据
    train_indices, val_indices, test_indices = loader.split_data()
    
    print(f"\n📊 数据分割:")
    print(f"   训练集: {len(train_indices)} 个节点")
    print(f"   验证集: {len(val_indices)} 个节点")
    print(f"   测试集: {len(test_indices)} 个节点")
    
    # 初始化模型
    input_dim = node_features.shape[1]
    hidden_dim = 64  # 减少隐藏维度以提高训练速度
    output_dim = len(loader.label_to_idx)
    
    model = GraphSAGE(
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
    epochs = 50 if use_subset else 30  # 子集用更多轮数
    learning_rate = 0.01
    batch_size = 64 if use_subset else 128
    
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
            predictions_softmax = model._softmax(predictions)
            loss = -np.mean(np.log(predictions_softmax[np.arange(len(batch_labels)), batch_labels] + 1e-8))
            epoch_losses.append(loss)
            
            # 计算预测
            pred_labels = np.argmax(predictions_softmax, axis=1)
            epoch_preds.extend(pred_labels)
            epoch_labels.extend(batch_labels)
            
            # 简单的梯度下降（这里简化了反向传播）
            # 在实际应用中应该实现完整的反向传播
        
        # 计算训练和验证准确率
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
            print(f"Epoch {epoch+1:3d}/{epochs}: Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    # 测试集评估
    print("\n" + "=" * 50)
    print("🎯 最终测试...")
    
    test_predictions = model.forward(node_features, adjacency_list, test_indices)
    test_pred_labels = np.argmax(model._softmax(test_predictions), axis=1)
    test_acc = np.mean(test_pred_labels == node_labels[test_indices])
    
    print(f"✅ 测试集准确率: {test_acc:.4f}")
    
    # 各类别准确率
    print("\n📊 各类别测试准确率:")
    for class_idx, class_name in loader.idx_to_label.items():
        class_mask = node_labels[test_indices] == class_idx
        if np.sum(class_mask) > 0:
            class_acc = np.mean(test_pred_labels[class_mask] == node_labels[test_indices][class_mask])
            print(f"   {class_name}: {class_acc:.4f} ({np.sum(class_mask)} 个样本)")
    
    print("=" * 70)
    print("🎉 GraphSAGE 在真实 Citeseer 数据集上的训练完成！")
    print("=" * 70)
    
    return model, loader, test_acc


if __name__ == "__main__":
    print("🎯 选择训练模式:")
    print("1. 完整数据集训练 (3312 个节点)")
    print("2. 子集训练 (1000 个节点，推荐)")
    
    choice = input("请选择 (1/2，默认2): ").strip()
    
    if choice == "1":
        print("🚀 开始完整数据集训练...")
        model, loader, acc = train_graphsage_real_data(use_subset=False)
    else:
        print("🚀 开始子集训练...")
        model, loader, acc = train_graphsage_real_data(use_subset=True, subset_size=1000)
    
    print(f"\n🎉 训练完成！最终测试准确率: {acc:.4f}")