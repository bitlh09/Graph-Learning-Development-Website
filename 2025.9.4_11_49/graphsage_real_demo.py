#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphSAGE 使用真实 Citeseer 数据集的演示
不依赖外部库，使用纯 Python 实现
"""

import os
import random
import math
from collections import defaultdict

# 设置随机种子
random.seed(42)

def load_real_citeseer_data(data_path="data/citeseer", max_nodes=1000):
    """加载真实的 Citeseer 数据集"""
    print("=" * 60)
    print("📚 正在加载真实 Citeseer 数据集...")
    print("=" * 60)
    
    content_file = os.path.join(data_path, "citeseer.content")
    cites_file = os.path.join(data_path, "citeseer.cites")
    
    if not os.path.exists(content_file):
        print(f"❌ 数据文件不存在: {content_file}")
        return None
    
    if not os.path.exists(cites_file):
        print(f"❌ 数据文件不存在: {cites_file}")
        return None
    
    # 存储节点信息
    node_ids = []
    node_features_list = []
    node_labels_list = []
    
    print("🔄 读取节点特征和标签...")
    
    # 读取节点内容
    try:
        with open(content_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    parts = line.strip().split('\t')
                    if len(parts) < 3:
                        continue
                    
                    node_id = parts[0]
                    features = [float(x) for x in parts[1:-1]]
                    label = parts[-1]
                    
                    node_ids.append(node_id)
                    node_features_list.append(features)
                    node_labels_list.append(label)
                    
                    # 如果达到最大节点数，停止读取
                    if len(node_ids) >= max_nodes:
                        break
                        
                except Exception as e:
                    print(f"⚠️ 解析第 {line_num+1} 行时出错: {e}")
                    continue
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return None
    
    print(f"✅ 成功读取 {len(node_ids)} 个节点")
    
    if len(node_ids) == 0:
        print("❌ 没有读取到任何节点数据")
        return None
    
    # 统计特征维度
    feature_dim = len(node_features_list[0]) if node_features_list else 0
    
    # 统计类别
    unique_labels = sorted(list(set(node_labels_list)))
    label_counts = {}
    for label in node_labels_list:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"📊 特征维度: {feature_dim}")
    print(f"📊 类别数量: {len(unique_labels)}")
    print(f"📊 类别分布:")
    for label, count in label_counts.items():
        percentage = count / len(node_labels_list) * 100
        print(f"   {label}: {count} 个节点 ({percentage:.1f}%)")
    
    # 创建节点ID到索引的映射
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # 创建标签映射
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    print("🔄 读取图结构...")
    
    # 读取边信息
    adjacency_list = defaultdict(list)
    edge_count = 0
    
    try:
        with open(cites_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
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
                        
                except Exception as e:
                    if line_num < 10:  # 只显示前几个错误
                        print(f"⚠️ 解析边文件第 {line_num+1} 行时出错: {e}")
                    continue
    except Exception as e:
        print(f"❌ 读取边文件时出错: {e}")
        return None
    
    # 去重邻接表
    for node in adjacency_list:
        adjacency_list[node] = list(set(adjacency_list[node]))
    
    # 统计图结构信息
    total_edges = sum(len(neighbors) for neighbors in adjacency_list.values()) // 2
    avg_degree = total_edges * 2 / len(node_ids) if len(node_ids) > 0 else 0
    
    print("=" * 60)
    print("✅ 真实 Citeseer 数据集加载完成!")
    print(f"📊 节点数量: {len(node_ids)}")
    print(f"📊 特征维度: {feature_dim}")
    print(f"📊 类别数量: {len(unique_labels)}")
    print(f"📊 边数量: {total_edges}")
    print(f"📊 平均度数: {avg_degree:.2f}")
    print(f"📊 类别: {unique_labels}")
    print("=" * 60)
    
    return {
        'node_features': node_features_list,
        'node_labels': node_labels_list,
        'adjacency_list': adjacency_list,
        'label_to_idx': label_to_idx,
        'unique_labels': unique_labels,
        'node_ids': node_ids,
        'stats': {
            'num_nodes': len(node_ids),
            'feature_dim': feature_dim,
            'num_classes': len(unique_labels),
            'num_edges': total_edges,
            'avg_degree': avg_degree,
            'label_counts': label_counts
        }
    }

def analyze_graph_structure(data):
    """分析图结构特性"""
    print("\n🔍 图结构分析:")
    print("=" * 40)
    
    adjacency_list = data['adjacency_list']
    num_nodes = data['stats']['num_nodes']
    
    # 度数分布
    degrees = [len(adjacency_list.get(i, [])) for i in range(num_nodes)]
    
    if degrees:
        max_degree = max(degrees)
        min_degree = min(degrees)
        avg_degree = sum(degrees) / len(degrees)
        
        print(f"📊 度数统计:")
        print(f"   最大度数: {max_degree}")
        print(f"   最小度数: {min_degree}")
        print(f"   平均度数: {avg_degree:.2f}")
        
        # 度数分布
        degree_counts = {}
        for degree in degrees:
            degree_counts[degree] = degree_counts.get(degree, 0) + 1
        
        print(f"📊 度数分布 (前10个):")
        sorted_degrees = sorted(degree_counts.items())[:10]
        for degree, count in sorted_degrees:
            print(f"   度数 {degree}: {count} 个节点")
    
    # 连通性分析
    isolated_nodes = sum(1 for i in range(num_nodes) if len(adjacency_list.get(i, [])) == 0)
    print(f"📊 孤立节点: {isolated_nodes} 个 ({isolated_nodes/num_nodes*100:.1f}%)")
    
    return degrees

def simple_graphsage_demo(data, target_node=0):
    """简单的 GraphSAGE 前向传播演示"""
    print("\n🧠 GraphSAGE 前向传播演示:")
    print("=" * 40)
    
    node_features = data['node_features']
    adjacency_list = data['adjacency_list']
    node_labels = data['node_labels']
    label_to_idx = data['label_to_idx']
    
    if target_node >= len(node_features):
        target_node = 0
    
    # 获取目标节点信息
    target_features = node_features[target_node]
    target_label = node_labels[target_node]
    neighbors = adjacency_list.get(target_node, [])
    
    print(f"🎯 目标节点: {target_node}")
    print(f"📊 节点标签: {target_label}")
    print(f"📊 特征维度: {len(target_features)}")
    print(f"📊 邻居数量: {len(neighbors)}")
    
    if len(neighbors) > 0:
        print(f"📊 邻居节点: {neighbors[:5]}{'...' if len(neighbors) > 5 else ''}")
        
        # 计算邻居特征的平均值 (简化的聚合函数)
        neighbor_features_sum = [0] * len(target_features)
        for neighbor_idx in neighbors:
            if neighbor_idx < len(node_features):
                neighbor_feat = node_features[neighbor_idx]
                for i in range(len(neighbor_feat)):
                    neighbor_features_sum[i] += neighbor_feat[i]
        
        # 计算平均值
        neighbor_features_avg = [x / len(neighbors) for x in neighbor_features_sum]
        
        print(f"📊 目标节点特征前5维: {target_features[:5]}")
        print(f"📊 聚合邻居特征前5维: {neighbor_features_avg[:5]}")
        
        # 简化的特征连接 (只取前5维演示)
        combined_features = target_features[:5] + neighbor_features_avg[:5]
        print(f"📊 连接后特征 (前10维): {combined_features}")
        
    else:
        print("⚠️ 该节点没有邻居节点")
    
    return target_features, neighbors

def demonstrate_real_data_usage():
    """演示真实数据的使用"""
    print("🎯 GraphSAGE 真实数据使用演示")
    print("💡 本演示展示如何使用真实的 Citeseer 数据集而非模拟数据")
    print()
    
    # 加载真实数据 (限制节点数以提高演示速度)
    data = load_real_citeseer_data(max_nodes=1000)
    
    if data is None:
        print("❌ 无法加载数据集，请检查数据文件是否存在")
        return
    
    # 分析图结构
    degrees = analyze_graph_structure(data)
    
    # 演示 GraphSAGE 前向传播
    target_nodes = [0, 10, 50, 100, 200]  # 选择几个节点进行演示
    
    for target_node in target_nodes:
        if target_node < data['stats']['num_nodes']:
            print(f"\n" + "="*50)
            simple_graphsage_demo(data, target_node)
    
    print("\n" + "="*70)
    print("✅ 演示完成!")
    print("💡 主要区别:")
    print("   🔸 使用真实的 Citeseer 学术论文数据")
    print("   🔸 真实的引用关系网络结构")
    print("   🔸 真实的词汇特征 (3703 维)")
    print("   🔸 真实的研究领域标签 (6 个类别)")
    print("\n   而之前的模拟数据是:")
    print("   🔸 随机生成的特征向量")
    print("   🔸 随机生成的图结构")
    print("   🔸 随机分配的标签")
    print()
    print("🚀 要使用完整的机器学习训练，需要:")
    print("   1. 安装 numpy 进行矩阵运算")
    print("   2. 实现完整的反向传播算法")
    print("   3. 使用批量梯度下降训练")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_real_data_usage()