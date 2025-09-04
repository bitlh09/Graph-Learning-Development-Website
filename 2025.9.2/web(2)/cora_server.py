#!/usr/bin/env python3
"""
Cora数据集处理服务
处理真实的Cora数据集文件，提供REST API接口
"""

import os
import pickle
import numpy as np
import scipy.sparse as sp
from flask import Flask, jsonify, request
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 数据路径
DATA_DIR = './data'

def load_pickle_file(filename):
    """加载pickle文件"""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def load_cora_data():
    """
    加载完整的Cora数据集
    返回: features, adjacency_matrix, labels, train_mask, val_mask, test_mask
    """
    # 加载各个文件
    x = load_pickle_file('ind.cora.x')          # 训练节点特征
    y = load_pickle_file('ind.cora.y')          # 训练节点标签
    tx = load_pickle_file('ind.cora.tx')        # 测试节点特征
    ty = load_pickle_file('ind.cora.ty')        # 测试节点标签
    allx = load_pickle_file('ind.cora.allx')    # 所有节点特征
    ally = load_pickle_file('ind.cora.ally')    # 所有节点标签
    graph = load_pickle_file('ind.cora.graph')  # 图结构
    
    # 读取测试节点索引
    with open(os.path.join(DATA_DIR, 'ind.cora.test.index'), 'r') as f:
        test_idx_reorder = [int(line.strip()) for line in f.readlines()]
    test_idx_range = np.sort(test_idx_reorder)
    
    # 构建特征矩阵
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    
    # 构建邻接矩阵
    adj = nx_to_scipy_sparse_matrix(graph)
    
    # 构建标签矩阵
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    # 创建训练/验证/测试掩码
    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))
    
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    
    return features.toarray(), adj.toarray(), labels, train_mask, val_mask, test_mask

def nx_to_scipy_sparse_matrix(graph):
    """将networkx图转换为scipy稀疏矩阵"""
    node_list = sorted(graph.keys())
    n_nodes = len(node_list)
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # 构建邻接矩阵
    rows, cols = [], []
    for node, neighbors in graph.items():
        node_idx = node_to_idx[node]
        for neighbor in neighbors:
            if neighbor in node_to_idx:
                neighbor_idx = node_to_idx[neighbor]
                rows.append(node_idx)
                cols.append(neighbor_idx)
    
    data = np.ones(len(rows))
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    return adj

def sample_mask(idx, l):
    """创建样本掩码"""
    mask = np.zeros(l, dtype=bool)
    mask[idx] = True
    return mask

def preprocess_features(features):
    """特征预处理：行归一化"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def preprocess_adj(adj):
    """邻接矩阵预处理：添加自环并归一化"""
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

@app.route('/api/cora/info')
def get_cora_info():
    """获取Cora数据集基本信息"""
    try:
        features, adj, labels, train_mask, val_mask, test_mask = load_cora_data()
        
        info = {
            'num_nodes': features.shape[0],
            'num_features': features.shape[1],
            'num_classes': labels.shape[1],
            'num_edges': int(np.sum(adj) // 2),  # 无向图，除以2
            'train_size': int(np.sum(train_mask)),
            'val_size': int(np.sum(val_mask)),
            'test_size': int(np.sum(test_mask)),
            'class_names': ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 
                          'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
        }
        
        return jsonify({'success': True, 'data': info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/cora/sample')
def get_cora_sample():
    """获取Cora数据集的小样本用于前端演示"""
    try:
        sample_size = int(request.args.get('size', 100))
        
        features, adj, labels, train_mask, val_mask, test_mask = load_cora_data()
        
        # 随机选择样本节点
        indices = np.random.choice(features.shape[0], sample_size, replace=False)
        
        # 提取子图
        sample_features = features[indices].tolist()
        sample_adj = adj[np.ix_(indices, indices)].tolist()
        sample_labels = np.argmax(labels[indices], axis=1).tolist()
        
        # 重新计算掩码
        sample_train_mask = train_mask[indices].tolist()
        sample_val_mask = val_mask[indices].tolist()
        sample_test_mask = test_mask[indices].tolist()
        
        data = {
            'features': sample_features,
            'adjacency': sample_adj,
            'labels': sample_labels,
            'train_mask': sample_train_mask,
            'val_mask': sample_val_mask,
            'test_mask': sample_test_mask,
            'num_nodes': sample_size,
            'num_features': features.shape[1],
            'num_classes': labels.shape[1]
        }
        
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/cora/full')
def get_cora_full():
    """获取完整的Cora数据集（压缩格式）"""
    try:
        features, adj, labels, train_mask, val_mask, test_mask = load_cora_data()
        
        # 预处理数据
        features_processed = preprocess_features(sp.csr_matrix(features)).toarray()
        adj_processed = preprocess_adj(sp.csr_matrix(adj)).toarray()
        
        # 转换为稀疏格式以减少传输量
        def to_sparse_format(matrix):
            sparse_matrix = sp.csr_matrix(matrix)
            return {
                'data': sparse_matrix.data.tolist(),
                'indices': sparse_matrix.indices.tolist(),
                'indptr': sparse_matrix.indptr.tolist(),
                'shape': sparse_matrix.shape
            }
        
        data = {
            'features': to_sparse_format(features_processed),
            'adjacency': to_sparse_format(adj_processed),
            'labels': np.argmax(labels, axis=1).tolist(),
            'train_mask': train_mask.tolist(),
            'val_mask': val_mask.tolist(),
            'test_mask': test_mask.tolist(),
            'num_nodes': features.shape[0],
            'num_features': features.shape[1],
            'num_classes': labels.shape[1]
        }
        
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/cora/visualization')
def get_cora_visualization():
    """获取用于可视化的Cora数据子集"""
    try:
        vis_size = int(request.args.get('size', 50))
        
        features, adj, labels, train_mask, val_mask, test_mask = load_cora_data()
        
        # 选择每个类别的代表节点
        label_indices = np.argmax(labels, axis=1)
        selected_indices = []
        
        for class_id in range(labels.shape[1]):
            class_nodes = np.where(label_indices == class_id)[0]
            if len(class_nodes) > 0:
                # 从每个类别中随机选择几个节点
                num_select = min(vis_size // labels.shape[1], len(class_nodes))
                selected = np.random.choice(class_nodes, num_select, replace=False)
                selected_indices.extend(selected)
        
        # 补充到指定大小
        if len(selected_indices) < vis_size:
            remaining = vis_size - len(selected_indices)
            all_indices = set(range(features.shape[0]))
            available = list(all_indices - set(selected_indices))
            additional = np.random.choice(available, min(remaining, len(available)), replace=False)
            selected_indices.extend(additional)
        
        selected_indices = np.array(selected_indices[:vis_size])
        
        # 构建可视化数据
        vis_features = features[selected_indices]
        vis_adj = adj[np.ix_(selected_indices, selected_indices)]
        vis_labels = label_indices[selected_indices]
        
        # 创建节点和边的数据
        nodes = []
        for i, idx in enumerate(selected_indices):
            nodes.append({
                'id': i,
                'original_id': int(idx),
                'label': int(vis_labels[i]),
                'class_name': ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 
                             'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'][vis_labels[i]],
                'features': vis_features[i].tolist()[:10],  # 只取前10个特征用于显示
                'is_train': bool(train_mask[idx]),
                'is_val': bool(val_mask[idx]),
                'is_test': bool(test_mask[idx])
            })
        
        edges = []
        for i in range(len(selected_indices)):
            for j in range(i + 1, len(selected_indices)):
                if vis_adj[i, j] > 0:
                    edges.append({
                        'source': i,
                        'target': j,
                        'weight': float(vis_adj[i, j])
                    })
        
        data = {
            'nodes': nodes,
            'edges': edges,
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'class_names': ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 
                          'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
        }
        
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("启动Cora数据集服务...")
    print("可用接口:")
    print("  GET /api/cora/info - 获取数据集基本信息")
    print("  GET /api/cora/sample?size=100 - 获取样本数据")
    print("  GET /api/cora/full - 获取完整数据集")
    print("  GET /api/cora/visualization?size=50 - 获取可视化数据")
    
    app.run(host='0.0.0.0', port=5001, debug=True)