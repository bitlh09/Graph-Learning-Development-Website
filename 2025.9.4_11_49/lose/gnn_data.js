/**
 * 图神经网络数据处理模块
 * 支持多种图数据集的加载、预处理和格式转换
 */

class GraphDataProcessor {
    constructor() {
        this.datasets = {
            karate: null,
            cora: null,
            citeseer: null,
            synthetic: null
        };
        this.currentDataset = null;
    }

    /**
     * 生成Karate Club数据集
     */
    generateKarateClub() {
        const nodes = [];
        const edges = [];
        const features = [];
        const labels = [];

        // Karate Club网络的边
        const karateEdges = [
            [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 10], [0, 11], [0, 12], [0, 13], [0, 17], [0, 19], [0, 21], [0, 31],
            [1, 2], [1, 3], [1, 7], [1, 13], [1, 17], [1, 19], [1, 21], [1, 30],
            [2, 3], [2, 7], [2, 8], [2, 9], [2, 13], [2, 27], [2, 28], [2, 32],
            [3, 7], [3, 12], [3, 13],
            [4, 6], [4, 10],
            [5, 6], [5, 10], [5, 16],
            [6, 16],
            [8, 30], [8, 32], [8, 33],
            [9, 33],
            [13, 33],
            [14, 32], [14, 33],
            [15, 32], [15, 33],
            [18, 32], [18, 33],
            [19, 33],
            [20, 32], [20, 33],
            [22, 32], [22, 33],
            [23, 25], [23, 27], [23, 29], [23, 32], [23, 33],
            [24, 25], [24, 27], [24, 31],
            [25, 31],
            [26, 29], [26, 33],
            [27, 33],
            [28, 31], [28, 33],
            [29, 32], [29, 33],
            [30, 32], [30, 33],
            [31, 32], [31, 33],
            [32, 33]
        ];

        // 创建节点
        for (let i = 0; i < 34; i++) {
            nodes.push({ id: i, label: i < 17 ? 0 : 1 });
            // 生成随机特征
            const feature = [];
            for (let j = 0; j < 8; j++) {
                feature.push(Math.random() * 2 - 1);
            }
            features.push(feature);
            labels.push(i < 17 ? 0 : 1);
        }

        // 创建边
        karateEdges.forEach(([source, target]) => {
            edges.push({ source, target, weight: 1.0 });
        });

        return {
            nodes,
            edges,
            features: tf.tensor2d(features),
            labels: tf.tensor1d(labels, 'int32'),
            numClasses: 2,
            numFeatures: 8
        };
    }

    /**
     * 生成合成图数据
     */
    generateSyntheticData(numNodes = 100, numClasses = 3, numFeatures = 16) {
        const nodes = [];
        const edges = [];
        const features = [];
        const labels = [];

        // 生成节点和标签
        for (let i = 0; i < numNodes; i++) {
            const label = Math.floor(i / (numNodes / numClasses));
            nodes.push({ id: i, label });
            labels.push(label);

            // 生成类别相关的特征
            const feature = [];
            for (let j = 0; j < numFeatures; j++) {
                const base = label * 2 - 1; // 类别偏移
                feature.push(base + Math.random() * 0.5 - 0.25);
            }
            features.push(feature);
        }

        // 生成边（同类节点更容易连接）
        for (let i = 0; i < numNodes; i++) {
            const numEdges = Math.floor(Math.random() * 8) + 2;
            const targets = new Set();
            
            for (let e = 0; e < numEdges; e++) {
                let target;
                if (Math.random() < 0.7) {
                    // 70%概率连接同类节点
                    const sameClassNodes = nodes.filter(n => n.label === nodes[i].label && n.id !== i);
                    if (sameClassNodes.length > 0) {
                        target = sameClassNodes[Math.floor(Math.random() * sameClassNodes.length)].id;
                    } else {
                        target = Math.floor(Math.random() * numNodes);
                    }
                } else {
                    // 30%概率连接其他节点
                    target = Math.floor(Math.random() * numNodes);
                }
                
                if (target !== i && !targets.has(target)) {
                    targets.add(target);
                    edges.push({ source: i, target, weight: 1.0 });
                }
            }
        }

        return {
            nodes,
            edges,
            features: tf.tensor2d(features),
            labels: tf.tensor1d(labels, 'int32'),
            numClasses,
            numFeatures
        };
    }

    /**
     * 生成Cora数据集的简化版本
     */
    generateCoraLike() {
        const numNodes = 200;
        const numClasses = 7;
        const numFeatures = 32;
        
        const nodes = [];
        const edges = [];
        const features = [];
        const labels = [];

        // 论文主题
        const topics = [
            'Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
            'Probabilistic_Methods', 'Reinforcement_Learning',
            'Rule_Learning', 'Theory'
        ];

        // 生成节点
        for (let i = 0; i < numNodes; i++) {
            const label = Math.floor(i / (numNodes / numClasses));
            nodes.push({ id: i, label, topic: topics[label] });
            labels.push(label);

            // 生成稀疏的词袋特征
            const feature = new Array(numFeatures).fill(0);
            const numActiveFeatures = Math.floor(Math.random() * 8) + 3;
            
            for (let j = 0; j < numActiveFeatures; j++) {
                const idx = Math.floor(Math.random() * numFeatures);
                feature[idx] = Math.random() * 0.8 + 0.2; // 0.2-1.0之间的值
            }
            features.push(feature);
        }

        // 生成引用关系（边）
        for (let i = 0; i < numNodes; i++) {
            const numCitations = Math.floor(Math.random() * 6) + 1;
            const targets = new Set();
            
            for (let c = 0; c < numCitations; c++) {
                let target;
                if (Math.random() < 0.6) {
                    // 60%概率引用同领域论文
                    const sameTopicNodes = nodes.filter(n => n.label === nodes[i].label && n.id !== i);
                    if (sameTopicNodes.length > 0) {
                        target = sameTopicNodes[Math.floor(Math.random() * sameTopicNodes.length)].id;
                    } else {
                        target = Math.floor(Math.random() * numNodes);
                    }
                } else {
                    // 40%概率引用其他领域论文
                    target = Math.floor(Math.random() * numNodes);
                }
                
                if (target !== i && !targets.has(target)) {
                    targets.add(target);
                    edges.push({ source: i, target, weight: 1.0 });
                }
            }
        }

        return {
            nodes,
            edges,
            features: tf.tensor2d(features),
            labels: tf.tensor1d(labels, 'int32'),
            numClasses,
            numFeatures,
            topics
        };
    }

    /**
     * 生成CiteSeer数据集的简化版本
     */
    generateCiteSeerLike() {
        const numNodes = 150;
        const numClasses = 6;
        const numFeatures = 24;
        
        const nodes = [];
        const edges = [];
        const features = [];
        const labels = [];

        const categories = [
            'Agents', 'AI', 'DB', 'IR', 'ML', 'HCI'
        ];

        // 生成节点
        for (let i = 0; i < numNodes; i++) {
            const label = Math.floor(i / (numNodes / numClasses));
            nodes.push({ id: i, label, category: categories[label] });
            labels.push(label);

            // 生成特征向量
            const feature = [];
            for (let j = 0; j < numFeatures; j++) {
                const base = (label - numClasses/2) * 0.5;
                feature.push(base + Math.random() * 0.8 - 0.4);
            }
            features.push(feature);
        }

        // 生成边
        for (let i = 0; i < numNodes; i++) {
            const numEdges = Math.floor(Math.random() * 5) + 2;
            const targets = new Set();
            
            for (let e = 0; e < numEdges; e++) {
                let target = Math.floor(Math.random() * numNodes);
                
                if (target !== i && !targets.has(target)) {
                    targets.add(target);
                    edges.push({ source: i, target, weight: 1.0 });
                }
            }
        }

        return {
            nodes,
            edges,
            features: tf.tensor2d(features),
            labels: tf.tensor1d(labels, 'int32'),
            numClasses,
            numFeatures,
            categories
        };
    }

    /**
     * 构建邻接矩阵
     */
    buildAdjacencyMatrix(nodes, edges) {
        const numNodes = nodes.length;
        const adj = tf.zeros([numNodes, numNodes]);
        const adjArray = Array(numNodes).fill().map(() => Array(numNodes).fill(0));

        // 填充邻接矩阵
        edges.forEach(edge => {
            adjArray[edge.source][edge.target] = edge.weight;
            adjArray[edge.target][edge.source] = edge.weight; // 无向图
        });

        return tf.tensor2d(adjArray);
    }

    /**
     * 计算度矩阵
     */
    buildDegreeMatrix(adjacencyMatrix) {
        const degrees = adjacencyMatrix.sum(1);
        return tf.diag(degrees);
    }

    /**
     * 计算归一化拉普拉斯矩阵
     */
    buildNormalizedLaplacian(adjacencyMatrix) {
        const degrees = adjacencyMatrix.sum(1);
        const degreesSqrtInv = degrees.pow(-0.5).where(
            degrees.greater(0),
            tf.zeros(degrees.shape)
        );
        
        const D_inv_sqrt = tf.diag(degreesSqrtInv);
        const I = tf.eye(adjacencyMatrix.shape[0]);
        const A_norm = adjacencyMatrix.add(I); // 添加自环
        
        return D_inv_sqrt.matMul(A_norm).matMul(D_inv_sqrt);
    }

    /**
     * 数据集分割
     */
    splitDataset(data, trainRatio = 0.6, valRatio = 0.2) {
        const numNodes = data.nodes.length;
        const indices = Array.from({length: numNodes}, (_, i) => i);
        
        // 随机打乱
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        
        const trainSize = Math.floor(numNodes * trainRatio);
        const valSize = Math.floor(numNodes * valRatio);
        
        const trainIndices = indices.slice(0, trainSize);
        const valIndices = indices.slice(trainSize, trainSize + valSize);
        const testIndices = indices.slice(trainSize + valSize);
        
        return {
            train: {
                indices: trainIndices,
                mask: this.createMask(numNodes, trainIndices)
            },
            val: {
                indices: valIndices,
                mask: this.createMask(numNodes, valIndices)
            },
            test: {
                indices: testIndices,
                mask: this.createMask(numNodes, testIndices)
            }
        };
    }

    /**
     * 创建掩码张量
     */
    createMask(totalSize, indices) {
        const mask = new Array(totalSize).fill(false);
        indices.forEach(idx => mask[idx] = true);
        return tf.tensor1d(mask, 'bool');
    }

    /**
     * 加载指定数据集
     */
    async loadDataset(datasetName, options = {}) {
        let data;
        
        switch (datasetName) {
            case 'karate':
                data = this.generateKarateClub();
                break;
            case 'cora':
                data = this.generateCoraLike();
                break;
            case 'citeseer':
                data = this.generateCiteSeerLike();
                break;
            case 'synthetic':
                data = this.generateSyntheticData(
                    options.numNodes || 100,
                    options.numClasses || 3,
                    options.numFeatures || 16
                );
                break;
            default:
                throw new Error(`Unknown dataset: ${datasetName}`);
        }

        // 构建图结构
        data.adjacencyMatrix = this.buildAdjacencyMatrix(data.nodes, data.edges);
        data.normalizedAdjacency = this.buildNormalizedLaplacian(data.adjacencyMatrix);
        
        // 数据分割
        const split = this.splitDataset(data, options.trainRatio || 0.6);
        data.split = split;
        
        this.datasets[datasetName] = data;
        this.currentDataset = data;
        
        return data;
    }

    /**
     * 获取当前数据集
     */
    getCurrentDataset() {
        return this.currentDataset;
    }

    /**
     * 数据预处理
     */
    preprocessFeatures(features) {
        // 特征归一化
        const mean = features.mean(0, true);
        const std = features.sub(mean).square().mean(0, true).sqrt().add(1e-8);
        return features.sub(mean).div(std);
    }

    // 新增：综合数据预处理，统一归一化特征并生成归一化邻接矩阵
    preprocessData(graphData) {
        // 特征归一化
        if (graphData.features) {
            graphData.features = this.preprocessFeatures(graphData.features);
        }

        // 邻接矩阵归一化（GCN 需要）
        if (graphData.adjacencyMatrix) {
            const adj = graphData.adjacencyMatrix;
            const numNodes = adj.shape[0];

            // 添加自环
            const eye = tf.eye(numNodes);
            const adjPlusI = adj.add(eye);

            // 计算 D^-0.5
            const degrees = adjPlusI.sum(1);
            const degreeInvSqrt = degrees.pow(-0.5);

            // 处理无穷值
            const safeDeg = tf.where(
                degreeInvSqrt.isFinite ? degreeInvSqrt.isFinite() : degreeInvSqrt.notEqual(tf.scalar(Infinity)),
                degreeInvSqrt,
                tf.zerosLike(degreeInvSqrt)
            );
            const degInvSqrtDiag = tf.diag(safeDeg);

            graphData.normalizedAdjacency = degInvSqrtDiag.matMul(adjPlusI).matMul(degInvSqrtDiag);
        }

        return graphData;
    }

    /**
     * 获取邻居节点
     */
    getNeighbors(nodeId, edges, maxNeighbors = null) {
        const neighbors = [];
        
        edges.forEach(edge => {
            if (edge.source === nodeId) {
                neighbors.push(edge.target);
            } else if (edge.target === nodeId) {
                neighbors.push(edge.source);
            }
        });
        
        if (maxNeighbors && neighbors.length > maxNeighbors) {
            // 随机采样
            const shuffled = neighbors.sort(() => 0.5 - Math.random());
            return shuffled.slice(0, maxNeighbors);
        }
        
        return neighbors;
    }

    /**
     * 批量采样
     */
    sampleBatch(nodeIndices, sampleSize = 10) {
        const batch = [];
        
        nodeIndices.forEach(nodeId => {
            const neighbors = this.getNeighbors(nodeId, this.currentDataset.edges, sampleSize);
            batch.push({
                nodeId,
                neighbors,
                features: this.currentDataset.features.slice([nodeId, 0], [1, -1]).squeeze()
            });
        });
        
        return batch;
    }

    /**
     * 清理资源
     */
    dispose() {
        Object.values(this.datasets).forEach(dataset => {
            if (dataset) {
                dataset.features?.dispose();
                dataset.labels?.dispose();
                dataset.adjacencyMatrix?.dispose();
                dataset.normalizedAdjacency?.dispose();
                dataset.split?.train?.mask?.dispose();
                dataset.split?.val?.mask?.dispose();
                dataset.split?.test?.mask?.dispose();
            }
        });
    }
}

// 全局数据处理器实例
const graphDataProcessor = new GraphDataProcessor();

