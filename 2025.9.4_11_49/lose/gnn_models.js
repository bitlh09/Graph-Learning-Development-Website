/**
 * 图神经网络模型实现
 * 基于TensorFlow.js的GCN、GAT、GraphSAGE算法实现
 */

/**
 * 图卷积网络 (GCN) 实现
 */
class GCNModel {
    constructor(config) {
        this.config = {
            inputDim: config.inputDim,
            hiddenDim: config.hiddenDim || 16,
            outputDim: config.outputDim,
            dropout: config.dropout || 0.5,
            learningRate: config.learningRate || 0.01,
            ...config
        };
        
        this.model = null;
        this.optimizer = null;
        this.isTraining = false;
        this.trainingHistory = {
            loss: [],
            accuracy: [],
            valLoss: [],
            valAccuracy: []
        };
        // 训练权重
        this.weights = null;
    }

    /**
     * 构建GCN模型（初始化权重与优化器）
     */
    buildModel() {
        // 初始化两层 GCN 的权重（XW -> A·(XW)）
        if (this.weights) {
            // 清理旧权重
            Object.values(this.weights).forEach(w => w.dispose && w.dispose());
        }
        this.weights = {
            gcn1_W: tf.variable(tf.randomNormal([this.config.inputDim, this.config.hiddenDim])),
            gcn2_W: tf.variable(tf.randomNormal([this.config.hiddenDim, this.config.outputDim])),
        };
    }

    /**
     * 单层 GCN 的计算：A · (X · W)
     * 注意：features/adjacency 必须是 Tensor，不能是 SymbolicTensor
     */
    gcnLayerForward(x, adjacency, W) {
        // XW
        const transformed = tf.matMul(x, W);
        // A(XW)
        const out = tf.matMul(adjacency, transformed);
        return out;
    }

    /**
     * 前向传播：两层 GCN + Softmax
     */
    forward(features, adjacency, {training = false} = {}) {
        return tf.tidy(() => {
            // 第1层
            let x = this.gcnLayerForward(features, adjacency, this.weights.gcn1_W);
            x = tf.relu(x);
            // 可选 Dropout（仅在训练时）
            if (training && this.config.dropout && this.config.dropout > 0) {
                // 使用层实现以便返回 Tensor
                x = tf.layers.dropout({rate: this.config.dropout}).apply(x);
            }
            // 第2层
            x = this.gcnLayerForward(x, adjacency, this.weights.gcn2_W);
            // 输出 softmax（用于损失计算前一般用 logits；这里保持原行为）
            return tf.softmax(x);
        });
    }

    /**
     * 预测（推理）
     */
    predict(features, adjacency) {
        return this.forward(features, adjacency, {training: false});
    }

    /**
     * 训练步骤
     */
    async trainStep(features, adjacency, labels, trainMask) {
        const f = () => {
            // 使用 logits 计算交叉熵更稳定，这里先按 softmax 后 crossEntropy 与原实现保持一致
            const predictions = this.forward(features, adjacency, {training: true});
            
            // 只计算训练集的损失
            if (!trainMask) {
                throw new Error('trainMask is required but got null or undefined');
            }
            const trainIndices = tf.where(trainMask);
            const trainPredictions = tf.gather(predictions, trainIndices.squeeze());
            const trainLabels = tf.gather(labels, trainIndices.squeeze());
            trainIndices.dispose();
            
            const loss = tf.losses.softmaxCrossEntropy(trainLabels, trainPredictions);
            return loss;
        };
        
        const {value: loss, grads} = tf.variableGrads(f);
        this.optimizer.applyGradients(grads);
        
        // 释放 grads 的中间张量
        Object.values(grads).forEach(g => g.dispose && g.dispose());
        
        return loss;
    }

    /**
     * 评估模型
     */
    evaluate(features, adjacency, labels, mask) {
        return tf.tidy(() => {
            const predictions = this.forward(features, adjacency, {training: false});
            if (!mask) {
                throw new Error('mask is required but got null or undefined');
            }
            const maskIndices = tf.where(mask);
            const maskedPredictions = tf.gather(predictions, maskIndices.squeeze());
            const maskedLabels = tf.gather(labels, maskIndices.squeeze());
            maskIndices.dispose();
            
            const loss = tf.losses.softmaxCrossEntropy(maskedLabels, maskedPredictions);
            
            const predictedClasses = tf.argMax(maskedPredictions, 1);
            const trueClasses = tf.argMax(maskedLabels, 1);
            const accuracy = tf.mean(tf.cast(tf.equal(predictedClasses, trueClasses), 'float32'));
            
            return {loss: loss.dataSync()[0], accuracy: accuracy.dataSync()[0]};
        });
    }

    /**
     * 训练模型
     */
    async train(data, epochs = 200, callbacks = {}) {
        this.isTraining = true;
        const startTime = Date.now();
        
        // 准备数据
        const features = data.features;
        const adjacency = data.normalizedAdjacency;
        
        // 检查标签是否存在
        if (!data.labels) {
            throw new Error('训练数据中缺少标签信息');
        }
        
        const labels = tf.oneHot(data.labels, data.numClasses);
        
        for (let epoch = 0; epoch < epochs && this.isTraining; epoch++) {
            // 训练步骤
            const loss = await this.trainStep(
                features, adjacency, labels, data.split.train.mask
            );
            
            // 验证
            const trainMetrics = this.evaluate(
                features, adjacency, labels, data.split.train.mask
            );
            const valMetrics = this.evaluate(
                features, adjacency, labels, data.split.val.mask
            );
            
            // 记录历史
            this.trainingHistory.loss.push(trainMetrics.loss);
            this.trainingHistory.accuracy.push(trainMetrics.accuracy);
            this.trainingHistory.valLoss.push(valMetrics.loss);
            this.trainingHistory.valAccuracy.push(valMetrics.accuracy);
            
            // 回调
            if (callbacks.onEpochEnd) {
                await callbacks.onEpochEnd(epoch, {
                    loss: trainMetrics.loss,
                    accuracy: trainMetrics.accuracy,
                    valLoss: valMetrics.loss,
                    valAccuracy: valMetrics.accuracy,
                    time: Date.now() - startTime
                });
            }
            
            // 让出控制权
            await tf.nextFrame();
        }
        
        this.isTraining = false;
        return this.trainingHistory;
    }

    /**
     * 停止训练
     */
    stopTraining() {
        this.isTraining = false;
    }

    /**
     * 重置模型
     */
    reset() {
        if (this.model && this.model.dispose) {
            this.model.dispose();
        }
        if (this.weights) {
            Object.values(this.weights).forEach(w => w.dispose && w.dispose());
            this.weights = null;
        }
        this.trainingHistory = {
            loss: [],
            accuracy: [],
            valLoss: [],
            valAccuracy: []
        };
    }
}

/**
 * 图注意力网络 (GAT) 实现
 */
class GATModel {
    constructor(config) {
        this.config = {
            inputDim: config.inputDim,
            hiddenDim: config.hiddenDim || 8,
            outputDim: config.outputDim,
            numHeads: config.numHeads || 8,
            dropout: config.dropout || 0.6,
            learningRate: config.learningRate || 0.005,
            ...config
        };
        
        this.model = null;
        this.optimizer = null;
        this.isTraining = false;
        this.trainingHistory = {
            loss: [],
            accuracy: [],
            valLoss: [],
            valAccuracy: []
        };
        this.attentionWeights = null;
    }

    /**
     * 构建GAT模型
     */
    buildModel() {
        const featuresInput = tf.input({shape: [this.config.inputDim], name: 'features'});
        const adjacencyInput = tf.input({shape: [null, null], name: 'adjacency'});
        
        // GAT层1 (多头注意力)
        let x = this.gatLayer(
            featuresInput, adjacencyInput, 
            this.config.hiddenDim, this.config.numHeads, 'gat1'
        );
        x = tf.layers.activation({activation: 'elu'}).apply(x);
        x = tf.layers.dropout({rate: this.config.dropout}).apply(x);
        
        // GAT层2 (输出层，单头)
        x = this.gatLayer(x, adjacencyInput, this.config.outputDim, 1, 'gat2');
        const output = tf.layers.activation({activation: 'softmax'}).apply(x);
        
        this.model = tf.model({
            inputs: [featuresInput, adjacencyInput],
            outputs: output
        });
        
        this.optimizer = tf.train.adam(this.config.learningRate);
        
        return this.model;
    }

    /**
     * GAT层实现
     */
    gatLayer(features, adjacency, outputDim, numHeads, name) {
        return tf.tidy(() => {
            const inputDim = features.shape[features.shape.length - 1];
            const headDim = Math.floor(outputDim / numHeads);
            
            const heads = [];
            
            for (let h = 0; h < numHeads; h++) {
                // 权重矩阵
                const W = tf.variable(
                    tf.randomNormal([inputDim, headDim], 0, 0.1),
                    true,
                    `${name}_W_${h}`
                );
                
                // 注意力权重
                const a = tf.variable(
                    tf.randomNormal([2 * headDim, 1], 0, 0.1),
                    true,
                    `${name}_a_${h}`
                );
                
                // 特征变换
                const h_transformed = tf.matMul(features, W);
                
                // 计算注意力分数
                const attention = this.computeAttention(h_transformed, a, adjacency);
                
                // 应用注意力权重
                const head_output = tf.matMul(attention, h_transformed);
                heads.push(head_output);
            }
            
            // 多头拼接或平均
            if (numHeads === 1) {
                return heads[0];
            } else {
                return tf.concat(heads, -1);
            }
        });
    }

    /**
     * 计算注意力权重
     */
    computeAttention(features, attentionWeights, adjacency) {
        return tf.tidy(() => {
            const numNodes = features.shape[0];
            const featureDim = features.shape[1];
            
            // 计算所有节点对的注意力分数
            const features_i = tf.expandDims(features, 1).tile([1, numNodes, 1]);
            const features_j = tf.expandDims(features, 0).tile([numNodes, 1, 1]);
            const concat_features = tf.concat([features_i, features_j], -1);
            
            // 注意力分数
            const logits = tf.matMul(
                tf.reshape(concat_features, [-1, 2 * featureDim]),
                attentionWeights
            );
            const logitsReshaped = tf.reshape(logits, [numNodes, numNodes]);
            
            // 应用LeakyReLU
            const alpha = tf.leakyRelu(logitsReshaped, 0.2);
            
            // 掩码（只考虑邻接的节点）
            const masked_alpha = tf.where(
                adjacency.greater(0),
                alpha,
                tf.fill(alpha.shape, -1e9)
            );
            
            // Softmax归一化
            const attention = tf.softmax(masked_alpha, 1);
            
            // 保存注意力权重用于可视化
            this.attentionWeights = attention;
            
            return attention;
        });
    }

    /**
     * 训练步骤
     */
    async trainStep(features, adjacency, labels, trainMask) {
        return tf.tidy(() => {
            const f = () => {
                const predictions = this.model.apply([features, adjacency]);
                if (!trainMask) {
                    throw new Error('trainMask is required but got null or undefined');
                }
                const trainIndices = tf.where(trainMask);
                const trainPredictions = tf.gather(predictions, trainIndices.squeeze());
                const trainLabels = tf.gather(labels, trainIndices.squeeze());
                trainIndices.dispose();
                const loss = tf.losses.softmaxCrossEntropy(trainLabels, trainPredictions);
                return loss;
            };
            
            const {value: loss, grads} = tf.variableGrads(f);
            this.optimizer.applyGradients(grads);
            
            return loss;
        });
    }

    /**
     * 评估模型
     */
    evaluate(features, adjacency, labels, mask) {
        return tf.tidy(() => {
            const predictions = this.model.predict([features, adjacency]);
            if (!mask) {
                throw new Error('mask is required but got null or undefined');
            }
            const maskIndices = tf.where(mask);
            const maskedPredictions = tf.gather(predictions, maskIndices.squeeze());
            const maskedLabels = tf.gather(labels, maskIndices.squeeze());
            maskIndices.dispose();
            
            const loss = tf.losses.softmaxCrossEntropy(maskedLabels, maskedPredictions);
            
            const predictedClasses = tf.argMax(maskedPredictions, 1);
            const trueClasses = tf.argMax(maskedLabels, 1);
            const accuracy = tf.mean(tf.cast(tf.equal(predictedClasses, trueClasses), 'float32'));
            
            return {loss: loss.dataSync()[0], accuracy: accuracy.dataSync()[0]};
        });
    }

    /**
     * 训练模型
     */
    async train(data, epochs = 200, callbacks = {}) {
        this.isTraining = true;
        const startTime = Date.now();
        
        const features = data.features;
        const adjacency = data.normalizedAdjacency;
        
        // 检查标签是否存在
        if (!data.labels) {
            throw new Error('训练数据中缺少标签信息');
        }
        
        const labels = tf.oneHot(data.labels, data.numClasses);
        
        for (let epoch = 0; epoch < epochs && this.isTraining; epoch++) {
            const loss = await this.trainStep(
                features, adjacency, labels, data.split.train.mask
            );
            
            const trainMetrics = this.evaluate(
                features, adjacency, labels, data.split.train.mask
            );
            const valMetrics = this.evaluate(
                features, adjacency, labels, data.split.val.mask
            );
            
            this.trainingHistory.loss.push(trainMetrics.loss);
            this.trainingHistory.accuracy.push(trainMetrics.accuracy);
            this.trainingHistory.valLoss.push(valMetrics.loss);
            this.trainingHistory.valAccuracy.push(valMetrics.accuracy);
            
            if (callbacks.onEpochEnd) {
                await callbacks.onEpochEnd(epoch, {
                    loss: trainMetrics.loss,
                    accuracy: trainMetrics.accuracy,
                    valLoss: valMetrics.loss,
                    valAccuracy: valMetrics.accuracy,
                    time: Date.now() - startTime
                });
            }
            
            await tf.nextFrame();
        }
        
        this.isTraining = false;
        return this.trainingHistory;
    }

    /**
     * 获取注意力权重
     */
    getAttentionWeights() {
        return this.attentionWeights;
    }

    /**
     * 停止训练
     */
    stopTraining() {
        this.isTraining = false;
    }

    /**
     * 重置模型
     */
    reset() {
        if (this.model) {
            this.model.dispose();
        }
        this.trainingHistory = {
            loss: [],
            accuracy: [],
            valLoss: [],
            valAccuracy: []
        };
        this.attentionWeights = null;
    }
}

/**
 * GraphSAGE 实现
 */
class GraphSAGEModel {
    constructor(config) {
        this.config = {
            inputDim: config.inputDim,
            hiddenDim: config.hiddenDim || 16,
            outputDim: config.outputDim,
            numLayers: config.numLayers || 2,
            sampleSize: config.sampleSize || 10,
            aggregator: config.aggregator || 'mean',
            learningRate: config.learningRate || 0.01,
            ...config
        };
        
        this.model = null;
        this.optimizer = null;
        this.isTraining = false;
        this.trainingHistory = {
            loss: [],
            accuracy: [],
            valLoss: [],
            valAccuracy: []
        };
        this.nodeEmbeddings = null;
    }

    /**
     * 构建GraphSAGE模型
     */
    buildModel() {
        const featuresInput = tf.input({shape: [this.config.inputDim], name: 'features'});
        const neighborsInput = tf.input({shape: [null, this.config.inputDim], name: 'neighbors'});
        
        let x = featuresInput;
        
        // GraphSAGE层
        for (let i = 0; i < this.config.numLayers; i++) {
            const outputDim = i === this.config.numLayers - 1 ? 
                this.config.outputDim : this.config.hiddenDim;
            
            x = this.sageLayer(x, neighborsInput, outputDim, `sage_${i}`);
            
            if (i < this.config.numLayers - 1) {
                x = tf.layers.activation({activation: 'relu'}).apply(x);
                x = tf.layers.dropout({rate: 0.5}).apply(x);
            }
        }
        
        const output = tf.layers.activation({activation: 'softmax'}).apply(x);
        
        this.model = tf.model({
            inputs: [featuresInput, neighborsInput],
            outputs: output
        });
        
        this.optimizer = tf.train.adam(this.config.learningRate);
        
        return this.model;
    }

    /**
     * GraphSAGE层实现
     */
    sageLayer(nodeFeatures, neighborFeatures, outputDim, name) {
        return tf.tidy(() => {
            const inputDim = nodeFeatures.shape[nodeFeatures.shape.length - 1];
            
            // 聚合邻居特征
            let aggregated;
            switch (this.config.aggregator) {
                case 'mean':
                    aggregated = tf.mean(neighborFeatures, 1);
                    break;
                case 'max':
                    aggregated = tf.max(neighborFeatures, 1);
                    break;
                case 'lstm':
                    // 简化的LSTM聚合
                    aggregated = this.lstmAggregator(neighborFeatures, name);
                    break;
                default:
                    aggregated = tf.mean(neighborFeatures, 1);
            }
            
            // 拼接节点特征和聚合的邻居特征
            const combined = tf.concat([nodeFeatures, aggregated], -1);
            
            // 线性变换
            const weights = tf.variable(
                tf.randomNormal([combined.shape[combined.shape.length - 1], outputDim], 0, 0.1),
                true,
                `${name}_weights`
            );
            
            const bias = tf.variable(
                tf.zeros([outputDim]),
                true,
                `${name}_bias`
            );
            
            const output = tf.add(tf.matMul(combined, weights), bias);
            
            // L2归一化
            return tf.l2Normalize(output, -1);
        });
    }

    /**
     * LSTM聚合器
     */
    lstmAggregator(neighborFeatures, name) {
        return tf.tidy(() => {
            // 简化的LSTM实现
            const inputDim = neighborFeatures.shape[neighborFeatures.shape.length - 1];
            const hiddenDim = Math.min(inputDim, 32);
            
            const lstmCell = tf.layers.lstmCell({
                units: hiddenDim,
                name: `${name}_lstm`
            });
            
            // 处理序列
            const sequenceLength = neighborFeatures.shape[1];
            let h = tf.zeros([neighborFeatures.shape[0], hiddenDim]);
            let c = tf.zeros([neighborFeatures.shape[0], hiddenDim]);
            
            for (let t = 0; t < sequenceLength; t++) {
                const input = neighborFeatures.slice([0, t, 0], [-1, 1, -1]).squeeze([1]);
                const [newH, newC] = lstmCell.apply([input, [h, c]]);
                h = newH;
                c = newC;
            }
            
            return h;
        });
    }

    /**
     * 采样邻居
     */
    sampleNeighbors(data, nodeIndices) {
        const batch = [];
        
        nodeIndices.forEach(nodeId => {
            const neighbors = graphDataProcessor.getNeighbors(
                nodeId, data.edges, this.config.sampleSize
            );
            
            // 填充或截断到固定大小
            const paddedNeighbors = new Array(this.config.sampleSize).fill(0);
            for (let i = 0; i < Math.min(neighbors.length, this.config.sampleSize); i++) {
                paddedNeighbors[i] = neighbors[i];
            }
            
            batch.push({
                nodeId,
                neighbors: paddedNeighbors
            });
        });
        
        return batch;
    }

    /**
     * 训练步骤
     */
    async trainStep(data, trainIndices) {
        return tf.tidy(() => {
            const f = () => {
                const batch = this.sampleNeighbors(data, trainIndices);
                
                // 准备批次数据
                const nodeFeatures = tf.stack(
                    batch.map(item => data.features.slice([item.nodeId, 0], [1, -1]).squeeze())
                );
                
                const neighborFeatures = tf.stack(
                    batch.map(item => tf.stack(
                        item.neighbors.map(nId => data.features.slice([nId, 0], [1, -1]).squeeze())
                    ))
                );
                
                const predictions = this.model.apply([nodeFeatures, neighborFeatures]);
                
                // 检查标签数据是否存在
                if (!data.labels) {
                    throw new Error('训练数据中缺少标签信息');
                }
                
                const labels = tf.oneHot(
                    tf.tensor1d(batch.map(item => data.labels.dataSync()[item.nodeId]), 'int32'),
                    data.numClasses
                );
                
                const loss = tf.losses.softmaxCrossEntropy(labels, predictions);
                return loss;
            };
            
            const {value: loss, grads} = tf.variableGrads(f);
            this.optimizer.applyGradients(grads);
            
            return loss;
        });
    }

    /**
     * 评估模型
     */
    evaluate(data, indices) {
        return tf.tidy(() => {
            const batch = this.sampleNeighbors(data, indices);
            
            const nodeFeatures = tf.stack(
                batch.map(item => data.features.slice([item.nodeId, 0], [1, -1]).squeeze())
            );
            
            const neighborFeatures = tf.stack(
                batch.map(item => tf.stack(
                    item.neighbors.map(nId => data.features.slice([nId, 0], [1, -1]).squeeze())
                ))
            );
            
            const predictions = this.model.predict([nodeFeatures, neighborFeatures]);
            
            // 检查标签数据是否存在
            if (!data.labels) {
                throw new Error('训练数据中缺少标签信息');
            }
            
            const labels = tf.oneHot(
                tf.tensor1d(batch.map(item => data.labels.dataSync()[item.nodeId]), 'int32'),
                data.numClasses
            );
            
            const loss = tf.losses.softmaxCrossEntropy(labels, predictions);
            
            const predictedClasses = tf.argMax(predictions, 1);
            const trueClasses = tf.argMax(labels, 1);
            const accuracy = tf.mean(tf.cast(tf.equal(predictedClasses, trueClasses), 'float32'));
            
            return {loss: loss.dataSync()[0], accuracy: accuracy.dataSync()[0]};
        });
    }

    /**
     * 训练模型
     */
    async train(data, epochs = 200, callbacks = {}) {
        this.isTraining = true;
        const startTime = Date.now();
        
        for (let epoch = 0; epoch < epochs && this.isTraining; epoch++) {
            // 随机采样训练批次
            const batchSize = Math.min(32, data.split.train.indices.length);
            const shuffledIndices = data.split.train.indices.sort(() => 0.5 - Math.random());
            const batchIndices = shuffledIndices.slice(0, batchSize);
            
            const loss = await this.trainStep(data, batchIndices);
            
            // 评估
            const trainMetrics = this.evaluate(data, data.split.train.indices.slice(0, 50));
            const valMetrics = this.evaluate(data, data.split.val.indices.slice(0, 30));
            
            this.trainingHistory.loss.push(trainMetrics.loss);
            this.trainingHistory.accuracy.push(trainMetrics.accuracy);
            this.trainingHistory.valLoss.push(valMetrics.loss);
            this.trainingHistory.valAccuracy.push(valMetrics.accuracy);
            
            if (callbacks.onEpochEnd) {
                await callbacks.onEpochEnd(epoch, {
                    loss: trainMetrics.loss,
                    accuracy: trainMetrics.accuracy,
                    valLoss: valMetrics.loss,
                    valAccuracy: valMetrics.accuracy,
                    time: Date.now() - startTime
                });
            }
            
            await tf.nextFrame();
        }
        
        this.isTraining = false;
        return this.trainingHistory;
    }

    /**
     * 生成节点嵌入
     */
    generateEmbeddings(data) {
        const embeddings = [];
        
        for (let i = 0; i < data.nodes.length; i++) {
            const batch = this.sampleNeighbors(data, [i]);
            
            const nodeFeatures = data.features.slice([i, 0], [1, -1]);
            const neighborFeatures = tf.stack(
                batch[0].neighbors.map(nId => data.features.slice([nId, 0], [1, -1]).squeeze())
            ).expandDims(0);
            
            const embedding = this.model.predict([nodeFeatures, neighborFeatures]);
            embeddings.push(embedding.dataSync());
        }
        
        this.nodeEmbeddings = embeddings;
        return embeddings;
    }

    /**
     * 停止训练
     */
    stopTraining() {
        this.isTraining = false;
    }

    /**
     * 重置模型
     */
    reset() {
        if (this.model) {
            this.model.dispose();
        }
        this.trainingHistory = {
            loss: [],
            accuracy: [],
            valLoss: [],
            valAccuracy: []
        };
        this.nodeEmbeddings = null;
    }
}

// 导出模型类
window.GCNModel = GCNModel;
window.GATModel = GATModel;
window.GraphSAGEModel = GraphSAGEModel;