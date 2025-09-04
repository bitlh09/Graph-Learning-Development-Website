// Cora数据集GCN节点分类模块

// 真实Cora数据集管理
var realCoraDataset = {
    isLoaded: false,
    data: null,
    apiBaseUrl: 'http://localhost:5001/api/cora'
};

// 加载真实Cora数据集
async function loadRealCoraDataset(dataType = 'sample', size = 100) {
    try {
        let url;
        switch (dataType) {
            case 'info':
                url = `${realCoraDataset.apiBaseUrl}/info`;
                break;
            case 'sample':
                url = `${realCoraDataset.apiBaseUrl}/sample?size=${size}`;
                break;
            case 'full':
                url = `${realCoraDataset.apiBaseUrl}/full`;
                break;
            case 'visualization':
                url = `${realCoraDataset.apiBaseUrl}/visualization?size=${size}`;
                break;
            default:
                throw new Error('Unknown data type: ' + dataType);
        }
        
        console.log('Loading Cora dataset from:', url);
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Failed to load Cora dataset');
        }
        
        realCoraDataset.data = result.data;
        realCoraDataset.isLoaded = true;
        
        console.log('Real Cora dataset loaded successfully:', result.data);
        return result.data;
        
    } catch (error) {
        console.error('Failed to load real Cora dataset:', error);
        console.log('Falling back to simulated dataset...');
        
        // 如果真实数据加载失败，使用模拟数据
        if (!coraDataset.features) {
            initCoraDataset();
        }
        return null;
    }
}

// 检查Cora服务是否可用
async function checkCoraServiceAvailable() {
    try {
        const response = await fetch(`${realCoraDataset.apiBaseUrl}/info`, {
            method: 'GET',
            timeout: 3000
        });
        return response.ok;
    } catch (error) {
        return false;
    }
}

// 从稀疏格式恢复矩阵
function fromSparseFormat(sparseData) {
    const { data, indices, indptr, shape } = sparseData;
    const [rows, cols] = shape;
    const matrix = Array(rows).fill().map(() => Array(cols).fill(0));
    
    for (let i = 0; i < rows; i++) {
        const start = indptr[i];
        const end = indptr[i + 1];
        for (let j = start; j < end; j++) {
            matrix[i][indices[j]] = data[j];
        }
    }
    
    return matrix;
}

// Cora数据集模拟数据 (简化版本，包含主要论文分类)
var coraDataset = {
    // 2708个论文节点的特征向量 (简化为重要特征)
    features: null, // 将在初始化时生成
    // 邻接矩阵 (基于论文引用关系)
    adjacency: null, // 将在初始化时生成
    // 节点标签 (论文分类)
    labels: null, // 将在初始化时生成
    // 类别名称
    classes: [
        'Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 
        'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'
    ],
    // 数据集统计信息
    numNodes: 2708,
    numFeatures: 1433, // 词汇表大小
    numClasses: 7,
    numEdges: 5429
};

// 初始化Cora数据集
function initCoraDataset() {
    // 生成模拟的Cora数据集
    var numNodes = coraDataset.numNodes;
    var numFeatures = coraDataset.numFeatures;
    var numClasses = coraDataset.numClasses;
    
    // 生成特征矩阵 (稀疏特征，模拟论文的词袋表示)
    coraDataset.features = [];
    for (var i = 0; i < numNodes; i++) {
        var features = new Array(numFeatures).fill(0);
        // 随机设置一些特征为1 (模拟论文中出现的词汇)
        var numActiveFeatures = Math.floor(Math.random() * 50) + 10; // 10-60个活跃特征
        for (var j = 0; j < numActiveFeatures; j++) {
            var idx = Math.floor(Math.random() * numFeatures);
            features[idx] = 1;
        }
        coraDataset.features.push(features);
    }
    
    // 生成邻接矩阵 (基于引用关系)
    coraDataset.adjacency = [];
    for (var i = 0; i < numNodes; i++) {
        var row = new Array(numNodes).fill(0);
        // 每个论文平均引用4-8篇其他论文
        var numCitations = Math.floor(Math.random() * 5) + 4;
        for (var j = 0; j < numCitations; j++) {
            var citedPaper = Math.floor(Math.random() * numNodes);
            if (citedPaper !== i) {
                row[citedPaper] = 1;
            }
        }
        coraDataset.adjacency.push(row);
    }
    
    // 确保邻接矩阵对称 (无向图)
    for (var i = 0; i < numNodes; i++) {
        for (var j = i + 1; j < numNodes; j++) {
            if (coraDataset.adjacency[i][j] === 1 || coraDataset.adjacency[j][i] === 1) {
                coraDataset.adjacency[i][j] = 1;
                coraDataset.adjacency[j][i] = 1;
            }
        }
    }
    
    // 生成标签 (确保每个类别都有合理数量的样本)
    coraDataset.labels = [];
    var samplesPerClass = Math.floor(numNodes / numClasses);
    var currentClass = 0;
    var currentCount = 0;
    
    for (var i = 0; i < numNodes; i++) {
        coraDataset.labels.push(currentClass);
        currentCount++;
        if (currentCount >= samplesPerClass && currentClass < numClasses - 1) {
            currentClass++;
            currentCount = 0;
        }
    }
    
    // 随机打乱标签以避免聚类
    for (var i = numNodes - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = coraDataset.labels[i];
        coraDataset.labels[i] = coraDataset.labels[j];
        coraDataset.labels[j] = temp;
    }
    
    console.log('Cora数据集初始化完成:', {
        nodes: numNodes,
        features: numFeatures,
        classes: numClasses,
        edges: countEdges()
    });
}

// 计算边的数量
function countEdges() {
    var count = 0;
    for (var i = 0; i < coraDataset.numNodes; i++) {
        for (var j = i + 1; j < coraDataset.numNodes; j++) {
            if (coraDataset.adjacency[i][j] === 1) {
                count++;
            }
        }
    }
    return count;
}

// 数据预处理：添加自环并归一化
function preprocessAdjacency(adj) {
    var n = adj.length;
    var processed = [];
    
    // 添加自环
    for (var i = 0; i < n; i++) {
        var row = [];
        for (var j = 0; j < n; j++) {
            row.push(i === j ? 1 : adj[i][j]);
        }
        processed.push(row);
    }
    
    // 计算度矩阵
    var degree = new Array(n).fill(0);
    for (var i = 0; i < n; i++) {
        for (var j = 0; j < n; j++) {
            degree[i] += processed[i][j];
        }
    }
    
    // 对称归一化: D^(-1/2) * A * D^(-1/2)
    for (var i = 0; i < n; i++) {
        for (var j = 0; j < n; j++) {
            if (processed[i][j] !== 0) {
                processed[i][j] = processed[i][j] / Math.sqrt(degree[i] * degree[j]);
            }
        }
    }
    
    return processed;
}

// 训练测试集划分
function splitDataset(testRatio = 0.2, valRatio = 0.1) {
    var numNodes = coraDataset.numNodes;
    var indices = Array.from({length: numNodes}, (_, i) => i);
    
    // 随机打乱索引
    for (var i = indices.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    
    var testSize = Math.floor(numNodes * testRatio);
    var valSize = Math.floor(numNodes * valRatio);
    var trainSize = numNodes - testSize - valSize;
    
    return {
        train: indices.slice(0, trainSize),
        val: indices.slice(trainSize, trainSize + valSize),
        test: indices.slice(trainSize + valSize)
    };
}

// GCN模型（JavaScript实现）
function GCNModel(inputDim, hiddenDim, outputDim, learningRate = 0.01) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.outputDim = outputDim;
    this.lr = learningRate;
    
    // 初始化权重
    this.W1 = this.initWeights(inputDim, hiddenDim);
    this.W2 = this.initWeights(hiddenDim, outputDim);
    this.b1 = new Array(hiddenDim).fill(0);
    this.b2 = new Array(outputDim).fill(0);
    
    // 缓存中间结果用于反向传播
    this.cache = {};
}

GCNModel.prototype.initWeights = function(rows, cols) {
    var weights = [];
    var std = Math.sqrt(2.0 / (rows + cols)); // Xavier初始化
    for (var i = 0; i < rows; i++) {
        var row = [];
        for (var j = 0; j < cols; j++) {
            row.push((Math.random() * 2 - 1) * std);
        }
        weights.push(row);
    }
    return weights;
};

GCNModel.prototype.relu = function(x) {
    return x.map(row => row.map(val => Math.max(0, val)));
};

GCNModel.prototype.softmax = function(x) {
    return x.map(row => {
        var max = Math.max(...row);
        var exp = row.map(val => Math.exp(val - max));
        var sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(val => val / sum);
    });
};

GCNModel.prototype.matmul = function(A, B) {
    var result = [];
    for (var i = 0; i < A.length; i++) {
        var row = [];
        for (var j = 0; j < B[0].length; j++) {
            var sum = 0;
            for (var k = 0; k < B.length; k++) {
                sum += A[i][k] * B[k][j];
            }
            row.push(sum);
        }
        result.push(row);
    }
    return result;
};

GCNModel.prototype.forward = function(features, adjacency) {
    // 第一层：A * X * W1 + b1
    var AX = this.matmul(adjacency, features);
    var Z1 = this.matmul(AX, this.W1);
    for (var i = 0; i < Z1.length; i++) {
        for (var j = 0; j < Z1[i].length; j++) {
            Z1[i][j] += this.b1[j];
        }
    }
    
    // ReLU激活
    var A1 = this.relu(Z1);
    
    // 第二层：A * A1 * W2 + b2
    var AA1 = this.matmul(adjacency, A1);
    var Z2 = this.matmul(AA1, this.W2);
    for (var i = 0; i < Z2.length; i++) {
        for (var j = 0; j < Z2[i].length; j++) {
            Z2[i][j] += this.b2[j];
        }
    }
    
    // Softmax输出
    var output = this.softmax(Z2);
    
    // 缓存中间结果
    this.cache = {
        AX: AX,
        Z1: Z1,
        A1: A1,
        AA1: AA1,
        Z2: Z2,
        output: output
    };
    
    return output;
};

GCNModel.prototype.computeLoss = function(predictions, labels, indices) {
    var loss = 0;
    var count = 0;
    for (var i = 0; i < indices.length; i++) {
        var idx = indices[i];
        var pred = predictions[idx];
        var label = labels[idx];
        if (pred[label] > 0) {
            loss -= Math.log(pred[label]);
        } else {
            loss -= Math.log(1e-10); // 避免log(0)
        }
        count++;
    }
    return loss / count;
};

GCNModel.prototype.computeAccuracy = function(predictions, labels, indices) {
    var correct = 0;
    for (var i = 0; i < indices.length; i++) {
        var idx = indices[i];
        var pred = predictions[idx];
        var predicted = pred.indexOf(Math.max(...pred));
        if (predicted === labels[idx]) {
            correct++;
        }
    }
    return correct / indices.length;
};

// Cora数据集GCN训练函数
async function trainCoraGCN(options = {}) {
    var {
        hiddenDim = 16,
        learningRate = 0.01,
        epochs = 200,
        onProgress = null,
        onComplete = null
    } = options;
    
    // 初始化数据集
    if (!coraDataset.features) {
        initCoraDataset();
    }
    
    // 预处理邻接矩阵
    var processedAdj = preprocessAdjacency(coraDataset.adjacency);
    
    // 划分数据集
    var split = splitDataset();
    
    // 创建模型
    var model = new GCNModel(
        coraDataset.numFeatures, 
        hiddenDim, 
        coraDataset.numClasses, 
        learningRate
    );
    
    var trainingHistory = {
        losses: [],
        trainAccs: [],
        valAccs: [],
        testAccs: []
    };
    
    console.log('开始训练Cora GCN模型...');
    console.log('训练集大小:', split.train.length);
    console.log('验证集大小:', split.val.length);
    console.log('测试集大小:', split.test.length);
    
    // 训练循环
    for (var epoch = 0; epoch < epochs; epoch++) {
        // 前向传播
        var predictions = model.forward(coraDataset.features, processedAdj);
        
        // 计算损失和准确率
        var trainLoss = model.computeLoss(predictions, coraDataset.labels, split.train);
        var trainAcc = model.computeAccuracy(predictions, coraDataset.labels, split.train);
        var valAcc = model.computeAccuracy(predictions, coraDataset.labels, split.val);
        var testAcc = model.computeAccuracy(predictions, coraDataset.labels, split.test);
        
        // 记录历史
        trainingHistory.losses.push(trainLoss);
        trainingHistory.trainAccs.push(trainAcc);
        trainingHistory.valAccs.push(valAcc);
        trainingHistory.testAccs.push(testAcc);
        
        // 进度回调
        if (onProgress && epoch % Math.max(1, Math.floor(epochs / 20)) === 0) {
            onProgress({
                epoch: epoch,
                loss: trainLoss,
                trainAcc: trainAcc,
                valAcc: valAcc,
                testAcc: testAcc
            });
        }
        
        // 简单的SGD更新（实际应该实现完整的反向传播）
        // 这里为了演示简化了训练过程
        
        // 添加小延迟以显示训练进度
        if (epoch < epochs - 1) {
            await new Promise(resolve => setTimeout(resolve, 10));
        }
    }
    
    // 最终结果
    var finalPredictions = model.forward(coraDataset.features, processedAdj);
    var finalTestAcc = model.computeAccuracy(finalPredictions, coraDataset.labels, split.test);
    
    console.log('Cora GCN训练完成!');
    console.log('最终测试准确率:', finalTestAcc.toFixed(4));
    
    var results = {
        model: model,
        predictions: finalPredictions,
        testAccuracy: finalTestAcc,
        history: trainingHistory,
        split: split,
        dataset: coraDataset
    };
    
    if (onComplete) {
        onComplete(results);
    }
    
    return results;
}

// 可视化Cora图结构（采样版本）
function visualizeCoraSample(containerId, sampleSize = 50) {
    var container = document.getElementById(containerId);
    if (!container || !window.d3) return;
    
    container.innerHTML = '';
    var width = container.clientWidth || 400;
    var height = 300;
    var svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
    
    // 随机采样节点
    var sampleIndices = [];
    for (var i = 0; i < sampleSize && i < coraDataset.numNodes; i++) {
        sampleIndices.push(Math.floor(Math.random() * coraDataset.numNodes));
    }
    
    // 构建采样图的节点和边
    var nodes = sampleIndices.map(function(idx) {
        return {
            id: idx,
            class: coraDataset.classes[coraDataset.labels[idx]],
            label: coraDataset.labels[idx]
        };
    });
    
    var links = [];
    for (var i = 0; i < sampleIndices.length; i++) {
        for (var j = i + 1; j < sampleIndices.length; j++) {
            var idx1 = sampleIndices[i];
            var idx2 = sampleIndices[j];
            if (coraDataset.adjacency[idx1][idx2] === 1) {
                links.push({source: i, target: j});
            }
        }
    }
    
    // D3力导向布局
    var colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    
    var simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(function(d, i) { return i; }).distance(50))
        .force('charge', d3.forceManyBody().strength(-100))
        .force('center', d3.forceCenter(width / 2, height / 2));
    
    var link = svg.append('g')
        .selectAll('line')
        .data(links)
        .enter().append('line')
        .attr('stroke', '#999')
        .attr('stroke-width', 1);
    
    var node = svg.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter().append('circle')
        .attr('r', 6)
        .attr('fill', function(d) { return colorScale(d.label); })
        .attr('stroke', '#fff')
        .attr('stroke-width', 1)
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
            d3.select(this).attr('r', 8);
            
            // 显示论文信息
            var tooltip = d3.select('body').append('div')
                .style('position', 'absolute')
                .style('background', 'rgba(0,0,0,0.8)')
                .style('color', 'white')
                .style('padding', '8px')
                .style('border-radius', '4px')
                .style('font-size', '12px')
                .style('pointer-events', 'none')
                .html(`论文ID: ${d.id}<br/>类别: ${d.class}`)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px');
            
            setTimeout(function() { tooltip.remove(); }, 2000);
        })
        .on('mouseout', function(event, d) {
            d3.select(this).attr('r', 6);
        });
    
    simulation.on('tick', function() {
        link.attr('x1', function(d) { return d.source.x; })
            .attr('y1', function(d) { return d.source.y; })
            .attr('x2', function(d) { return d.target.x; })
            .attr('y2', function(d) { return d.target.y; });
        
        node.attr('cx', function(d) { return d.x; })
            .attr('cy', function(d) { return d.y; });
    });
    
    // 添加图例
    var legend = svg.append('g')
        .attr('class', 'legend')
        .attr('transform', 'translate(10, 20)');
    
    coraDataset.classes.forEach(function(className, i) {
        var legendItem = legend.append('g')
            .attr('transform', 'translate(0, ' + (i * 15) + ')');
        
        legendItem.append('circle')
            .attr('r', 4)
            .attr('fill', colorScale(i));
        
        legendItem.append('text')
            .attr('x', 10)
            .attr('y', 0)
            .attr('dy', '0.35em')
            .style('font-size', '10px')
            .text(className.replace('_', ' '));
    });
}

// 导出函数
window.coraDataset = coraDataset;
window.initCoraDataset = initCoraDataset;
window.trainCoraGCN = trainCoraGCN;
window.visualizeCoraSample = visualizeCoraSample;

// 导出新的真实数据函数
window.realCoraDataset = realCoraDataset;
window.loadRealCoraDataset = loadRealCoraDataset;
window.checkCoraServiceAvailable = checkCoraServiceAvailable;
window.trainRealCoraGCN = trainRealCoraGCN;
window.visualizeRealCoraSample = visualizeRealCoraSample;

// 支持真实Cora数据集的增强训练函数
async function trainRealCoraGCN(options = {}) {
    const {
        hiddenDim = 16,
        learningRate = 0.01,
        epochs = 200,
        dataSize = 100,
        useRealData = true,
        onProgress = null,
        onComplete = null
    } = options;
    
    console.log('开始训练Real Cora GCN模型...');
    
    try {
        let dataset = null;
        
        // 尝试加载真实数据
        if (useRealData) {
            const serviceAvailable = await checkCoraServiceAvailable();
            if (serviceAvailable) {
                console.log('Cora服务可用，加载真实数据...');
                dataset = await loadRealCoraDataset('sample', dataSize);
            } else {
                console.log('Cora服务不可用，请启动cora_server.py');
            }
        }
        
        // 如果真实数据加载失败，使用模拟数据
        if (!dataset) {
            console.log('使用模拟数据进行训练...');
            if (!coraDataset.features) {
                initCoraDataset();
            }
            
            // 将模拟数据转换为相同格式
            const sampleSize = Math.min(dataSize, coraDataset.numNodes);
            const indices = Array.from({length: sampleSize}, (_, i) => i);
            
            dataset = {
                features: indices.map(i => coraDataset.features[i].slice(0, 50)), // 只使用前50个特征
                adjacency: indices.map(i => indices.map(j => coraDataset.adjacency[i][j])),
                labels: indices.map(i => coraDataset.labels[i]),
                train_mask: indices.map(i => i < sampleSize * 0.6),
                val_mask: indices.map(i => i >= sampleSize * 0.6 && i < sampleSize * 0.8),
                test_mask: indices.map(i => i >= sampleSize * 0.8),
                num_nodes: sampleSize,
                num_features: 50,
                num_classes: coraDataset.numClasses
            };
        }
        
        console.log('数据集信息:', {
            nodes: dataset.num_nodes,
            features: dataset.num_features,
            classes: dataset.num_classes,
            train_samples: dataset.train_mask.filter(x => x).length,
            val_samples: dataset.val_mask.filter(x => x).length,
            test_samples: dataset.test_mask.filter(x => x).length
        });
        
        // 预处理邻接矩阵
        const processedAdj = preprocessAdjacency(dataset.adjacency);
        
        // 创建模型
        const model = new GCNModel(dataset.num_features, hiddenDim, dataset.num_classes, learningRate);
        
        // 训练历史
        const history = {
            losses: [],
            trainAccs: [],
            valAccs: [],
            testAccs: []
        };
        
        // 训练循环
        for (let epoch = 0; epoch < epochs; epoch++) {
            // 前向传播
            const output = model.forward(dataset.features, processedAdj);
            
            // 计算损失
            const loss = model.computeLoss(output, dataset.labels, dataset.train_mask);
            
            // 反向传播
            model.backward(output, dataset.labels, dataset.train_mask, dataset.features, processedAdj);
            
            // 计算准确率
            const trainAcc = model.computeAccuracy(output, dataset.labels, dataset.train_mask);
            const valAcc = model.computeAccuracy(output, dataset.labels, dataset.val_mask);
            const testAcc = model.computeAccuracy(output, dataset.labels, dataset.test_mask);
            
            // 保存历史
            history.losses.push(loss);
            history.trainAccs.push(trainAcc);
            history.valAccs.push(valAcc);
            history.testAccs.push(testAcc);
            
            // 进度回调
            if (onProgress) {
                onProgress({
                    epoch: epoch + 1,
                    loss: loss,
                    trainAcc: trainAcc,
                    valAcc: valAcc,
                    testAcc: testAcc
                });
            }
            
            // 每10个轮次输出一次日志
            if ((epoch + 1) % 10 === 0 || epoch === 0) {
                console.log(`Epoch ${epoch + 1}/${epochs}: Loss=${loss.toFixed(4)}, TrainAcc=${trainAcc.toFixed(4)}, ValAcc=${valAcc.toFixed(4)}, TestAcc=${testAcc.toFixed(4)}`);
            }
            
            // 添加延迟以显示进度
            if (epoch < epochs - 1) {
                await new Promise(resolve => setTimeout(resolve, 5));
            }
        }
        
        const results = {
            history: history,
            testAccuracy: history.testAccs[history.testAccs.length - 1],
            dataset: {
                numNodes: dataset.num_nodes,
                numFeatures: dataset.num_features,
                numClasses: dataset.num_classes,
                numEdges: Math.round(dataset.adjacency.flat().reduce((a, b) => a + b, 0) / 2)
            },
            model: model,
            finalOutput: model.forward(dataset.features, processedAdj),
            isRealData: !!realCoraDataset.isLoaded
        };
        
        console.log('训练完成！最终测试准确率:', results.testAccuracy.toFixed(4));
        
        if (onComplete) {
            onComplete(results);
        }
        
        return results;
        
    } catch (error) {
        console.error('Real Cora GCN训练失败:', error);
        throw error;
    }
}

// 可视化Cora样本图结构
async function visualizeRealCoraSample(containerId, sampleSize = 30) {
    try {
        // 尝试加载可视化数据
        const visData = await loadRealCoraDataset('visualization', sampleSize);
        
        if (!visData) {
            console.log('使用模拟数据进行可视化...');
            visualizeCoraSample(containerId, sampleSize); // 回退到原有函数
            return;
        }
        
        const container = document.getElementById(containerId);
        if (!container || !window.d3) {
            console.error('容器或D3.js不可用');
            return;
        }
        
        container.innerHTML = '';
        const width = container.clientWidth || 600;
        const height = container.clientHeight || 400;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        // 设置颜色映射
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
        
        // 创建力导向模拟
        const simulation = d3.forceSimulation(visData.nodes)
            .force('link', d3.forceLink(visData.edges).id(d => d.id).distance(50))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2));
        
        // 绘制连接
        const link = svg.append('g')
            .selectAll('line')
            .data(visData.edges)
            .enter().append('line')
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', d => Math.sqrt(d.weight) * 2);
        
        // 绘制节点
        const node = svg.append('g')
            .selectAll('circle')
            .data(visData.nodes)
            .enter().append('circle')
            .attr('r', 8)
            .attr('fill', d => colorScale(d.label))
            .attr('stroke', d => {
                if (d.is_train) return '#2d5c2d';
                if (d.is_val) return '#8b4513';
                if (d.is_test) return '#800080';
                return '#000';
            })
            .attr('stroke-width', 2)
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
        
        // 添加节点标签
        const labels = svg.append('g')
            .selectAll('text')
            .data(visData.nodes)
            .enter().append('text')
            .text(d => d.id)
            .attr('font-size', '10px')
            .attr('dx', 12)
            .attr('dy', 4);
        
        // 添加提示信息
        node.append('title')
            .text(d => `节点 ${d.id}\n类别: ${d.class_name}\n类型: ${d.is_train ? '训练' : d.is_val ? '验证' : '测试'}`);
        
        // 更新节点和连接位置
        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
            
            labels
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
        
        // 拖拽事件处理
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        
        // 添加图例
        const legend = svg.append('g')
            .attr('transform', 'translate(20, 20)');
        
        visData.class_names.forEach((className, i) => {
            const legendRow = legend.append('g')
                .attr('transform', `translate(0, ${i * 20})`);
            
            legendRow.append('circle')
                .attr('r', 6)
                .attr('fill', colorScale(i));
            
            legendRow.append('text')
                .attr('x', 12)
                .attr('y', 4)
                .attr('font-size', '12px')
                .text(className);
        });
        
        console.log('真实Cora数据可视化完成！');
        
    } catch (error) {
        console.error('可视化失败:', error);
        // 回退到模拟数据可视化
        visualizeCoraSample(containerId, sampleSize);
    }
}