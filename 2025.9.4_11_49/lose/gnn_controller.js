/**
 * 图神经网络在线实践控制器
 * 整合数据处理、模型训练、可视化等功能
 */

class GNNController {
    constructor() {
        this.currentModel = null;
        this.currentData = null;
        this.currentAlgorithm = 'gcn';
        this.isTraining = false;
        this.trainingHistory = null;
        
        // 可视化器
        this.graphVisualizer = null;
        this.trainingVisualizer = null;
        this.attentionVisualizer = null;
        this.embeddingVisualizer = null;
        
        // 状态和错误信息元素
        this.statusElement = document.getElementById('status-message');
        this.errorElement = document.getElementById('error-message');
        
        // 配置参数
        this.config = {
            gcn: {
                hiddenDim: 16,
                learningRate: 0.01,
                dropout: 0.5,
                epochs: 200
            },
            gat: {
                hiddenDim: 8,
                numHeads: 8,
                learningRate: 0.005,
                dropout: 0.6,
                epochs: 200
            },
            graphsage: {
                hiddenDim: 16,
                numLayers: 2,
                sampleSize: 10,
                aggregator: 'mean',
                learningRate: 0.01,
                epochs: 200
            }
        };
        
        this.initializeInterface();
        this.bindEvents();
        this.initializeTensorFlowBackend();
    }

    /**
     * 初始化TensorFlow.js后端配置
     */
    async initializeTensorFlowBackend() {
        try {
            // 检查WebGL支持
            if (tf.getBackend() === 'webgl') {
                console.log('使用WebGL后端');
            } else {
                console.log('WebGL不可用，使用CPU后端');
                await tf.setBackend('cpu');
            }
        } catch (error) {
            console.warn('后端初始化失败，切换到CPU后端:', error);
            try {
                await tf.setBackend('cpu');
                console.log('已切换到CPU后端');
            } catch (cpuError) {
                console.error('CPU后端也无法使用:', cpuError);
                this.showError('TensorFlow.js后端初始化失败，请刷新页面重试');
            }
        }
    }

    /**
     * 处理WebGL错误并切换到CPU后端
     */
    async handleWebGLError(error) {
        console.warn('检测到WebGL错误，尝试切换到CPU后端:', error);
        try {
            await tf.setBackend('cpu');
            console.log('已成功切换到CPU后端');
            this.updateStatus('检测到GPU问题，已切换到CPU模式继续训练');
            return true;
        } catch (cpuError) {
            console.error('切换到CPU后端失败:', cpuError);
            this.showError('无法切换到CPU后端，请刷新页面重试');
            return false;
        }
    }

    /**
     * 初始化界面
     */
    initializeInterface() {
        // 初始化可视化器
        this.graphVisualizer = new GNNVisualizer('graph-container');
        this.trainingVisualizer = new TrainingVisualizer('training-plots');
        this.attentionVisualizer = new AttentionVisualizer('attention-plot');
        this.embeddingVisualizer = new EmbeddingVisualizer('embedding-plot');
        
        // 设置默认数据集
        this.loadDataset('karate');
        
        // 更新界面状态
        this.updateInterface();
    }

    /**
     * 绑定事件
     */
    bindEvents() {
        // 绑定可视化切换事件
        this.bindVisualizationTabs();
        
        // 算法选择
        document.querySelectorAll('.algorithm-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.selectAlgorithm(e.target.dataset.algorithm);
            });
        });
        
        // 数据集选择
        document.getElementById('dataset-select').addEventListener('change', (e) => {
            this.loadDataset(e.target.value);
        });
        
        // 参数控制
        this.bindParameterControls();
        
        // 训练控制
        document.getElementById('start-training').addEventListener('click', () => {
            this.startTraining();
        });
        
        document.getElementById('stop-training').addEventListener('click', () => {
            this.stopTraining();
        });
        
        document.getElementById('reset-model').addEventListener('click', () => {
            this.resetModel();
        });
        
        // 可视化控制
        document.getElementById('show-predictions').addEventListener('change', (e) => {
            this.togglePredictions(e.target.checked);
        });
        
        document.getElementById('show-attention').addEventListener('change', (e) => {
            this.toggleAttentionVisualization(e.target.checked);
        });
        
        // 导出功能
        document.getElementById('export-model').addEventListener('click', () => {
            this.exportModel();
        });
        
        document.getElementById('export-data').addEventListener('click', () => {
            this.exportTrainingData();
        });
    }

    /**
     * 绑定参数控制事件
     */
    bindParameterControls() {
        // 通用参数
        ['hidden-dim', 'learning-rate', 'epochs'].forEach(param => {
            const element = document.getElementById(param);
            if (element) {
                element.addEventListener('input', () => {
                    this.updateParameter(param, parseFloat(element.value));
                });
            }
        });
        
        // GCN特定参数
        const gcnDropout = document.getElementById('gcn-dropout');
        if (gcnDropout) {
            gcnDropout.addEventListener('input', () => {
                this.config.gcn.dropout = parseFloat(gcnDropout.value);
            });
        }
        
        // GAT特定参数
        const gatHeads = document.getElementById('gat-heads');
        const gatDropout = document.getElementById('gat-dropout');
        
        if (gatHeads) {
            gatHeads.addEventListener('input', () => {
                this.config.gat.numHeads = parseInt(gatHeads.value);
            });
        }
        
        if (gatDropout) {
            gatDropout.addEventListener('input', () => {
                this.config.gat.dropout = parseFloat(gatDropout.value);
            });
        }
        
        // GraphSAGE特定参数
        const sageAggregator = document.getElementById('sage-aggregator');
        const sageSample = document.getElementById('sage-sample-size');
        const sageLayers = document.getElementById('sage-num-layers');
        
        if (sageAggregator) {
            sageAggregator.addEventListener('change', () => {
                this.config.graphsage.aggregator = sageAggregator.value;
            });
        }
        
        if (sageSample) {
            sageSample.addEventListener('input', () => {
                this.config.graphsage.sampleSize = parseInt(sageSample.value);
            });
        }
        
        if (sageLayers) {
            sageLayers.addEventListener('input', () => {
                this.config.graphsage.numLayers = parseInt(sageLayers.value);
            });
        }
    }

    /**
     * 选择算法
     */
    selectAlgorithm(algorithm) {
        this.currentAlgorithm = algorithm;
        
        // 更新标签页状态
        document.querySelectorAll('.algorithm-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-algorithm="${algorithm}"]`).classList.add('active');
        
        // 显示对应的参数面板
        document.querySelectorAll('.algorithm-params').forEach(panel => {
            panel.style.display = 'none';
        });
        document.getElementById(`${algorithm}-params`).style.display = 'block';
        
        // 重置模型
        this.resetModel();
        
        // 更新界面
        this.updateInterface();
    }

    /**
     * 加载数据集
     */
    async loadDataset(datasetName) {
        this.clearError();
        this.updateStatus(`正在加载 ${datasetName} 数据集...`);
        this.isTraining = true;

        try {
            let rawData = await graphDataProcessor.loadDataset(datasetName);
            
            if (!rawData.adj) {
                rawData.adj = graphDataProcessor.buildAdjacencyMatrix(rawData.nodes, rawData.edges);
            }

            // Preserve the split from loadDataset before preprocessing
            const dataSplit = rawData.split;
            this.currentData = await graphDataProcessor.preprocessData(rawData);
            // Restore the split after preprocessing
            this.currentData.split = dataSplit;

            this.updateStatus(`${datasetName} 数据集加载和预处理完成。`);
            this.updateDatasetInfo();
            this.graphVisualizer.visualizeGraph(this.currentData);

        } catch (error) {
            this.showError(`加载数据集失败: ${error.message}`);
            console.error(error);
        } finally {
            this.isTraining = false;
        }
    }

    /**
     * 更新数据集信息
     */
    updateDatasetInfo() {
        if (!this.currentData) return;
        document.getElementById('dataset-nodes').textContent = this.currentData.nodes.length;
        document.getElementById('dataset-edges').textContent = this.currentData.edges.length;
        document.getElementById('dataset-classes').textContent = this.currentData.numClasses;
        document.getElementById('dataset-features').textContent = this.currentData.features.shape[1];
    }

    /**
     * 更新参数
     */
    updateParameter(param, value) {
        const config = this.config[this.currentAlgorithm];
        
        switch (param) {
            case 'hidden-dim':
                config.hiddenDim = value;
                break;
            case 'learning-rate':
                config.learningRate = value;
                break;
            case 'epochs':
                config.epochs = value;
                break;
        }
    }

    /**
     * 开始训练
     */
    async startTraining() {
        if (!this.currentData) {
            this.showStatus('请先加载数据集', 'error');
            return;
        }
        
        if (this.isTraining) {
            this.showStatus('模型正在训练中', 'warning');
            return;
        }
        
        try {
            this.isTraining = true;
            this.updateTrainingControls();
            this.trainingVisualizer.reset();
            
            // 创建模型
            this.createModel();
            
            this.showStatus('开始训练模型...', 'info');
            
            // 训练回调
            const callbacks = {
                onEpochEnd: async (epoch, metrics) => {
                    // 更新训练可视化
                    this.trainingVisualizer.updateTrainingData(epoch, metrics);
                    
                    // 更新状态显示
                    this.updateTrainingStatus(epoch, metrics);
                    
                    // 定期更新图可视化和节点嵌入
                    if (epoch % 10 === 0) {
                        await this.updateVisualization();
                        
                        // 如果当前显示的是嵌入面板，更新节点嵌入可视化
                        const embeddingPanel = document.getElementById('embedding-panel');
                        if (embeddingPanel && embeddingPanel.classList.contains('active')) {
                            await this.updateEmbeddingVisualization();
                        }
                    }
                }
            };
            
            // 开始训练
            this.trainingHistory = await this.currentModel.train(
                this.currentData,
                this.config[this.currentAlgorithm].epochs,
                callbacks
            );
            
            this.showStatus('训练完成!', 'success');
            
            // 最终可视化更新
            await this.updateVisualization();
            
        } catch (error) {
            console.error('训练失败:', error);
            
            // 检查是否为WebGL相关错误
            if (this.isWebGLError(error)) {
                const switched = await this.handleWebGLError(error);
                if (switched) {
                    // 重新创建模型并重试训练
                    try {
                        this.showStatus('正在CPU模式下重新训练...', 'info');
                        this.createModel();
                        this.trainingHistory = await this.currentModel.train(
                            this.currentData,
                            this.config[this.currentAlgorithm].epochs,
                            callbacks
                        );
                        this.showStatus('训练完成! (CPU模式)', 'success');
                        await this.updateVisualization();
                    } catch (retryError) {
                        console.error('CPU模式重试失败:', retryError);
                        this.showStatus('训练失败: ' + retryError.message, 'error');
                    }
                } else {
                    this.showStatus('训练失败: ' + error.message, 'error');
                }
            } else {
                this.showStatus('训练失败: ' + error.message, 'error');
            }
        } finally {
            this.isTraining = false;
            this.updateTrainingControls();
        }
    }

    /**
     * 创建模型
     */
    createModel() {
        const config = {
            inputDim: this.currentData.features.shape[1],
            outputDim: this.currentData.numClasses,
            ...this.config[this.currentAlgorithm]
        };
        
        switch (this.currentAlgorithm) {
            case 'gcn':
                this.currentModel = new GCNModel(config);
                break;
            case 'gat':
                this.currentModel = new GATModel(config);
                break;
            case 'graphsage':
                this.currentModel = new GraphSAGEModel(config);
                break;
            default:
                throw new Error('未知的算法类型');
        }
        
        this.currentModel.buildModel();
    }

    /**
     * 检查是否为WebGL相关错误
     */
    isWebGLError(error) {
        const errorMessage = error.message || error.toString();
        return errorMessage.includes('Failed to link vertex and fragment shaders') ||
               errorMessage.includes('webgl') ||
               errorMessage.includes('gpgpu') ||
               errorMessage.includes('OneHot.js') ||
               errorMessage.includes('backend_webgl');
    }

    /**
     * 停止训练
     */
    stopTraining() {
        if (this.currentModel && this.isTraining) {
            this.currentModel.stopTraining();
            this.showStatus('训练已停止', 'warning');
        }
    }

    /**
     * 重置模型
     */
    resetModel() {
        if (this.currentModel) {
            this.currentModel.reset();
            this.currentModel = null;
        }
        
        this.trainingHistory = null;
        this.trainingVisualizer.reset();
        
        // 重置图可视化
        if (this.currentData) {
            this.graphVisualizer.visualizeGraph(this.currentData);
        }
        
        this.showStatus('模型已重置', 'info');
        this.updateInterface();
    }

    /**
     * 更新可视化
     */
    async updateVisualization() {
        if (!this.currentModel || !this.currentData) return;
        
        try {
            // 获取预测结果
            let predictions = null;
            let attentionWeights = null;
            
            if (this.currentAlgorithm === 'gcn') {
                const output = this.currentModel.predict(
                    this.currentData.features,
                    this.currentData.normalizedAdjacency
                );
                predictions = tf.argMax(output, 1);
            } else if (this.currentAlgorithm === 'gat') {
                const output = this.currentModel.model.predict([
                    this.currentData.features,
                    this.currentData.normalizedAdjacency
                ]);
                predictions = tf.argMax(output, 1);
                attentionWeights = this.currentModel.getAttentionWeights();
            } else if (this.currentAlgorithm === 'graphsage') {
                // GraphSAGE需要特殊处理
                const embeddings = this.currentModel.generateEmbeddings(this.currentData);
                // 简化处理，直接使用嵌入的argmax
                predictions = tf.argMax(tf.tensor2d(embeddings), 1);
            }
            
            // 更新图可视化
            this.graphVisualizer.visualizeGraph(
                this.currentData,
                predictions,
                attentionWeights
            );
            
            // 更新注意力可视化（仅GAT）
            if (this.currentAlgorithm === 'gat' && attentionWeights) {
                this.attentionVisualizer.visualizeAttentionMatrix(attentionWeights);
            }
            
            // 更新嵌入可视化
            if (this.currentAlgorithm === 'graphsage') {
                const embeddings = this.currentModel.generateEmbeddings(this.currentData);
                const labels = this.currentData.labels.dataSync();
                this.embeddingVisualizer.visualizeEmbeddings(embeddings, labels);
            }
            
        } catch (error) {
            console.error('更新可视化失败:', error);
        }
    }

    /**
     * 切换预测显示
     */
    togglePredictions(show) {
        if (show && this.currentModel) {
            this.updateVisualization();
        } else if (this.currentData) {
            this.graphVisualizer.visualizeGraph(this.currentData);
        }
    }

    /**
     * 切换注意力可视化
     */
    toggleAttentionVisualization(show) {
        const attentionContainer = document.getElementById('attention-container');
        attentionContainer.style.display = show ? 'block' : 'none';
        
        if (show && this.currentAlgorithm === 'gat' && this.currentModel) {
            this.updateVisualization();
        }
    }

    /**
     * 更新训练状态显示
     */
    updateTrainingStatus(epoch, metrics) {
        document.getElementById('current-epoch').textContent = epoch + 1;
        document.getElementById('current-loss').textContent = metrics.loss.toFixed(4);
        document.getElementById('current-accuracy').textContent = (metrics.accuracy * 100).toFixed(2) + '%';
        document.getElementById('val-loss').textContent = metrics.valLoss.toFixed(4);
        document.getElementById('val-accuracy').textContent = (metrics.valAccuracy * 100).toFixed(2) + '%';
    }

    /**
     * 更新训练控制按钮
     */
    updateTrainingControls() {
        const startBtn = document.getElementById('start-training');
        const stopBtn = document.getElementById('stop-training');
        const resetBtn = document.getElementById('reset-model');
        
        if (this.isTraining) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            resetBtn.disabled = true;
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            resetBtn.disabled = false;
        }
    }

    /**
     * 绑定可视化切换事件
     */
    bindVisualizationTabs() {
        const tabs = document.querySelectorAll('.viz-tab');
        const panels = document.querySelectorAll('.viz-panel');
        
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const targetTab = tab.dataset.tab;
                
                // 更新标签状态
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                // 更新面板显示
                panels.forEach(panel => {
                    panel.classList.remove('active');
                    if (panel.id === `${targetTab}-panel`) {
                        panel.classList.add('active');
                    }
                });
                
                // 根据切换的标签更新相应的可视化
                this.updateVisualizationForTab(targetTab);
            });
        });
    }

    /**
     * 根据标签更新可视化
     */
    async updateVisualizationForTab(tab) {
        try {
            switch (tab) {
                case 'graph':
                    if (this.currentData) {
                        await this.updateVisualization();
                    }
                    break;
                case 'training':
                    // 训练曲线会在训练过程中自动更新
                    break;
                case 'embedding':
                    if (this.currentModel && this.currentData) {
                        await this.updateEmbeddingVisualization();
                    }
                    break;
            }
        } catch (error) {
            console.warn('更新可视化时出错:', error);
        }
    }

    /**
     * 更新节点嵌入可视化
     */
    async updateEmbeddingVisualization() {
        try {
            const embeddings = this.currentModel.generateEmbeddings(this.currentData);
            const labels = this.currentData.labels ? this.currentData.labels.dataSync() : null;
            this.embeddingVisualizer.visualizeEmbeddings(embeddings, labels);
        } catch (error) {
            console.warn('生成节点嵌入时出错:', error);
            this.embeddingVisualizer.showError('无法生成节点嵌入');
        }
    }

    /**
     * 清除错误信息
     */
    clearError() {
        const errorElement = document.getElementById('error-message');
        if (errorElement) {
            errorElement.textContent = '';
            errorElement.style.display = 'none';
        }
    }

    /**
     * 显示错误信息
     */
    showError(message) {
        const errorElement = document.getElementById('error-message');
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = 'block';
        }
        console.error('GNN Controller Error:', message);
    }

    /**
     * 更新状态信息
     */
    updateStatus(message) {
        const statusElement = document.getElementById('status-message');
        if (statusElement) {
            statusElement.textContent = message;
        }
        console.log('GNN Controller Status:', message);
    }

    /**
     * 更新界面状态
     */
    updateInterface() {
        // 更新参数显示
        const config = this.config[this.currentAlgorithm];
        
        document.getElementById('hidden-dim').value = config.hiddenDim;
        document.getElementById('learning-rate').value = config.learningRate;
        document.getElementById('epochs').value = config.epochs;
        
        // 更新算法特定参数
        if (this.currentAlgorithm === 'gcn') {
            const gcnDropout = document.getElementById('gcn-dropout');
            if (gcnDropout) gcnDropout.value = config.dropout;
        } else if (this.currentAlgorithm === 'gat') {
            const gatHeads = document.getElementById('gat-heads');
            const gatDropout = document.getElementById('gat-dropout');
            if (gatHeads) gatHeads.value = config.numHeads;
            if (gatDropout) gatDropout.value = config.dropout;
        } else if (this.currentAlgorithm === 'graphsage') {
            const sageAggregator = document.getElementById('sage-aggregator');
            const sageSample = document.getElementById('sage-sample-size');
            const sageLayers = document.getElementById('sage-num-layers');
            if (sageAggregator) sageAggregator.value = config.aggregator;
            if (sageSample) sageSample.value = config.sampleSize;
            if (sageLayers) sageLayers.value = config.numLayers;
        }
        
        // 更新训练控制
        this.updateTrainingControls();
    }

    /**
     * 显示状态消息
     */
    showStatus(message, type = 'info') {
        const statusElement = document.getElementById('status-message');
        statusElement.textContent = message;
        statusElement.className = `status-message ${type}`;
        
        // 自动隐藏成功和信息消息
        if (type === 'success' || type === 'info') {
            setTimeout(() => {
                statusElement.textContent = '';
                statusElement.className = 'status-message';
            }, 3000);
        }
    }

    /**
     * 导出模型
     */
    async exportModel() {
        if (!this.currentModel) {
            this.showStatus('没有可导出的模型', 'error');
            return;
        }
        
        try {
            // 导出模型权重
            const modelData = {
                algorithm: this.currentAlgorithm,
                config: this.config[this.currentAlgorithm],
                trainingHistory: this.trainingHistory,
                timestamp: new Date().toISOString()
            };
            
            const blob = new Blob([JSON.stringify(modelData, null, 2)], {
                type: 'application/json'
            });
            
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `${this.currentAlgorithm}_model_${Date.now()}.json`;
            link.click();
            URL.revokeObjectURL(url);
            
            this.showStatus('模型导出成功', 'success');
            
        } catch (error) {
            console.error('导出模型失败:', error);
            this.showStatus('导出模型失败: ' + error.message, 'error');
        }
    }

    /**
     * 导出训练数据
     */
    exportTrainingData() {
        this.trainingVisualizer.exportData();
        this.showStatus('训练数据导出成功', 'success');
    }
}

// 初始化控制器
let gnnController;

document.addEventListener('DOMContentLoaded', async () => {
    // 等待 tf 就绪（兼容备用CDN异步加载）
    const waitForTF = async (retries = 20, interval = 250) => {
        for (let i = 0; i < retries; i++) {
            if (typeof window !== 'undefined' && typeof window.tf !== 'undefined') return true;
            await new Promise(r => setTimeout(r, interval));
        }
        return false;
    };

    const ready = await waitForTF();
     gnnController = new GNNController();

     // 提供给 HTML 的全局包装函数
     window.startGCNTraining = () => { gnnController.selectAlgorithm('gcn'); gnnController.startTraining(); };
     window.startGATTraining = () => { gnnController.selectAlgorithm('gat'); gnnController.startTraining(); };
     window.startGraphSAGETraining = () => { gnnController.selectAlgorithm('graphsage'); gnnController.startTraining(); };
     window.resetGCN = () => { gnnController.selectAlgorithm('gcn'); gnnController.resetModel(); };
     window.resetGAT = () => { gnnController.selectAlgorithm('gat'); gnnController.resetModel(); };
     window.resetGraphSAGE = () => { gnnController.selectAlgorithm('graphsage'); gnnController.resetModel(); };

     if (!ready) {
         console.warn('TensorFlow.js 仍未检测到，但控制器将尝试在内部加载备用 CDN');
     }
    console.log('GNN在线实践平台已初始化');
});