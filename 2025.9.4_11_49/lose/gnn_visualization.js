/**
 * 图神经网络可视化模块
 * 基于D3.js和Plotly.js的交互式可视化实现
 */

class GNNVisualizer {
    constructor(containerId) {
        this.container = d3.select(`#${containerId}`);
        this.width = 800;
        this.height = 600;
        this.svg = null;
        this.simulation = null;
        this.currentData = null;
        this.colorScale = d3.scaleOrdinal(d3.schemeCategory10);
        this.nodeRadius = 8;
        this.linkDistance = 50;
        
        this.initializeVisualization();
    }

    /**
     * 初始化可视化容器
     */
    initializeVisualization() {
        this.container.selectAll('*').remove();
        
        this.svg = this.container
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height)
            .style('border', '1px solid #ddd')
            .style('border-radius', '8px');
        
        // 添加缩放和拖拽
        const zoom = d3.zoom()
            .scaleExtent([0.1, 3])
            .on('zoom', (event) => {
                this.svg.select('.graph-container')
                    .attr('transform', event.transform);
            });
        
        this.svg.call(zoom);
        
        // 创建图形容器
        this.graphContainer = this.svg
            .append('g')
            .attr('class', 'graph-container');
        
        // 创建链接和节点组
        this.linksGroup = this.graphContainer.append('g').attr('class', 'links');
        this.nodesGroup = this.graphContainer.append('g').attr('class', 'nodes');
        
        // 添加图例
        this.addLegend();
    }

    /**
     * 添加图例
     */
    addLegend() {
        const legend = this.svg
            .append('g')
            .attr('class', 'legend')
            .attr('transform', 'translate(20, 20)');
        
        legend.append('text')
            .attr('x', 0)
            .attr('y', 0)
            .style('font-size', '14px')
            .style('font-weight', 'bold')
            .text('节点类别');
    }

    /**
     * 可视化图数据
     */
    visualizeGraph(data, predictions = null, attentionWeights = null) {
        try {
            this.currentData = data;
            
            // 准备节点数据 - 添加WebGL错误处理
            const nodes = data.nodes.map((node, i) => {
                let nodeClass = 0;
                let prediction = null;
                let features = null;
                
                try {
                    nodeClass = data.labels ? data.labels.dataSync()[i] : 0;
                } catch (error) {
                    console.warn('获取节点标签失败，使用默认值:', error.message);
                    nodeClass = 0;
                }
                
                try {
                    prediction = predictions ? predictions.dataSync()[i] : null;
                } catch (error) {
                    console.warn('获取预测结果失败:', error.message);
                    prediction = null;
                }
                
                try {
                    features = data.features ? data.features.slice([i, 0], [1, -1]).dataSync() : null;
                } catch (error) {
                    console.warn('获取节点特征失败:', error.message);
                    features = null;
                }
                
                return {
                    id: i,
                    label: node.label || i,
                    class: nodeClass,
                    prediction: prediction,
                    x: Math.random() * this.width,
                    y: Math.random() * this.height,
                    features: features
                };
            });
        
            // 准备边数据 - 添加WebGL错误处理
            const links = [];
            if (data.edges) {
                try {
                    // 检查 edges 是张量还是普通数组
                    if (data.edges.dataSync) {
                        // TensorFlow.js 张量格式
                        const edgeData = data.edges.dataSync();
                        for (let i = 0; i < edgeData.length; i += 2) {
                            const source = edgeData[i];
                            const target = edgeData[i + 1];
                            let weight = 1;
                            
                            try {
                                weight = attentionWeights ? 
                                    attentionWeights.dataSync()[source * nodes.length + target] : 1;
                            } catch (error) {
                                console.warn('获取注意力权重失败，使用默认值:', error.message);
                                weight = 1;
                            }
                            
                            links.push({
                                source: source,
                                target: target,
                                weight: weight
                            });
                        }
                    } else {
                        // 普通数组格式
                        data.edges.forEach(edge => {
                            links.push({
                                source: edge.source,
                                target: edge.target,
                                weight: edge.weight || 1
                            });
                        });
                    }
                } catch (error) {
                    console.warn('处理边数据失败:', error.message);
                    // 使用空的边数组继续渲染
                }
            }
        
            // 创建力导向布局
            this.simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links).id(d => d.id).distance(this.linkDistance))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(this.width / 2, this.height / 2))
                .force('collision', d3.forceCollide().radius(this.nodeRadius + 2));
            
            // 绘制边
            this.drawLinks(links, attentionWeights);
            
            // 绘制节点
            this.drawNodes(nodes, predictions);
            
            // 更新图例
            this.updateLegend(data);
            
            // 启动仿真
            this.simulation.on('tick', () => {
                this.linksGroup.selectAll('line')
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                
                this.nodesGroup.selectAll('circle')
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);
                
                this.nodesGroup.selectAll('text')
                    .attr('x', d => d.x)
                    .attr('y', d => d.y + 4);
            });
            
        } catch (error) {
            console.error('图可视化失败:', error);
            
            // 检查是否为WebGL相关错误
            if (error.message && (
                error.message.includes('Failed to link vertex and fragment shaders') ||
                error.message.includes('WebGL') ||
                error.message.includes('GPU') ||
                error.message.includes('shader')
            )) {
                console.warn('检测到WebGL错误，尝试使用简化可视化模式');
                this.renderFallbackVisualization(data);
            } else {
                // 显示错误信息给用户
                this.showVisualizationError(error.message);
            }
        }
    }

    /**
     * 绘制边
     */
    drawLinks(links, attentionWeights) {
        const linkSelection = this.linksGroup
            .selectAll('line')
            .data(links);
        
        linkSelection.exit().remove();
        
        const linkEnter = linkSelection.enter()
            .append('line')
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.6);
        
        linkSelection.merge(linkEnter)
            .attr('stroke-width', d => {
                if (attentionWeights) {
                    return Math.max(0.5, d.weight * 5);
                }
                return 1;
            })
            .attr('stroke-opacity', d => {
                if (attentionWeights) {
                    return Math.max(0.1, d.weight);
                }
                return 0.6;
            });
    }

    /**
     * 备用可视化方案（当WebGL失败时使用）
     */
    renderFallbackVisualization(data) {
        try {
            // 清空容器
            this.container.selectAll('*').remove();
            
            // 创建简单的HTML表格显示
            const fallbackDiv = this.container
                .append('div')
                .style('padding', '20px')
                .style('background', '#f8f9fa')
                .style('border-radius', '8px')
                .style('border', '1px solid #dee2e6');
            
            fallbackDiv.append('h4')
                .style('color', '#495057')
                .style('margin-bottom', '15px')
                .text('图结构信息（简化显示）');
            
            fallbackDiv.append('p')
                .style('color', '#6c757d')
                .style('font-size', '14px')
                .style('margin-bottom', '10px')
                .text('由于WebGL不可用，使用简化显示模式');
            
            // 显示基本统计信息
            const statsDiv = fallbackDiv.append('div')
                .style('display', 'grid')
                .style('grid-template-columns', 'repeat(auto-fit, minmax(150px, 1fr))')
                .style('gap', '10px')
                .style('margin-top', '15px');
            
            const nodeCount = data.nodes ? data.nodes.length : 0;
            const edgeCount = data.edges ? (data.edges.dataSync ? data.edges.dataSync().length / 2 : data.edges.length) : 0;
            
            statsDiv.append('div')
                .style('background', 'white')
                .style('padding', '10px')
                .style('border-radius', '4px')
                .style('text-align', 'center')
                .html(`<strong>${nodeCount}</strong><br><small>节点数量</small>`);
            
            statsDiv.append('div')
                .style('background', 'white')
                .style('padding', '10px')
                .style('border-radius', '4px')
                .style('text-align', 'center')
                .html(`<strong>${edgeCount}</strong><br><small>边数量</small>`);
            
            console.log('备用可视化渲染完成');
            
        } catch (error) {
            console.error('备用可视化也失败了:', error);
            this.showVisualizationError('可视化功能暂时不可用');
        }
    }
    
    /**
     * 显示可视化错误信息
     */
    showVisualizationError(message) {
        this.container.selectAll('*').remove();
        
        const errorDiv = this.container
            .append('div')
            .style('padding', '20px')
            .style('text-align', 'center')
            .style('background', '#f8d7da')
            .style('border', '1px solid #f5c6cb')
            .style('border-radius', '8px')
            .style('color', '#721c24');
        
        errorDiv.append('h4')
            .style('margin-bottom', '10px')
            .text('可视化错误');
        
        errorDiv.append('p')
            .style('margin', '0')
            .text(message || '图可视化功能暂时不可用，请稍后重试');
    }

    /**
     * 绘制节点
     */
    drawNodes(nodes, predictions) {
        const nodeSelection = this.nodesGroup
            .selectAll('circle')
            .data(nodes);
        
        nodeSelection.exit().remove();
        
        const nodeEnter = nodeSelection.enter()
            .append('circle')
            .attr('r', this.nodeRadius)
            .call(this.dragHandler());
        
        // 更新节点颜色
        nodeSelection.merge(nodeEnter)
            .attr('fill', d => {
                if (predictions && d.prediction !== null) {
                    // 预测错误的节点用红色边框标识
                    return d.class === d.prediction ? 
                        this.colorScale(d.class) : '#ff6b6b';
                }
                return this.colorScale(d.class);
            })
            .attr('stroke', d => {
                if (predictions && d.prediction !== null && d.class !== d.prediction) {
                    return '#d63031';
                }
                return '#fff';
            })
            .attr('stroke-width', d => {
                if (predictions && d.prediction !== null && d.class !== d.prediction) {
                    return 3;
                }
                return 2;
            });
        
        // 添加节点标签
        const labelSelection = this.nodesGroup
            .selectAll('text')
            .data(nodes);
        
        labelSelection.exit().remove();
        
        const labelEnter = labelSelection.enter()
            .append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em')
            .style('font-size', '10px')
            .style('fill', '#333')
            .style('pointer-events', 'none');
        
        labelSelection.merge(labelEnter)
            .text(d => d.id);
        
        // 添加工具提示
        this.addTooltips(nodeSelection.merge(nodeEnter));
    }

    /**
     * 拖拽处理器
     */
    dragHandler() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }

    /**
     * 添加工具提示
     */
    addTooltips(selection) {
        selection
            .on('mouseover', (event, d) => {
                // 创建工具提示
                const tooltip = d3.select('body')
                    .append('div')
                    .attr('class', 'tooltip')
                    .style('position', 'absolute')
                    .style('background', 'rgba(0, 0, 0, 0.8)')
                    .style('color', 'white')
                    .style('padding', '8px')
                    .style('border-radius', '4px')
                    .style('font-size', '12px')
                    .style('pointer-events', 'none')
                    .style('z-index', 1000);
                
                let content = `节点 ${d.id}<br/>类别: ${d.class}`;
                if (d.prediction !== null) {
                    content += `<br/>预测: ${d.prediction}`;
                    content += `<br/>正确: ${d.class === d.prediction ? '是' : '否'}`;
                }
                
                tooltip.html(content)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 10) + 'px');
            })
            .on('mouseout', () => {
                d3.selectAll('.tooltip').remove();
            });
    }

    /**
     * 更新图例
     */
    updateLegend(data) {
        if (!data.labels) return;
        
        const classes = [...new Set(data.labels.dataSync())];
        
        const legend = this.svg.select('.legend');
        legend.selectAll('.legend-item').remove();
        
        const legendItems = legend.selectAll('.legend-item')
            .data(classes)
            .enter()
            .append('g')
            .attr('class', 'legend-item')
            .attr('transform', (d, i) => `translate(0, ${25 + i * 20})`);
        
        legendItems.append('circle')
            .attr('r', 6)
            .attr('fill', d => this.colorScale(d));
        
        legendItems.append('text')
            .attr('x', 15)
            .attr('y', 4)
            .style('font-size', '12px')
            .text(d => `类别 ${d}`);
    }

    /**
     * 高亮节点
     */
    highlightNodes(nodeIds, color = '#ff6b6b') {
        this.nodesGroup.selectAll('circle')
            .attr('stroke', (d, i) => {
                return nodeIds.includes(i) ? color : '#fff';
            })
            .attr('stroke-width', (d, i) => {
                return nodeIds.includes(i) ? 4 : 2;
            });
    }

    /**
     * 重置高亮
     */
    resetHighlight() {
        this.nodesGroup.selectAll('circle')
            .attr('stroke', '#fff')
            .attr('stroke-width', 2);
    }

    /**
     * 更新布局参数
     */
    updateLayout(params) {
        if (params.linkDistance) {
            this.linkDistance = params.linkDistance;
            if (this.simulation) {
                this.simulation.force('link').distance(this.linkDistance);
                this.simulation.alpha(0.3).restart();
            }
        }
        
        if (params.nodeRadius) {
            this.nodeRadius = params.nodeRadius;
            this.nodesGroup.selectAll('circle')
                .attr('r', this.nodeRadius);
        }
    }

    /**
     * 导出图像
     */
    exportImage(format = 'png') {
        const svgElement = this.svg.node();
        const serializer = new XMLSerializer();
        const svgString = serializer.serializeToString(svgElement);
        
        const canvas = document.createElement('canvas');
        canvas.width = this.width;
        canvas.height = this.height;
        const ctx = canvas.getContext('2d');
        
        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0);
            const link = document.createElement('a');
            link.download = `graph_visualization.${format}`;
            link.href = canvas.toDataURL(`image/${format}`);
            link.click();
        };
        
        img.src = 'data:image/svg+xml;base64,' + btoa(svgString);
    }
}

/**
 * 训练过程可视化器
 */
class TrainingVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.trainingData = {
            epochs: [],
            loss: [],
            accuracy: [],
            valLoss: [],
            valAccuracy: []
        };
        
        this.initializePlots();
    }

    /**
     * 初始化图表
     */
    initializePlots() {
        this.container.innerHTML = `
            <div class="training-plots">
                <div id="${this.containerId}_loss" class="plot-container"></div>
                <div id="${this.containerId}_accuracy" class="plot-container"></div>
            </div>
        `;
        
        // 损失图表
        this.lossPlot = {
            data: [
                {
                    x: [],
                    y: [],
                    type: 'scatter',
                    mode: 'lines',
                    name: '训练损失',
                    line: {color: '#e74c3c'}
                },
                {
                    x: [],
                    y: [],
                    type: 'scatter',
                    mode: 'lines',
                    name: '验证损失',
                    line: {color: '#3498db'}
                }
            ],
            layout: {
                title: '训练损失',
                xaxis: {title: '轮次'},
                yaxis: {title: '损失值'},
                margin: {l: 50, r: 50, t: 50, b: 50}
            }
        };
        
        // 准确率图表
        this.accuracyPlot = {
            data: [
                {
                    x: [],
                    y: [],
                    type: 'scatter',
                    mode: 'lines',
                    name: '训练准确率',
                    line: {color: '#27ae60'}
                },
                {
                    x: [],
                    y: [],
                    type: 'scatter',
                    mode: 'lines',
                    name: '验证准确率',
                    line: {color: '#f39c12'}
                }
            ],
            layout: {
                title: '训练准确率',
                xaxis: {title: '轮次'},
                yaxis: {title: '准确率', range: [0, 1]},
                margin: {l: 50, r: 50, t: 50, b: 50}
            }
        };
        
        Plotly.newPlot(`${this.containerId}_loss`, this.lossPlot.data, this.lossPlot.layout);
        Plotly.newPlot(`${this.containerId}_accuracy`, this.accuracyPlot.data, this.accuracyPlot.layout);
    }

    // 新增：确保Plotly图表与trace已就绪，避免无效索引
    ensurePlotsReady() {
        const lossId = `${this.containerId}_loss`;
        const accId = `${this.containerId}_accuracy`;
        if (!document.getElementById(lossId) || !document.getElementById(accId)) {
            this.initializePlots();
        }
        const lossEl = document.getElementById(lossId);
        const accEl = document.getElementById(accId);
        const lossLen = lossEl && lossEl.data ? lossEl.data.length : 0;
        const accLen = accEl && accEl.data ? accEl.data.length : 0;
        if (lossLen < 2) {
            Plotly.newPlot(lossId, this.lossPlot.data, this.lossPlot.layout);
        }
        if (accLen < 2) {
            Plotly.newPlot(accId, this.accuracyPlot.data, this.accuracyPlot.layout);
        }
    }

    /**
     * 更新训练数据
     */
    updateTrainingData(epoch, metrics) {
        this.trainingData.epochs.push(epoch);
        this.trainingData.loss.push(metrics.loss);
        this.trainingData.accuracy.push(metrics.accuracy);
        this.trainingData.valLoss.push(metrics.valLoss);
        this.trainingData.valAccuracy.push(metrics.valAccuracy);
        
        this.ensurePlotsReady();
        
        try {
            // 更新损失图表
            Plotly.extendTraces(`${this.containerId}_loss`, {
                x: [[epoch], [epoch]],
                y: [[metrics.loss], [metrics.valLoss]]
            }, [0, 1]);
            
            // 更新准确率图表
            Plotly.extendTraces(`${this.containerId}_accuracy`, {
                x: [[epoch], [epoch]],
                y: [[metrics.accuracy], [metrics.valAccuracy]]
            }, [0, 1]);
        } catch (err) {
            // 若出现索引错误，重置为初始两条trace再追加
            Plotly.react(`${this.containerId}_loss`, this.lossPlot.data, this.lossPlot.layout);
            Plotly.react(`${this.containerId}_accuracy`, this.accuracyPlot.data, this.accuracyPlot.layout);
            
            Plotly.extendTraces(`${this.containerId}_loss`, {
                x: [[epoch], [epoch]],
                y: [[metrics.loss], [metrics.valLoss]]
            }, [0, 1]);
            Plotly.extendTraces(`${this.containerId}_accuracy`, {
                x: [[epoch], [epoch]],
                y: [[metrics.accuracy], [metrics.valAccuracy]]
            }, [0, 1]);
        }
    }

    /**
     * 重置图表
     */
    reset() {
        this.trainingData = {
            epochs: [],
            loss: [],
            accuracy: [],
            valLoss: [],
            valAccuracy: []
        };
        
        this.ensurePlotsReady();
        // 使用react替代delete/add，避免索引错误
        Plotly.react(`${this.containerId}_loss`, this.lossPlot.data, this.lossPlot.layout);
        Plotly.react(`${this.containerId}_accuracy`, this.accuracyPlot.data, this.accuracyPlot.layout);
    }
}

/**
 * 注意力权重可视化器
 */
class AttentionVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
    }

    /**
     * 可视化注意力权重矩阵
     */
    visualizeAttentionMatrix(attentionWeights, nodeLabels = null) {
        const data = attentionWeights.dataSync();
        const size = Math.sqrt(data.length);
        
        // 重塑为矩阵
        const matrix = [];
        for (let i = 0; i < size; i++) {
            matrix.push(data.slice(i * size, (i + 1) * size));
        }
        
        const heatmapData = [{
            z: matrix,
            type: 'heatmap',
            colorscale: 'Viridis',
            showscale: true
        }];
        
        const layout = {
            title: '注意力权重矩阵',
            xaxis: {
                title: '目标节点',
                tickmode: 'array',
                tickvals: nodeLabels ? nodeLabels.map((_, i) => i) : null,
                ticktext: nodeLabels || null
            },
            yaxis: {
                title: '源节点',
                tickmode: 'array',
                tickvals: nodeLabels ? nodeLabels.map((_, i) => i) : null,
                ticktext: nodeLabels || null
            },
            margin: {l: 80, r: 80, t: 80, b: 80}
        };
        
        Plotly.newPlot(this.containerId, heatmapData, layout);
    }

    /**
     * 可视化节点注意力分布
     */
    visualizeNodeAttention(nodeId, attentionWeights, nodeLabels = null) {
        const data = attentionWeights.dataSync();
        const size = Math.sqrt(data.length);
        
        // 获取特定节点的注意力权重
        const nodeAttention = data.slice(nodeId * size, (nodeId + 1) * size);
        
        const barData = [{
            x: nodeLabels || Array.from({length: size}, (_, i) => `节点${i}`),
            y: nodeAttention,
            type: 'bar',
            marker: {
                color: nodeAttention,
                colorscale: 'Viridis'
            }
        }];
        
        const layout = {
            title: `节点 ${nodeId} 的注意力分布`,
            xaxis: {title: '目标节点'},
            yaxis: {title: '注意力权重'},
            margin: {l: 50, r: 50, t: 50, b: 100}
        };
        
        Plotly.newPlot(this.containerId, barData, layout);
    }
}

/**
 * 嵌入空间可视化器
 */
class EmbeddingVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
    }

    /**
     * 使用t-SNE可视化嵌入
     */
    visualizeEmbeddings(embeddings, labels = null, method = 'tsne') {
        // 简化的2D投影（实际应用中可以使用t-SNE或PCA库）
        const projected = this.projectTo2D(embeddings, method);
        
        const traces = [];
        
        if (labels) {
            const uniqueLabels = [...new Set(labels)];
            const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
            
            uniqueLabels.forEach(label => {
                const indices = labels.map((l, i) => l === label ? i : -1).filter(i => i !== -1);
                
                traces.push({
                    x: indices.map(i => projected[i][0]),
                    y: indices.map(i => projected[i][1]),
                    mode: 'markers',
                    type: 'scatter',
                    name: `类别 ${label}`,
                    marker: {
                        color: colorScale(label),
                        size: 8
                    },
                    text: indices.map(i => `节点 ${i}`),
                    hovertemplate: '%{text}<br>x: %{x}<br>y: %{y}<extra></extra>'
                });
            });
        } else {
            traces.push({
                x: projected.map(p => p[0]),
                y: projected.map(p => p[1]),
                mode: 'markers',
                type: 'scatter',
                name: '节点嵌入',
                marker: {
                    color: '#3498db',
                    size: 8
                },
                text: projected.map((_, i) => `节点 ${i}`),
                hovertemplate: '%{text}<br>x: %{x}<br>y: %{y}<extra></extra>'
            });
        }
        
        const layout = {
            title: `节点嵌入可视化 (${method.toUpperCase()})`,
            xaxis: {title: '维度 1'},
            yaxis: {title: '维度 2'},
            margin: {l: 50, r: 50, t: 50, b: 50},
            hovermode: 'closest'
        };
        
        Plotly.newPlot(this.containerId, traces, layout);
    }

    /**
     * 简化的2D投影
     */
    projectTo2D(embeddings, method) {
        // 这里实现简化的PCA投影
        // 实际应用中应该使用专门的降维库
        
        const numPoints = embeddings.length;
        const dim = embeddings[0].length;
        
        // 中心化数据
        const mean = new Array(dim).fill(0);
        for (let i = 0; i < numPoints; i++) {
            for (let j = 0; j < dim; j++) {
                mean[j] += embeddings[i][j];
            }
        }
        for (let j = 0; j < dim; j++) {
            mean[j] /= numPoints;
        }
        
        const centered = embeddings.map(emb => 
            emb.map((val, j) => val - mean[j])
        );
        
        // 简化投影到前两个主成分
        const projected = centered.map(emb => [
            emb[0] || Math.random() * 2 - 1,
            emb[1] || Math.random() * 2 - 1
        ]);
        
        return projected;
    }

    /**
     * 显示错误信息
     */
    showError(message) {
        const container = document.getElementById(this.containerId);
        if (container) {
            container.innerHTML = `
                <div class="error-message" style="
                    padding: 20px;
                    text-align: center;
                    color: #dc3545;
                    background: #f8d7da;
                    border: 1px solid #f5c6cb;
                    border-radius: 5px;
                    margin: 20px;
                ">
                    <h4>节点嵌入可视化错误</h4>
                    <p>${message}</p>
                </div>
            `;
        }
    }
}

// 导出可视化类
window.GNNVisualizer = GNNVisualizer;
window.TrainingVisualizer = TrainingVisualizer;
window.AttentionVisualizer = AttentionVisualizer;
window.EmbeddingVisualizer = EmbeddingVisualizer;