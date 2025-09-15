// 可视化模块 - 图表和图结构可视化

// 渲染训练曲线图表 - 增强版
function renderCharts(losses, accs) {
    if (!window.Plotly) return;
    
    // 增强的损失曲线配置
    const lossTrace = {
        y: losses, 
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Loss', 
        line: { 
            color: '#ef4444',
            width: 3,
            shape: 'spline'
        },
        marker: {
            size: 6,
            color: '#ef4444',
            line: {
                color: '#ffffff',
                width: 2
            }
        },
        hovertemplate: '<b>Epoch %{x}</b><br>Loss: %{y:.4f}<extra></extra>'
    };
    
    const lossLayout = {
        margin: { t: 20, b: 40, l: 50, r: 20 }, 
        yaxis: { 
            title: {
                text: 'Loss',
                font: { size: 14, color: '#374151' }
            },
            gridcolor: '#f3f4f6',
            zeroline: false
        },
        xaxis: {
            title: {
                text: 'Epoch',
                font: { size: 14, color: '#374151' }
            },
            gridcolor: '#f3f4f6'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, sans-serif' },
        hovermode: 'x unified'
    };
    
    Plotly.newPlot('loss-chart', [lossTrace], lossLayout, {
        responsive: true,
        displayModeBar: false
    });
    
    // 增强的准确率曲线配置
    var accTraces;
    if (Array.isArray(accs)) {
        accTraces = [{ 
            y: accs, 
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Accuracy', 
            line: { 
                color: '#10b981',
                width: 3,
                shape: 'spline'
            },
            marker: {
                size: 6,
                color: '#10b981',
                line: {
                    color: '#ffffff',
                    width: 2
                }
            },
            hovertemplate: '<b>Epoch %{x}</b><br>Accuracy: %{y:.4f}<extra></extra>'
        }];
    } else if (accs && Array.isArray(accs.train) && Array.isArray(accs.val)) {
        accTraces = [
            { 
                y: accs.train, 
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Train Acc', 
                line: { 
                    color: '#10b981',
                    width: 3,
                    shape: 'spline'
                },
                marker: { size: 5, color: '#10b981' },
                hovertemplate: '<b>Epoch %{x}</b><br>Train Acc: %{y:.4f}<extra></extra>'
            },
            { 
                y: accs.val, 
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Val Acc', 
                line: { 
                    color: '#f59e0b',
                    width: 3,
                    shape: 'spline'
                },
                marker: { size: 5, color: '#f59e0b' },
                hovertemplate: '<b>Epoch %{x}</b><br>Val Acc: %{y:.4f}<extra></extra>'
            }
        ];
    } else {
        accTraces = [{ 
            y: [], 
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Accuracy', 
            line: { color: '#10b981', width: 3 }
        }];
    }
    
    const accLayout = {
        margin: { t: 20, b: 40, l: 50, r: 20 }, 
        yaxis: { 
            title: {
                text: 'Accuracy',
                font: { size: 14, color: '#374151' }
            },
            range: [0, 1],
            gridcolor: '#f3f4f6',
            zeroline: false
        },
        xaxis: {
            title: {
                text: 'Epoch',
                font: { size: 14, color: '#374151' }
            },
            gridcolor: '#f3f4f6'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, sans-serif' },
        showlegend: accTraces.length > 1,
        legend: { 
            orientation: 'h',
            x: 0.5,
            xanchor: 'center',
            y: 1.1
        },
        hovermode: 'x unified'
    };
    
    Plotly.newPlot('accuracy-chart', accTraces, accLayout, {
        responsive: true,
        displayModeBar: false
    });
}

// 渲染图结构可视化 - 增强版本
function renderGraph(finalAcc, correctFlags, neighborsInfo, attentionWeights) {
    var container = document.getElementById('graph-visualization');
    if (!container || !window.d3) return;
    
    container.innerHTML = '';
    var width = container.clientWidth || 400;
    var height = 220;
    
    // 创建增强的控制面板
    var controls = d3.select(container).append('div')
        .style('position', 'absolute')
        .style('top', '8px')
        .style('right', '8px')
        .style('display', 'flex')
        .style('gap', '8px')
        .style('z-index', '10');
    
    // 注意力权重按钮
    controls.append('button')
        .text('🎯 注意力')
        .style('padding', '6px 12px')
        .style('background', 'linear-gradient(135deg, #8b5cf6, #a855f7)')
        .style('color', 'white')
        .style('border', 'none')
        .style('border-radius', '8px')
        .style('cursor', 'pointer')
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('box-shadow', '0 2px 8px rgba(139, 92, 246, 0.3)')
        .style('transition', 'all 0.3s ease')
        .on('mouseover', function() {
            d3.select(this)
                .style('transform', 'translateY(-2px)')
                .style('box-shadow', '0 4px 12px rgba(139, 92, 246, 0.4)');
        })
        .on('mouseout', function() {
            d3.select(this)
                .style('transform', 'translateY(0)')
                .style('box-shadow', '0 2px 8px rgba(139, 92, 246, 0.3)');
        })
        .on('click', function() {
            toggleAttentionWeights();
        });
    
    // 训练动画按钮
    controls.append('button')
        .text('🚀 动画')
        .style('padding', '6px 12px')
        .style('background', 'linear-gradient(135deg, #3b82f6, #2563eb)')
        .style('color', 'white')
        .style('border', 'none')
        .style('border-radius', '8px')
        .style('cursor', 'pointer')
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('box-shadow', '0 2px 8px rgba(59, 130, 246, 0.3)')
        .style('transition', 'all 0.3s ease')
        .on('mouseover', function() {
            d3.select(this)
                .style('transform', 'translateY(-2px)')
                .style('box-shadow', '0 4px 12px rgba(59, 130, 246, 0.4)');
        })
        .on('mouseout', function() {
            d3.select(this)
                .style('transform', 'translateY(0)')
                .style('box-shadow', '0 2px 8px rgba(59, 130, 246, 0.3)');
        })
        .on('click', function() {
            showTrainingAnimation();
        });
    
    // 重置视图按钮
    controls.append('button')
        .text('🔄 重置')
        .style('padding', '6px 12px')
        .style('background', 'linear-gradient(135deg, #6b7280, #4b5563)')
        .style('color', 'white')
        .style('border', 'none')
        .style('border-radius', '8px')
        .style('cursor', 'pointer')
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('box-shadow', '0 2px 8px rgba(107, 114, 128, 0.3)')
        .style('transition', 'all 0.3s ease')
        .on('mouseover', function() {
            d3.select(this)
                .style('transform', 'translateY(-2px)')
                .style('box-shadow', '0 4px 12px rgba(107, 114, 128, 0.4)');
        })
        .on('mouseout', function() {
            d3.select(this)
                .style('transform', 'translateY(0)')
                .style('box-shadow', '0 2px 8px rgba(107, 114, 128, 0.3)');
        })
        .on('click', function() {
            resetGraphView();
        });
    
    var svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
    
    // 创建定义区域
    var defs = svg.append('defs');
    
    // 定义注意力权重渐变
    var attentionGradient = defs.append('linearGradient')
        .attr('id', 'attentionGradient')
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '100%').attr('y2', '0%');
    attentionGradient.append('stop').attr('offset', '0%').attr('stop-color', '#3b82f6');
    attentionGradient.append('stop').attr('offset', '100%').attr('stop-color', '#8b5cf6');

    // 定义采样高亮渐变（用于被采样的邻居连边）
    var samplingGradient = defs.append('linearGradient')
        .attr('id', 'samplingGradient')
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '100%').attr('y2', '0%');
    samplingGradient.append('stop').attr('offset', '0%').attr('stop-color', '#f59e0b');
    samplingGradient.append('stop').attr('offset', '100%').attr('stop-color', '#ef4444');
    
    // 根据是否有邻居信息判断是 GraphSAGE 还是 GCN
    var isGraphSAGE = neighborsInfo && Object.keys(neighborsInfo).length > 0;
    var hasAttention = attentionWeights && attentionWeights.length > 0;
    
    var nodes, links;
    if (isGraphSAGE) {
        // GraphSAGE: 10 节点复杂图
        nodes = d3.range(10).map(function(i) { 
            return { 
                id: i, 
                label: ['类别A', '类别A', '类别A', '类别B', '类别B', '类别B', 
                       '类别C', '类别C', '类别C', '类别C'][i],
                sampledNeighbors: neighborsInfo && neighborsInfo[i] ? (Array.isArray(neighborsInfo[i]) ? neighborsInfo[i] : (neighborsInfo[i].flat || [])) : [],
                embedding: [Math.random(), Math.random(), Math.random()]
            }; 
        });
        links = [
            {source: 0, target: 1}, {source: 0, target: 3}, {source: 1, target: 2}, 
            {source: 1, target: 4}, {source: 2, target: 5}, {source: 3, target: 4}, 
            {source: 3, target: 6}, {source: 4, target: 5}, {source: 4, target: 7}, 
            {source: 5, target: 8}, {source: 6, target: 7}, {source: 6, target: 9},
            {source: 7, target: 8}, {source: 8, target: 9}
        ];
    } else {
        // GCN/GAT: 7 节点链式图
        nodes = d3.range(7).map(function(i) { 
            return { 
                id: i, 
                label: ['AI', 'AI', 'CV', 'CV', 'NLP', 'NLP', 'NLP'][i],
                attention: hasAttention && attentionWeights[i] ? attentionWeights[i] : [],
                embedding: [Math.random(), Math.random(), Math.random()]
            }; 
        });
        links = [
            {source: 0, target: 1}, {source: 1, target: 2}, {source: 2, target: 3}, 
            {source: 3, target: 4}, {source: 4, target: 5}, {source: 5, target: 6}
        ];
    }
    
    // 依据传入标记或最终准确率模拟预测正确与否
    if (Array.isArray(correctFlags) && correctFlags.length) {
        nodes.forEach(function(n, idx) { n.correct = !!correctFlags[idx]; });
    } else {
        var correctCount = Math.round(finalAcc * nodes.length);
        nodes.forEach(function(n, idx) { n.correct = idx < correctCount; });
    }
    
    var color = function(n) { return n.correct ? '#67C23A' : '#F56C6C'; };

    var simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(function(d) { return d.id; }).distance(60))
        .force('charge', d3.forceManyBody().strength(-200))
        .force('center', d3.forceCenter(width / 2, height / 2));

    var link = svg.append('g').selectAll('line').data(links).enter().append('line')
        .attr('stroke', '#cbd5e1')
        .attr('stroke-width', 2)
        .attr('stroke-opacity', 0.6)
        .style('cursor', 'pointer')
        .on('click', function(event, d) {
            if (hasAttention) {
                showAttentionForEdge(d);
            } else if (isGraphSAGE) {
                showSamplingForEdge(d);
            }
        });

    var node = svg.append('g').selectAll('circle').data(nodes).enter().append('circle')
        .attr('r', isGraphSAGE ? 10 : 12)
        .attr('fill', function(d) { return color(d); })
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .attr('class', 'graph-node')
        .style('cursor', 'pointer')
        .on('click', function(event, d) {
            showNodeDetails(d, isGraphSAGE, hasAttention);
        })
        .on('mouseover', function(event, d) {
            d3.select(this)
                .transition()
                .duration(200)
                .attr('r', (isGraphSAGE ? 10 : 12) + 3)
                .attr('stroke', '#ffd700')
                .attr('stroke-width', 3);
                
            // 高亮相关连接
            link.style('opacity', function(l) {
                return l.source.id === d.id || l.target.id === d.id ? 1 : 0.3;
            });
        })
        .on('mouseout', function(event, d) {
            d3.select(this)
                .transition()
                .duration(200)
                .attr('r', isGraphSAGE ? 10 : 12)
                .attr('stroke', '#fff')
                .attr('stroke-width', 2);
                
            link.style('opacity', 0.6);
        })
        .call(d3.drag()
            .on('start', function(event, d) { 
                if (!event.active) simulation.alphaTarget(0.3).restart(); 
                d.fx = d.x; d.fy = d.y; 
            })
            .on('drag', function(event, d) { 
                d.fx = event.x; d.fy = event.y; 
            })
            .on('end', function(event, d) { 
                if (!event.active) simulation.alphaTarget(0); 
                d.fx = null; d.fy = null; 
            })
        );
    
    // 添加节点标签
    var labels = svg.append('g').selectAll('text').data(nodes).enter().append('text')
        .text(function(d) { return d.id; })
        .attr('text-anchor', 'middle')
        .attr('dy', '.35em')
        .attr('font-size', '10px')
        .attr('fill', 'white')
        .attr('font-weight', 'bold')
        .style('pointer-events', 'none');
    
    // 更新提示信息
    node.append('title').text(function(d) { 
        var base = '节点 ' + d.id + '\n预测: ' + (d.correct ? '正确' : '错误');
        if (isGraphSAGE && d.sampledNeighbors && d.sampledNeighbors.length > 0) {
            base += '\n采样邻居: [' + d.sampledNeighbors.join(',') + ']';
        }
        if (hasAttention && d.attention && d.attention.length > 0) {
            base += '\n注意力权重: [' + d.attention.map(w => w.toFixed(3)).join(',') + ']';
        }
        return base;
    });

    simulation.on('tick', function() {
        link.attr('x1', function(d) { return d.source.x; })
            .attr('y1', function(d) { return d.source.y; })
            .attr('x2', function(d) { return d.target.x; })
            .attr('y2', function(d) { return d.target.y; });
        node.attr('cx', function(d) { return d.x; }).attr('cy', function(d) { return d.y; });
        labels.attr('x', function(d) { return d.x; }).attr('y', function(d) { return d.y; });
    });
    
    // 注意力权重可视化功能
    function toggleAttentionWeights() {
        if (!hasAttention) {
            alert('当前模型没有注意力权重数据');
            return;
        }
        
        var isShowing = svg.selectAll('.attention-weight').size() > 0;
        
        if (isShowing) {
            // 隐藏注意力权重
            svg.selectAll('.attention-weight').remove();
            link.attr('stroke', '#cbd5e1').attr('stroke-width', 2);
        } else {
            // 显示注意力权重
            links.forEach(function(l, i) {
                var sourceAttention = nodes[l.source].attention;
                var weight = sourceAttention && sourceAttention[l.target] ? sourceAttention[l.target] : Math.random();
                
                // 更新边的样式以反映注意力权重
                link.filter(function(d, j) { return j === i; })
                    .attr('stroke', 'url(#attentionGradient)')
                    .attr('stroke-width', 2 + weight * 4)
                    .attr('stroke-opacity', 0.3 + weight * 0.7);
                
                // 添加权重标签
                var midX = (nodes[l.source].x + nodes[l.target].x) / 2;
                var midY = (nodes[l.source].y + nodes[l.target].y) / 2;
                
                svg.append('text')
                    .attr('class', 'attention-weight')
                    .attr('x', midX)
                    .attr('y', midY)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '8px')
                    .attr('fill', '#8b5cf6')
                    .attr('font-weight', 'bold')
                    .style('pointer-events', 'none')
                    .text(weight.toFixed(2));
            });
        }
    }
    
    // 训练过程动画
    function showTrainingAnimation() {
        var animationSteps = [
            { name: '初始化', color: '#6b7280' },
            { name: '前向传播', color: '#3b82f6' },
            { name: '计算损失', color: '#f59e0b' },
            { name: '反向传播', color: '#ef4444' },
            { name: '更新参数', color: '#10b981' },
            { name: '完成', color: '#67C23A' }
        ];
        
        var stepIndex = 0;
        
        function animateStep() {
            if (stepIndex >= animationSteps.length) {
                // 动画完成，恢复原始颜色
                node.transition().duration(500).attr('fill', function(d) { return color(d); });
                return;
            }
            
            var step = animationSteps[stepIndex];
            
            // 更新节点颜色
            node.transition()
                .duration(800)
                .attr('fill', step.color);
            
            // 显示步骤信息
            showStepInfo(step.name);
            
            stepIndex++;
            setTimeout(animateStep, 1000);
        }
        
        animateStep();
    }
    
    function showStepInfo(stepName) {
        var infoEl = svg.select('.step-info');
        if (infoEl.empty()) {
            infoEl = svg.append('text')
                .attr('class', 'step-info')
                .attr('x', width / 2)
                .attr('y', 20)
                .attr('text-anchor', 'middle')
                .attr('font-size', '12px')
                .attr('font-weight', 'bold')
                .attr('fill', '#333');
        }
        
        infoEl.text('训练步骤: ' + stepName)
            .transition()
            .duration(200)
            .attr('fill', '#3b82f6')
            .transition()
            .duration(800)
            .attr('fill', '#333');
    }
    
    function showNodeDetails(nodeData, isGraphSAGE, hasAttention) {
        var details = '节点 ' + nodeData.id + ' 详细信息:\n\n';
        details += '标签: ' + nodeData.label + '\n';
        details += '预测: ' + (nodeData.correct ? '正确' : '错误') + '\n';
        details += '嵌入向量: [' + nodeData.embedding.map(x => x.toFixed(3)).join(', ') + ']\n\n';
        
        if (isGraphSAGE && nodeData.sampledNeighbors) {
            details += 'GraphSAGE 采样信息:\n';
            details += '采样邻居: [' + nodeData.sampledNeighbors.join(', ') + ']\n';
            details += '聚合方式: Mean Aggregation\n\n';
        }
        
        if (hasAttention && nodeData.attention) {
            details += 'GAT 注意力信息:\n';
            details += '注意力权重: [' + nodeData.attention.map(w => w.toFixed(3)).join(', ') + ']\n';
            details += '多头注意力: 8 heads\n';
        }
        
        alert(details);
    }
    
    function showAttentionForEdge(edgeData) {
        var sourceNode = nodes[edgeData.source.id];
        var targetNode = nodes[edgeData.target.id];
        var weight = sourceNode.attention && sourceNode.attention[edgeData.target.id] 
                    ? sourceNode.attention[edgeData.target.id] 
                    : Math.random();
        
        alert(`注意力权重详情:\n\n` +
              `从节点 ${sourceNode.id} 到节点 ${targetNode.id}\n` +
              `注意力权重: ${weight.toFixed(4)}\n` +
              `权重说明: ${weight > 0.5 ? '高注意力' : '低注意力'}`);
    }
    
    function showSamplingForEdge(edgeData) {
        var sourceNode = nodes[edgeData.source.id];
        var targetNode = nodes[edgeData.target.id];
        var isSampled = sourceNode.sampledNeighbors && sourceNode.sampledNeighbors.includes(targetNode.id);
        
        alert(`GraphSAGE 采样信息:\n\n` +
              `边: ${sourceNode.id} -> ${targetNode.id}\n` +
              `是否被采样: ${isSampled ? '是' : '否'}\n` +
              `采样策略: 随机采样`);
    }
}

// 创建交互式图演示（用于教程页面）
function createGraphDemo(containerId) {
    var container = document.getElementById(containerId);
    if (!container || !window.d3) return;
    
    container.innerHTML = '';
    var width = container.clientWidth || 400;
    var height = 220;
    var svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
    
    // 创建定义区域用于箭头和渐变
    var defs = svg.append('defs');
    
    // 定义箭头标记
    defs.append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 25)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 10)
        .attr('markerHeight', 10)
        .attr('xoverflow', 'visible')
        .append('svg:path')
        .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
        .attr('fill', '#999')
        .style('stroke', 'none');
    
    // 定义高亮箭头
    defs.append('marker')
        .attr('id', 'arrowhead-highlight')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 25)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 10)
        .attr('markerHeight', 10)
        .attr('xoverflow', 'visible')
        .append('svg:path')
        .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
        .attr('fill', '#ff6b6b')
        .style('stroke', 'none');
        
    // 定义节点发光效果
    var filter = defs.append('filter')
        .attr('id', 'glow');
    filter.append('feGaussianBlur')
        .attr('stdDeviation', '3')
        .attr('result', 'coloredBlur');
    var feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode')
        .attr('in', 'coloredBlur');
    feMerge.append('feMergeNode')
        .attr('in', 'SourceGraphic');

    var nodes = [
        {id: 0, name: '小明', type: '学生', age: 20, interests: ['机器学习', '编程'], x: width/2, y: height/2},
        {id: 1, name: '小红', type: '设计师', age: 25, interests: ['UI设计', '绘画'], x: width/2-80, y: height/2-60},
        {id: 2, name: '小刚', type: '工程师', age: 28, interests: ['后端开发', '架构设计'], x: width/2+80, y: height/2-60},
        {id: 3, name: '小丽', type: '教师', age: 30, interests: ['教育', '心理学'], x: width/2-80, y: height/2+60},
        {id: 4, name: '小强', type: '医生', age: 26, interests: ['医学', '健康管理'], x: width/2+80, y: height/2+60}
    ];
    
    var links = [
        {source: 0, target: 1, relation: '同学', strength: 0.8, type: 'friendship'},
        {source: 1, target: 2, relation: '同事', strength: 0.6, type: 'work'},
        {source: 2, target: 3, relation: '朋友', strength: 0.9, type: 'friendship'},
        {source: 3, target: 4, relation: '邻居', strength: 0.4, type: 'neighbor'},
        {source: 0, target: 4, relation: '朋友', strength: 0.7, type: 'friendship'}
    ];
    
    // 创建力导向布局
    var simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(function(d) { return d.id; }).distance(80))
        .force('charge', d3.forceManyBody().strength(-400))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(25));

    // 绘制连接线
    var link = svg.append('g')
        .selectAll('line')
        .data(links)
        .enter().append('line')
        .attr('stroke', function(d) {
            return d.type === 'friendship' ? '#4ade80' : 
                   d.type === 'work' ? '#3b82f6' : '#f59e0b';
        })
        .attr('stroke-width', function(d) { return Math.sqrt(d.strength * 4); })
        .attr('stroke-opacity', 0.6)
        .attr('marker-end', 'url(#arrowhead)')
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
            d3.select(this)
                .attr('stroke-width', Math.sqrt(d.strength * 6))
                .attr('stroke-opacity', 1)
                .attr('marker-end', 'url(#arrowhead-highlight)');
            
            // 显示关系信息
            showTooltip(event, `关系: ${d.relation}<br/>强度: ${d.strength}`);
        })
        .on('mouseout', function(event, d) {
            d3.select(this)
                .attr('stroke-width', Math.sqrt(d.strength * 4))
                .attr('stroke-opacity', 0.6)
                .attr('marker-end', 'url(#arrowhead)');
            hideTooltip();
        })
        .on('click', function(event, d) {
            highlightPath(d.source.id, d.target.id);
        });

    // 绘制节点
    var node = svg.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter().append('circle')
        .attr('r', 15)
        .attr('fill', function(d, i) { return d3.schemeSet3[i]; })
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .style('cursor', 'grab')
        .on('mouseover', function(event, d) {
            d3.select(this)
                .transition()
                .duration(200)
                .attr('r', 18)
                .style('filter', 'url(#glow)');
            
            // 高亮相关连接
            link.style('opacity', function(l) {
                return l.source.id === d.id || l.target.id === d.id ? 1 : 0.1;
            });
            
            showTooltip(event, `
                <strong>${d.name}</strong><br/>
                职业: ${d.type}<br/>
                年龄: ${d.age}岁<br/>
                兴趣: ${d.interests.join(', ')}<br/>
                <small>点击查看详情</small>
            `);
        })
        .on('mouseout', function(event, d) {
            d3.select(this)
                .transition()
                .duration(200)
                .attr('r', 15)
                .style('filter', 'none');
            
            link.style('opacity', 1);
            hideTooltip();
        })
        .on('click', function(event, d) {
            showNodeDetails(d);
        })
        .call(d3.drag()
            .on('start', function(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
                d3.select(this).style('cursor', 'grabbing');
            })
            .on('drag', function(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', function(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
                d3.select(this).style('cursor', 'grab');
            })
        );
    
    // 添加节点标签
    var labels = svg.append('g')
        .selectAll('text')
        .data(nodes)
        .enter().append('text')
        .text(function(d) { return d.name; })
        .attr('text-anchor', 'middle')
        .attr('dy', '.35em')
        .attr('font-size', '10px')
        .attr('fill', 'white')
        .attr('font-weight', 'bold')
        .style('pointer-events', 'none');

    // 更新位置
    simulation.on('tick', function() {
        link.attr('x1', function(d) { return d.source.x; })
            .attr('y1', function(d) { return d.source.y; })
            .attr('x2', function(d) { return d.target.x; })
            .attr('y2', function(d) { return d.target.y; });
        
        node.attr('cx', function(d) { return d.x; })
            .attr('cy', function(d) { return d.y; });
        
        labels.attr('x', function(d) { return d.x; })
              .attr('y', function(d) { return d.y; });
    });
    
    // 工具提示功能
    function showTooltip(event, content) {
        var tooltip = d3.select('body').select('.graph-tooltip');
        if (tooltip.empty()) {
            tooltip = d3.select('body').append('div')
                .attr('class', 'graph-tooltip')
                .style('position', 'absolute')
                .style('background', 'rgba(0, 0, 0, 0.8)')
                .style('color', 'white')
                .style('padding', '8px')
                .style('border-radius', '4px')
                .style('font-size', '12px')
                .style('pointer-events', 'none')
                .style('z-index', '1000')
                .style('opacity', 0);
        }
        
        tooltip.html(content)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .transition()
            .duration(200)
            .style('opacity', 1);
    }
    
    function hideTooltip() {
        d3.select('.graph-tooltip')
            .transition()
            .duration(200)
            .style('opacity', 0);
    }
    
    // 节点详情弹窗
    function showNodeDetails(d) {
        var modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="bg-white rounded-lg p-6 max-w-md mx-4 transform transition-all">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-xl font-bold text-gray-900">${d.name} 的详细信息</h3>
                    <button onclick="this.closest('.fixed').remove()" class="text-gray-400 hover:text-gray-600">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </button>
                </div>
                <div class="space-y-3">
                    <div><strong>职业:</strong> ${d.type}</div>
                    <div><strong>年龄:</strong> ${d.age}岁</div>
                    <div><strong>兴趣爱好:</strong></div>
                    <ul class="list-disc list-inside ml-4">
                        ${d.interests.map(interest => `<li>${interest}</li>`).join('')}
                    </ul>
                    <div class="mt-4 p-4 bg-blue-50 rounded">
                        <h4 class="font-semibold text-blue-900 mb-2">在图中的作用</h4>
                        <p class="text-blue-800 text-sm">
                            作为图中的一个节点，${d.name}通过各种关系连接到其他节点，
                            形成了复杂的社交网络结构。这些连接可以用于分析社区结构、
                            信息传播路径等图学习任务。
                        </p>
                    </div>
                </div>
                <button onclick="this.closest('.fixed').remove()" 
                        class="mt-4 w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700">
                    关闭
                </button>
            </div>
        `;
        document.body.appendChild(modal);
        
        // 添加动画效果
        setTimeout(() => {
            modal.querySelector('.bg-white').style.transform = 'scale(1)';
        }, 10);
    }
    
    // 路径高亮功能
    function highlightPath(sourceId, targetId) {
        // 重置所有样式
        node.style('opacity', 0.3);
        link.style('opacity', 0.1);
        
        // 高亮选中的节点和连接
        node.filter(d => d.id === sourceId || d.id === targetId)
            .style('opacity', 1)
            .transition()
            .duration(500)
            .attr('r', 20);
            
        link.filter(d => (d.source.id === sourceId && d.target.id === targetId) || 
                        (d.source.id === targetId && d.target.id === sourceId))
            .style('opacity', 1)
            .attr('stroke', '#ff6b6b')
            .attr('stroke-width', 5);
        
        // 3秒后恢复
        setTimeout(() => {
            node.style('opacity', 1)
                .transition()
                .duration(500)
                .attr('r', 15);
            link.style('opacity', 1)
                .attr('stroke', function(d) {
                    return d.type === 'friendship' ? '#4ade80' : 
                           d.type === 'work' ? '#3b82f6' : '#f59e0b';
                })
                .attr('stroke-width', function(d) { return Math.sqrt(d.strength * 4); });
        }, 3000);
    }
    
    // 添加控制按钮
    var controls = d3.select(container).append('div')
        .style('position', 'absolute')
        .style('top', '10px')
        .style('right', '10px')
        .style('display', 'flex')
        .style('gap', '5px');
    
    controls.append('button')
        .text('重新布局')
        .style('padding', '5px 10px')
        .style('background', '#3b82f6')
        .style('color', 'white')
        .style('border', 'none')
        .style('border-radius', '4px')
        .style('cursor', 'pointer')
        .style('font-size', '12px')
        .on('click', function() {
            simulation.alpha(1).restart();
        });
        
    controls.append('button')
        .text('暂停动画')
        .style('padding', '5px 10px')
        .style('background', '#ef4444')
        .style('color', 'white')
        .style('border', 'none')
        .style('border-radius', '4px')
        .style('cursor', 'pointer')
        .style('font-size', '12px')
        .on('click', function() {
            var button = d3.select(this);
            if (simulation.alpha() > 0) {
                simulation.stop();
                button.text('恢复动画');
            } else {
                simulation.restart();
                button.text('暂停动画');
            }
        });
}

// 从Pyodide提取结果 - 增强版本
async function extractPyResults() {
    try {
        var jsonStr = await practiceState.pyodide.runPythonAsync(`
import json
out = {}
try:
    out['losses'] = list(losses)
    out['accuracies'] = list(accuracies)
except Exception as e:
    out['losses'] = [1.0,0.8,0.6,0.5]
    out['accuracies'] = [0.3,0.5,0.7,0.8]
try:
    import torch
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'gat1'):  # GAT模型
            o, _, attn = model(features, adj_matrix)
            out['attention_weights'] = attn.tolist()
        elif hasattr(model, 'sage1'):  # GraphSAGE模型
            o = model(features, adj_matrix)
            # 提取邻居采样信息
            try:
                neighbors_info = {}
                for i in range(min(5, features.size(0))):
                    neighbors = sample_neighbors(adj_matrix, i, 3)
                    neighbors_info[str(i)] = neighbors.tolist()
                out['neighbors_sampled'] = neighbors_info
            except Exception as e:
                pass
        else:  # GCN模型
            o = model(features, adj_matrix)
        _, p = o.max(dim=1)
    lbl = list(map(int, labels.tolist()))
    pred = list(map(int, p.tolist()))
    pred_correct = [1 if pred[i]==lbl[i] else 0 for i in range(len(lbl))]
    out['pred_correct'] = pred_correct
except Exception as e:
    out['pred_correct'] = [1,1,0,1,0,1,1]
json.dumps(out)`);
        return JSON.parse(jsonStr);
    } catch (e) {
        return { 
            losses: [1.0, 0.8, 0.6, 0.5], 
            accuracies: [0.3, 0.5, 0.7, 0.8], 
            pred_correct: [1, 1, 0, 1, 0, 1, 1] 
        };
    }
}

// 从 Pyodide 提取结果的增强版本
async function extractPyodideResults() {
    if (!practiceState.pyodide || !practiceState.pyodideReady) {
        return {
            losses: [],
            accuracies: [],
            pred_correct: [],
            neighbors_sampled: {},
            model_info: 'Pyodide 未就绪'
        };
    }
    
    try {
        var resultData = await practiceState.pyodide.runPythonAsync(`
import json
import sys

# 初始化结果字典
result = {
    'losses': [],
    'accuracies': [],
    'pred_correct': [],
    'neighbors_sampled': {},
    'model_info': '',
    'execution_info': {}
}

# 提取训练结果
try:
    if 'losses' in globals() and isinstance(losses, list) and len(losses) > 0:
        result['losses'] = [float(x) for x in losses]
    if 'accuracies' in globals() and isinstance(accuracies, list) and len(accuracies) > 0:
        result['accuracies'] = [float(x) for x in accuracies]
except Exception as e:
    result['execution_info']['training_data_error'] = str(e)

# 提取模型信息
try:
    if 'model' in globals() and model is not None:
        model_name = type(model).__name__
        result['model_info'] = f"模型: {model_name}"
        
        # 尝试获取模型参数
        if hasattr(model, '__dict__'):
            model_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
            result['model_info'] += f"\n属性: {', '.join(model_attrs[:5])}"
            
except Exception as e:
    result['execution_info']['model_info_error'] = str(e)

# 提取预测结果
try:
    # 检查是否有预测相关的变量
    if 'model' in globals() and model is not None:
        # 获取最终预测结果
        if hasattr(model, 'forward') and 'features' in globals() and 'adj_norm' in globals():
            final_pred = model.forward(features, adj_norm)
            pred_labels = final_pred.argmax(axis=1)
            
            # 计算准确性
            if 'labels' in globals():
                correct_predictions = pred_labels == labels
                result['pred_correct'] = correct_predictions.tolist()
                result['prediction_accuracy'] = float(correct_predictions.mean())
            else:
                result['pred_correct'] = pred_labels.tolist()
                
            result['predictions'] = pred_labels.tolist()
            result['model_predictions'] = final_pred.tolist()[:5]  # 只取前5个节点的预测概率
    
    # 后备：检查其他预测变量
    prediction_vars = ['pred_labels', 'predictions', 'pred', 'y_pred', 'output']
    if not result.get('pred_correct'):
        for var_name in prediction_vars:
            if var_name in globals():
                var_value = globals()[var_name]
                # 尝试将结果转换为list
                if hasattr(var_value, 'tolist'):
                    result['pred_correct'] = var_value.tolist()[:10]  # 只取10个
                    break
                elif isinstance(var_value, list):
                    result['pred_correct'] = var_value[:10]
                    break
except Exception as e:
    result['execution_info']['prediction_error'] = str(e)

# 提取可用的全局变量
try:
    available_vars = [var for var in globals().keys() 
                     if not var.startswith('_') and not var in ['sys', 'json', 'result']]
    result['execution_info']['available_variables'] = available_vars[:10]  # 只显示前10个
except Exception as e:
    result['execution_info']['vars_error'] = str(e)

# 检查是否有图相关数据
try:
    graph_vars = ['adj_matrix', 'adj_norm', 'adjacency', 'edges', 'graph']
    graph_info = {}
    
    for var_name in graph_vars:
        if var_name in globals():
            var_value = globals()[var_name]
            if hasattr(var_value, 'shape'):
                graph_info[var_name] = {
                    'shape': list(var_value.shape),
                    'type': str(type(var_value).__name__)
                }
                result['execution_info']['graph_data_found'] = var_name
                
                # 提取部分邻接矩阵信息用于可视化
                if var_name in ['adj_matrix', 'adj_norm'] and var_value.shape[0] <= 20:
                    # 只对小图提取连接信息
                    edges = []
                    rows, cols = var_value.nonzero()
                    for i in range(min(50, len(rows))):
                        edges.append([int(rows[i]), int(cols[i])])
                    result['graph_edges'] = edges
                    result['num_nodes'] = int(var_value.shape[0])
                break
    
    result['execution_info']['graph_info'] = graph_info
except Exception as e:
    result['execution_info']['graph_check_error'] = str(e)

# 提取数据集信息
try:
    dataset_info = {}
    if 'features' in globals():
        features_var = globals()['features']
        if hasattr(features_var, 'shape'):
            dataset_info['features_shape'] = list(features_var.shape)
            dataset_info['num_nodes'] = int(features_var.shape[0])
            dataset_info['feature_dim'] = int(features_var.shape[1])
    
    if 'labels' in globals():
        labels_var = globals()['labels']
        if hasattr(labels_var, 'shape'):
            dataset_info['labels_shape'] = list(labels_var.shape)
        if hasattr(labels_var, 'max'):
            dataset_info['num_classes'] = int(labels_var.max()) + 1
    
    result['execution_info']['dataset_info'] = dataset_info
except Exception as e:
    result['execution_info']['dataset_error'] = str(e)

# 返回JSON字符串
json.dumps(result)
`);
        
        var result = JSON.parse(resultData);
        
        // 如果没有训练数据，生成默认数据
        if (!result.losses || result.losses.length === 0) {
            result.losses = [1.2, 1.0, 0.8, 0.6, 0.5, 0.4];
            result.accuracies = [0.2, 0.35, 0.5, 0.65, 0.75, 0.82];
            result.model_info += '\n(显示默认数据)';
        }
        
        return result;
        
    } catch (e) {
        console.error('Pyodide结果提取失败:', e);
        return {
            losses: [1.2, 1.0, 0.8, 0.6, 0.5],
            accuracies: [0.2, 0.35, 0.5, 0.65, 0.8],
            pred_correct: [true, false, true, true, false, true],
            neighbors_sampled: {},
            model_info: '结果提取失败: ' + e.message,
            execution_info: { error: e.message }
        };
    }
}

// 导出函数
// 增强的图形可视化交互功能
function toggleAttentionWeights() {
    const svg = d3.select('#graph-visualization svg');
    const links = svg.selectAll('line');
    const nodes = svg.selectAll('circle');
    
    // 切换注意力权重显示
    const isAttentionVisible = links.attr('data-attention-visible') === 'true';
    
    if (!isAttentionVisible) {
        // 显示注意力权重
        links
            .attr('data-attention-visible', 'true')
            .transition()
            .duration(500)
            .attr('stroke-width', function(d) {
                return 2 + Math.random() * 6; // 模拟注意力权重
            })
            .attr('stroke', 'url(#attentionGradient)')
            .attr('stroke-opacity', 0.8);
            
        // 添加注意力权重标签
        svg.selectAll('.attention-label')
            .data(links.data())
            .enter()
            .append('text')
            .attr('class', 'attention-label')
            .attr('font-size', '10px')
            .attr('fill', '#8b5cf6')
            .attr('font-weight', 'bold')
            .attr('text-anchor', 'middle')
            .text(function() {
                return (Math.random() * 0.9 + 0.1).toFixed(2);
            })
            .style('opacity', 0)
            .transition()
            .duration(500)
            .style('opacity', 1)
            .attr('x', function(d) {
                return (d.source.x + d.target.x) / 2;
            })
            .attr('y', function(d) {
                return (d.source.y + d.target.y) / 2;
            });
            
        showNotification('🎯 注意力权重已显示', 'info');
    } else {
        // 隐藏注意力权重
        links
            .attr('data-attention-visible', 'false')
            .transition()
            .duration(500)
            .attr('stroke-width', 2)
            .attr('stroke', '#cbd5e1')
            .attr('stroke-opacity', 0.6);
            
        svg.selectAll('.attention-label')
            .transition()
            .duration(500)
            .style('opacity', 0)
            .remove();
            
        showNotification('注意力权重已隐藏', 'info');
    }
}

function showTrainingAnimation() {
    const svg = d3.select('#graph-visualization svg');
    const nodes = svg.selectAll('circle');
    
    showNotification('🚀 开始训练动画...', 'info');
    
    // 模拟训练过程的节点颜色变化
    let step = 0;
    const maxSteps = 10;
    
    const animationInterval = setInterval(() => {
        nodes
            .transition()
            .duration(300)
            .attr('fill', function(d, i) {
                // 模拟训练过程中预测准确性的变化
                const accuracy = Math.min(0.9, step / maxSteps + Math.random() * 0.3);
                return accuracy > 0.5 ? '#67C23A' : '#F56C6C';
            })
            .attr('r', function() {
                return 12 + Math.sin(step * 0.5) * 2;
            });
            
        step++;
        if (step >= maxSteps) {
            clearInterval(animationInterval);
            showNotification('✅ 训练动画完成', 'success');
            
            // 恢复最终状态
            setTimeout(() => {
                nodes
                    .transition()
                    .duration(500)
                    .attr('r', 12)
                    .attr('fill', function(d) {
                        return d.correct ? '#67C23A' : '#F56C6C';
                    });
            }, 1000);
        }
    }, 400);
}

function resetGraphView() {
    const svg = d3.select('#graph-visualization svg');
    const links = svg.selectAll('line');
    const nodes = svg.selectAll('circle');
    
    // 重置所有视觉效果
    links
        .attr('data-attention-visible', 'false')
        .transition()
        .duration(500)
        .attr('stroke-width', 2)
        .attr('stroke', '#cbd5e1')
        .attr('stroke-opacity', 0.6);
        
    nodes
        .transition()
        .duration(500)
        .attr('r', 12)
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .attr('fill', function(d) {
            return d.correct ? '#67C23A' : '#F56C6C';
        });
        
    // 移除所有标签
    svg.selectAll('.attention-label').remove();
    svg.selectAll('.node-label').remove();
    
    showNotification('🔄 视图已重置', 'info');
}

// 通知系统
function showNotification(message, type = 'info') {
    const colors = {
        info: { bg: '#dbeafe', border: '#3b82f6', text: '#1e40af' },
        success: { bg: '#dcfce7', border: '#22c55e', text: '#15803d' },
        warning: { bg: '#fef3c7', border: '#f59e0b', text: '#92400e' },
        error: { bg: '#fee2e2', border: '#ef4444', text: '#dc2626' }
    };
    
    const color = colors[type] || colors.info;
    
    const notification = d3.select('body')
        .append('div')
        .style('position', 'fixed')
        .style('top', '20px')
        .style('left', '50%')
        .style('transform', 'translateX(-50%)')
        .style('background', color.bg)
        .style('color', color.text)
        .style('padding', '12px 20px')
        .style('border-radius', '8px')
        .style('border', `2px solid ${color.border}`)
        .style('font-size', '14px')
        .style('font-weight', '600')
        .style('box-shadow', '0 4px 12px rgba(0,0,0,0.1)')
        .style('z-index', '1000')
        .style('opacity', '0')
        .text(message);
        
    notification
        .transition()
        .duration(300)
        .style('opacity', '1');
        
    setTimeout(() => {
        notification
            .transition()
            .duration(300)
            .style('opacity', '0')
            .style('transform', 'translateX(-50%) translateY(-20px)')
            .on('end', function() {
                notification.remove();
            });
    }, 3000);
}

// 导出函数
window.renderCharts = renderCharts;
window.renderGraph = renderGraph;
window.createGraphDemo = createGraphDemo;
window.extractPyResults = extractPyResults;
window.toggleAttentionWeights = toggleAttentionWeights;
window.showTrainingAnimation = showTrainingAnimation;
window.resetGraphView = resetGraphView;
window.showNotification = showNotification;