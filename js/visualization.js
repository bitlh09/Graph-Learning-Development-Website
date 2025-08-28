// 可视化模块
const Visualization = {
    init() {
        this.initGraphDemo();
        this.initAlgorithmFlow();
        this.initAttentionVisualization();
        this.initNeighborSampling();
    },
    
    // 初始化图演示
    initGraphDemo() {
        const container = document.getElementById('graph-demo');
        if (!container || !window.d3) return;
        
        const width = container.clientWidth || 400;
        const height = 250;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        // 创建示例图数据
        const nodes = [
            { id: 0, name: '小明', type: 'student', x: 100, y: 100 },
            { id: 1, name: '小红', type: 'designer', x: 200, y: 80 },
            { id: 2, name: '小刚', type: 'engineer', x: 300, y: 120 },
            { id: 3, name: '小李', type: 'student', x: 150, y: 200 },
            { id: 4, name: '小王', type: 'designer', x: 250, y: 180 }
        ];
        
        const links = [
            { source: 0, target: 1, type: 'friend' },
            { source: 1, target: 2, type: 'colleague' },
            { source: 0, target: 3, type: 'classmate' },
            { source: 3, target: 4, type: 'friend' },
            { source: 2, target: 4, type: 'colleague' }
        ];
        
        // 颜色映射
        const colorMap = {
            student: '#3b82f6',
            designer: '#8b5cf6',
            engineer: '#10b981'
        };
        
        // 创建力导向布局
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2));
        
        // 绘制连接线
        const link = svg.append('g')
            .selectAll('line')
            .data(links)
            .enter()
            .append('line')
            .attr('stroke', '#cbd5e1')
            .attr('stroke-width', 2)
            .attr('marker-end', 'url(#arrowhead)');
        
        // 添加箭头标记
        svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 15)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#cbd5e1');
        
        // 绘制节点
        const node = svg.append('g')
            .selectAll('circle')
            .data(nodes)
            .enter()
            .append('circle')
            .attr('r', 20)
            .attr('fill', d => colorMap[d.type])
            .attr('stroke', '#ffffff')
            .attr('stroke-width', 2)
            .attr('class', 'graph-node')
            .style('cursor', 'pointer')
            .on('mouseover', function(event, d) {
                d3.select(this).attr('r', 25);
                showTooltip(event, d);
            })
            .on('mouseout', function(event, d) {
                d3.select(this).attr('r', 20);
                hideTooltip();
            })
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
        
        // 添加节点标签
        const label = svg.append('g')
            .selectAll('text')
            .data(nodes)
            .enter()
            .append('text')
            .text(d => d.name)
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em')
            .attr('font-size', '12px')
            .attr('fill', 'white')
            .attr('font-weight', 'bold');
        
        // 工具提示
        function showTooltip(event, d) {
            const tooltip = d3.select('body').append('div')
                .attr('class', 'tooltip')
                .style('position', 'absolute')
                .style('background', 'rgba(0,0,0,0.8)')
                .style('color', 'white')
                .style('padding', '8px')
                .style('border-radius', '4px')
                .style('font-size', '12px')
                .style('pointer-events', 'none')
                .style('z-index', '1000');
            
            tooltip.html(`
                <strong>${d.name}</strong><br>
                类型: ${d.type}<br>
                节点ID: ${d.id}
            `)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px');
        }
        
        function hideTooltip() {
            d3.selectAll('.tooltip').remove();
        }
        
        // 拖拽功能
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
        
        // 更新位置
        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
            
            label
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
    },
    
    // 初始化算法流程图
    initAlgorithmFlow() {
        const container = document.getElementById('algorithm-flow');
        if (!container) return;
        
        const steps = [
            { id: 1, title: '输入图数据', description: '节点特征 + 邻接矩阵', color: '#3b82f6' },
            { id: 2, title: '消息传递', description: '聚合邻居信息', color: '#8b5cf6' },
            { id: 3, title: '特征更新', description: '更新节点表示', color: '#10b981' },
            { id: 4, title: '输出预测', description: '分类或回归结果', color: '#f59e0b' }
        ];
        
        container.innerHTML = `
            <div class="flex flex-col space-y-4">
                ${steps.map((step, index) => `
                    <div class="flex items-center">
                        <div class="flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center text-white font-bold" 
                             style="background-color: ${step.color}">
                            ${step.id}
                        </div>
                        <div class="ml-4 flex-1">
                            <h4 class="font-semibold text-gray-900">${step.title}</h4>
                            <p class="text-sm text-gray-600">${step.description}</p>
                        </div>
                        ${index < steps.length - 1 ? `
                            <div class="flex-shrink-0 w-8 h-8 flex items-center justify-center">
                                <svg class="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path>
                                </svg>
                            </div>
                        ` : ''}
                    </div>
                `).join('')}
            </div>
        `;
    },
    
    // 初始化注意力可视化
    initAttentionVisualization() {
        const container = document.getElementById('attention-heatmap');
        if (!container || !window.Plotly) return;
        
        // 示例注意力权重矩阵
        const attentionWeights = [
            [0.3, 0.4, 0.2, 0.1],
            [0.2, 0.5, 0.2, 0.1],
            [0.1, 0.3, 0.4, 0.2],
            [0.1, 0.2, 0.3, 0.4]
        ];
        
        const data = [{
            z: attentionWeights,
            type: 'heatmap',
            colorscale: 'Viridis',
            showscale: true,
            colorbar: {
                title: '注意力权重',
                titleside: 'right'
            }
        }];
        
        const layout = {
            title: 'GAT 注意力权重热力图',
            xaxis: {
                title: '目标节点',
                tickvals: [0, 1, 2, 3],
                ticktext: ['节点0', '节点1', '节点2', '节点3']
            },
            yaxis: {
                title: '源节点',
                tickvals: [0, 1, 2, 3],
                ticktext: ['节点0', '节点1', '节点2', '节点3']
            },
            width: container.clientWidth || 400,
            height: 300,
            margin: { t: 50, b: 50, l: 50, r: 50 }
        };
        
        Plotly.newPlot(container, data, layout);
    },
    
    // 初始化邻居采样可视化
    initNeighborSampling() {
        const container = document.getElementById('neighbor-sampling');
        if (!container || !window.d3) return;
        
        const width = container.clientWidth || 400;
        const height = 300;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        // 创建分层图结构
        const layers = [
            { level: 0, nodes: [{ id: 'center', name: '中心节点', sampled: true }] },
            { level: 1, nodes: [
                { id: 'n1', name: '邻居1', sampled: true },
                { id: 'n2', name: '邻居2', sampled: true },
                { id: 'n3', name: '邻居3', sampled: false },
                { id: 'n4', name: '邻居4', sampled: false }
            ]},
            { level: 2, nodes: [
                { id: 'nn1', name: '邻居的邻居1', sampled: true },
                { id: 'nn2', name: '邻居的邻居2', sampled: false },
                { id: 'nn3', name: '邻居的邻居3', sampled: false }
            ]}
        ];
        
        // 计算节点位置
        const nodeRadius = 25;
        const layerSpacing = 80;
        
        layers.forEach((layer, layerIndex) => {
            const y = 50 + layerIndex * layerSpacing;
            const totalWidth = (layer.nodes.length - 1) * 60;
            const startX = (width - totalWidth) / 2;
            
            layer.nodes.forEach((node, nodeIndex) => {
                node.x = startX + nodeIndex * 60;
                node.y = y;
            });
        });
        
        // 绘制连接线
        const links = [
            { source: 'center', target: 'n1' },
            { source: 'center', target: 'n2' },
            { source: 'center', target: 'n3' },
            { source: 'center', target: 'n4' },
            { source: 'n1', target: 'nn1' },
            { source: 'n2', target: 'nn2' },
            { source: 'n3', target: 'nn3' }
        ];
        
        svg.append('g')
            .selectAll('line')
            .data(links)
            .enter()
            .append('line')
            .attr('x1', d => {
                const source = layers.flatMap(l => l.nodes).find(n => n.id === d.source);
                return source ? source.x : 0;
            })
            .attr('y1', d => {
                const source = layers.flatMap(l => l.nodes).find(n => n.id === d.source);
                return source ? source.y : 0;
            })
            .attr('x2', d => {
                const target = layers.flatMap(l => l.nodes).find(n => n.id === d.target);
                return target ? target.x : 0;
            })
            .attr('y2', d => {
                const target = layers.flatMap(l => l.nodes).find(n => n.id === d.target);
                return target ? target.y : 0;
            })
            .attr('stroke', '#cbd5e1')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', d => {
                const target = layers.flatMap(l => l.nodes).find(n => n.id === d.target);
                return target && !target.sampled ? '5,5' : 'none';
            });
        
        // 绘制节点
        const allNodes = layers.flatMap(l => l.nodes);
        
        svg.append('g')
            .selectAll('circle')
            .data(allNodes)
            .enter()
            .append('circle')
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
            .attr('r', nodeRadius)
            .attr('fill', d => d.sampled ? '#10b981' : '#e5e7eb')
            .attr('stroke', d => d.sampled ? '#059669' : '#d1d5db')
            .attr('stroke-width', 2)
            .style('cursor', 'pointer')
            .on('mouseover', function(event, d) {
                d3.select(this).attr('r', nodeRadius + 5);
                showSamplingTooltip(event, d);
            })
            .on('mouseout', function(event, d) {
                d3.select(this).attr('r', nodeRadius);
                hideSamplingTooltip();
            });
        
        // 添加节点标签
        svg.append('g')
            .selectAll('text')
            .data(allNodes)
            .enter()
            .append('text')
            .text(d => d.name)
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em')
            .attr('font-size', '10px')
            .attr('fill', d => d.sampled ? 'white' : '#6b7280')
            .attr('font-weight', 'bold');
        
        // 添加采样说明
        svg.append('text')
            .attr('x', 10)
            .attr('y', height - 10)
            .attr('font-size', '12px')
            .attr('fill', '#6b7280')
            .text('绿色: 已采样节点 | 灰色: 未采样节点');
        
        function showSamplingTooltip(event, d) {
            const tooltip = d3.select('body').append('div')
                .attr('class', 'sampling-tooltip')
                .style('position', 'absolute')
                .style('background', 'rgba(0,0,0,0.8)')
                .style('color', 'white')
                .style('padding', '8px')
                .style('border-radius', '4px')
                .style('font-size', '12px')
                .style('pointer-events', 'none')
                .style('z-index', '1000');
            
            tooltip.html(`
                <strong>${d.name}</strong><br>
                采样状态: ${d.sampled ? '已采样' : '未采样'}<br>
                层级: ${layers.find(l => l.nodes.includes(d)).level}
            `)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px');
        }
        
        function hideSamplingTooltip() {
            d3.selectAll('.sampling-tooltip').remove();
        }
    },
    
    // 渲染训练曲线
    renderTrainingCurves(losses, accuracies, containerId) {
        if (!window.Plotly) return;
        
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const epochs = Array.from({ length: losses.length }, (_, i) => i + 1);
        
        const data = [
            {
                x: epochs,
                y: losses,
                type: 'scatter',
                mode: 'lines+markers',
                name: '训练损失',
                line: { color: '#ef4444', width: 2 },
                marker: { size: 4 }
            },
            {
                x: epochs,
                y: accuracies,
                type: 'scatter',
                mode: 'lines+markers',
                name: '训练准确率',
                line: { color: '#10b981', width: 2 },
                marker: { size: 4 },
                yaxis: 'y2'
            }
        ];
        
        const layout = {
            title: '训练过程可视化',
            xaxis: {
                title: '训练轮数 (Epoch)',
                gridcolor: '#e5e7eb'
            },
            yaxis: {
                title: '损失 (Loss)',
                gridcolor: '#e5e7eb',
                side: 'left'
            },
            yaxis2: {
                title: '准确率 (Accuracy)',
                overlaying: 'y',
                side: 'right',
                range: [0, 1]
            },
            legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.8)'
            },
            margin: { t: 50, b: 50, l: 50, r: 50 },
            width: container.clientWidth || 400,
            height: 300,
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)'
        };
        
        Plotly.newPlot(container, data, layout);
    },
    
    // 渲染注意力热力图
    renderAttentionHeatmap(attentionWeights, containerId) {
        if (!window.Plotly) return;
        
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const data = [{
            z: attentionWeights,
            type: 'heatmap',
            colorscale: [
                [0, '#f0f9ff'],
                [0.25, '#bae6fd'],
                [0.5, '#7dd3fc'],
                [0.75, '#38bdf8'],
                [1, '#0ea5e9']
            ],
            showscale: true,
            colorbar: {
                title: '注意力权重',
                titleside: 'right'
            }
        }];
        
        const layout = {
            title: 'GAT 注意力权重分布',
            xaxis: {
                title: '目标节点',
                tickmode: 'array',
                tickvals: Array.from({ length: attentionWeights[0].length }, (_, i) => i),
                ticktext: Array.from({ length: attentionWeights[0].length }, (_, i) => `节点${i}`)
            },
            yaxis: {
                title: '源节点',
                tickmode: 'array',
                tickvals: Array.from({ length: attentionWeights.length }, (_, i) => i),
                ticktext: Array.from({ length: attentionWeights.length }, (_, i) => `节点${i}`)
            },
            width: container.clientWidth || 400,
            height: 300,
            margin: { t: 50, b: 50, l: 50, r: 50 }
        };
        
        Plotly.newPlot(container, data, layout);
    },
    
    // 渲染节点嵌入可视化
    renderNodeEmbeddings(embeddings, labels, containerId) {
        if (!window.Plotly) return;
        
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // 使用PCA降维到2D
        const pcaResult = this.pca(embeddings, 2);
        
        const uniqueLabels = [...new Set(labels)];
        const colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444'];
        
        const data = uniqueLabels.map((label, index) => ({
            x: pcaResult.filter((_, i) => labels[i] === label).map(point => point[0]),
            y: pcaResult.filter((_, i) => labels[i] === label).map(point => point[1]),
            mode: 'markers',
            type: 'scatter',
            name: `类别 ${label}`,
            marker: {
                size: 8,
                color: colors[index % colors.length],
                opacity: 0.7
            }
        }));
        
        const layout = {
            title: '节点嵌入可视化 (PCA降维)',
            xaxis: { title: '主成分1' },
            yaxis: { title: '主成分2' },
            width: container.clientWidth || 400,
            height: 300,
            margin: { t: 50, b: 50, l: 50, r: 50 },
            legend: {
                x: 0.02,
                y: 0.98
            }
        };
        
        Plotly.newPlot(container, data, layout);
    },
    
    // PCA降维算法
    pca(data, dimensions) {
        // 简化的PCA实现
        const n = data.length;
        const m = data[0].length;
        
        // 计算均值
        const mean = new Array(m).fill(0);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < m; j++) {
                mean[j] += data[i][j];
            }
        }
        for (let j = 0; j < m; j++) {
            mean[j] /= n;
        }
        
        // 中心化数据
        const centered = data.map(row => 
            row.map((val, j) => val - mean[j])
        );
        
        // 计算协方差矩阵
        const cov = new Array(m).fill(0).map(() => new Array(m).fill(0));
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < m; j++) {
                for (let k = 0; k < n; k++) {
                    cov[i][j] += centered[k][i] * centered[k][j];
                }
                cov[i][j] /= (n - 1);
            }
        }
        
        // 简化的特征值分解（这里使用随机投影作为近似）
        const result = [];
        for (let i = 0; i < n; i++) {
            const point = [];
            for (let d = 0; d < dimensions; d++) {
                let sum = 0;
                for (let j = 0; j < m; j++) {
                    sum += centered[i][j] * (Math.random() - 0.5);
                }
                point.push(sum);
            }
            result.push(point);
        }
        
        return result;
    },
    
    // 创建动画效果
    createAnimation(element, animationType = 'fadeIn') {
        const animations = {
            fadeIn: {
                opacity: [0, 1],
                transform: ['translateY(20px)', 'translateY(0)']
            },
            slideIn: {
                transform: ['translateX(-100%)', 'translateX(0)']
            },
            scaleIn: {
                transform: ['scale(0.8)', 'scale(1)'],
                opacity: [0, 1]
            }
        };
        
        const animation = animations[animationType];
        if (!animation) return;
        
        element.style.transition = 'all 0.5s ease-out';
        Object.keys(animation).forEach(property => {
            element.style[property] = animation[property][0];
        });
        
        setTimeout(() => {
            Object.keys(animation).forEach(property => {
                element.style[property] = animation[property][1];
            });
        }, 100);
    },
    
    // 创建进度条动画
    createProgressAnimation(progressBar, targetProgress, duration = 1000) {
        const startProgress = 0;
        const startTime = Date.now();
        
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const currentProgress = startProgress + (targetProgress - startProgress) * progress;
            progressBar.style.setProperty('--progress', currentProgress + '%');
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    },
    
    // 创建粒子效果
    createParticleEffect(container, x, y, color = '#3b82f6') {
        const particle = document.createElement('div');
        particle.style.position = 'absolute';
        particle.style.left = x + 'px';
        particle.style.top = y + 'px';
        particle.style.width = '4px';
        particle.style.height = '4px';
        particle.style.backgroundColor = color;
        particle.style.borderRadius = '50%';
        particle.style.pointerEvents = 'none';
        particle.style.zIndex = '1000';
        
        container.appendChild(particle);
        
        // 动画
        const animation = particle.animate([
            { 
                transform: 'translate(0, 0) scale(1)',
                opacity: 1
            },
            { 
                transform: 'translate(0, -50px) scale(0)',
                opacity: 0
            }
        ], {
            duration: 1000,
            easing: 'ease-out'
        });
        
        animation.onfinish = () => {
            if (particle.parentNode) {
                particle.parentNode.removeChild(particle);
            }
        };
    },
    
    // 代码执行可视化
    renderCodeExecution(code, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const lines = code.split('\n');
        let currentLine = 0;
        
        container.innerHTML = `
            <div class="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-auto max-h-64">
                <div id="code-output"></div>
            </div>
        `;
        
        const output = document.getElementById('code-output');
        
        const executeLine = () => {
            if (currentLine < lines.length) {
                const line = lines[currentLine];
                const lineElement = document.createElement('div');
                lineElement.className = 'code-line opacity-0';
                lineElement.innerHTML = `<span class="text-gray-500">${(currentLine + 1).toString().padStart(3, ' ')}</span> ${line}`;
                
                output.appendChild(lineElement);
                
                // 高亮当前行
                setTimeout(() => {
                    lineElement.classList.remove('opacity-0');
                    lineElement.classList.add('bg-green-900', 'text-green-300');
                }, 100);
                
                // 移除高亮
                setTimeout(() => {
                    lineElement.classList.remove('bg-green-900', 'text-green-300');
                }, 500);
                
                currentLine++;
                setTimeout(executeLine, 200);
            }
        };
        
        executeLine();
    },
    
    // 算法步骤可视化
    renderAlgorithmSteps(steps, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = `
            <div class="space-y-4">
                ${steps.map((step, index) => `
                    <div class="algorithm-step opacity-0 transform translate-y-4" data-step="${index}">
                        <div class="flex items-start space-x-4">
                            <div class="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
                                ${index + 1}
                            </div>
                            <div class="flex-1">
                                <h4 class="font-semibold text-gray-900">${step.title}</h4>
                                <p class="text-gray-600">${step.description}</p>
                                ${step.code ? `
                                    <pre class="mt-2 bg-gray-100 p-3 rounded text-sm overflow-x-auto">
                                        <code>${step.code}</code>
                                    </pre>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        
        // 逐步显示动画
        const stepElements = container.querySelectorAll('.algorithm-step');
        stepElements.forEach((element, index) => {
            setTimeout(() => {
                element.classList.remove('opacity-0', 'translate-y-4');
            }, index * 300);
        });
    }
};

// 全局函数导出
window.renderTrainingCurves = (losses, accuracies, containerId) => 
    Visualization.renderTrainingCurves(losses, accuracies, containerId);

window.renderAttentionHeatmap = (attentionWeights, containerId) => 
    Visualization.renderAttentionHeatmap(attentionWeights, containerId);

window.renderNodeEmbeddings = (embeddings, labels, containerId) => 
    Visualization.renderNodeEmbeddings(embeddings, labels, containerId);

window.renderCodeExecution = (code, containerId) => 
    Visualization.renderCodeExecution(code, containerId);

window.renderAlgorithmSteps = (steps, containerId) => 
    Visualization.renderAlgorithmSteps(steps, containerId);

// 页面加载完成后初始化
// Initialization moved to `GraphLearn.init()` in `js/core.js` to ensure a single, predictable startup sequence.
// Visualization.init() will be called from there.
