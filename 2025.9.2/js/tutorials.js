// 教程交互功能模块

// 教程页：最小代码运行按钮（展示示例输出）
function runGCNCode() {
    var out = document.getElementById('gcn-output');
    if (out) out.classList.remove('hidden');
}

function runGATCode() {
    var out = document.getElementById('gat-output');
    if (out) out.classList.remove('hidden');
}

function runGraphSAGECode() {
    var out = document.getElementById('graphsage-output');
    if (out) out.classList.remove('hidden');
}

// 微博问答检查
function checkWeiboAnswer() {
    var selected = document.querySelector('input[name="weibo-graph"]:checked');
    if (selected && selected.id === 'weibo-directed') {
        alert('正确！微博的关注关系是有向图，因为关注是单向的（A关注B，但B不一定关注A）。');
    } else if (selected) {
        alert('不完全正确。微博的关注关系主要是有向图，因为关注通常是单向的。');
    } else {
        alert('请先选择一个答案！');
    }
}

// 切换执行模式
function switchExecMode(mode) {
    practiceState.execMode = mode;
    var consoleEl = document.getElementById('console-output');
    if (mode === 'pyodide' && !practiceState.pyodideReady) {
        if (consoleEl) consoleEl.textContent = 'Pyodide 尚未就绪，正在加载...';
        initPyodideOnce();
    } else {
        if (consoleEl) consoleEl.textContent = '已切换执行模式：' + (mode === 'pyodide' ? 'Pyodide' : '前端模拟');
    }
}

// 创建交互式图演示（首页用）
function initHomeGraphDemo() {
    // 在页面加载完成后初始化首页的图演示
    setTimeout(function() {
        createGraphDemo('graph-demo');
    }, 500);
}

// 创建社交网络图演示
function initSocialGraphDemo() {
    setTimeout(function() {
        createSocialNetworkDemo('social-graph-demo');
    }, 500);
}

function createSocialNetworkDemo(containerId) {
    var container = document.getElementById(containerId);
    if (!container || !window.d3) return;
    
    container.innerHTML = '';
    var width = container.clientWidth || 400;
    var height = 300;
    
    // 创建控制面板
    var controls = d3.select(container)
        .append('div')
        .attr('class', 'social-controls mb-3 flex flex-wrap gap-2')
        .style('margin-bottom', '10px');
    
    controls.append('button')
        .attr('class', 'btn btn-sm bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600')
        .style('background-color', '#3b82f6')
        .style('color', 'white')
        .style('padding', '5px 10px')
        .style('border', 'none')
        .style('border-radius', '4px')
        .style('cursor', 'pointer')
        .style('margin-right', '5px')
        .text('检测社区')
        .on('click', detectCommunities);
    
    controls.append('button')
        .attr('class', 'btn btn-sm')
        .style('background-color', '#10b981')
        .style('color', 'white')
        .style('padding', '5px 10px')
        .style('border', 'none')
        .style('border-radius', '4px')
        .style('cursor', 'pointer')
        .style('margin-right', '5px')
        .text('重新布局')
        .on('click', restartSimulation);
    
    controls.append('button')
        .attr('class', 'btn btn-sm')
        .style('background-color', '#f59e0b')
        .style('color', 'white')
        .style('padding', '5px 10px')
        .style('border', 'none')
        .style('border-radius', '4px')
        .style('cursor', 'pointer')
        .style('margin-right', '5px')
        .text('显示路径')
        .on('click', showShortestPaths);
    
    controls.append('input')
        .attr('type', 'text')
        .attr('placeholder', '搜索用户...')
        .style('padding', '5px')
        .style('border', '1px solid #ccc')
        .style('border-radius', '4px')
        .style('margin-right', '5px')
        .on('input', searchUsers);
    
    var svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
    
    // 创建定义区域
    var defs = svg.append('defs');
    
    // 定义箭头标记
    defs.append('marker')
        .attr('id', 'social-arrow')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 20)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 8)
        .attr('markerHeight', 8)
        .append('path')
        .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
        .attr('fill', '#999');
    
    // 定义高亮箭头
    defs.append('marker')
        .attr('id', 'social-arrow-highlight')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 20)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 8)
        .attr('markerHeight', 8)
        .append('path')
        .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
        .attr('fill', '#ff6b6b');
    
    // 定义发光效果
    var filter = defs.append('filter').attr('id', 'social-glow');
    filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'coloredBlur');
    var feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');
    
    // 扩展节点数据
    var nodes = [
        {id: 0, name: '小明', age: 20, profession: '学生', interests: ['编程', '游戏'], city: '北京', followers: 520, following: 180, community: 0},
        {id: 1, name: '小红', age: 25, profession: '设计师', interests: ['设计', '摄影'], city: '上海', followers: 380, following: 240, community: 0},
        {id: 2, name: '小刚', age: 28, profession: '工程师', interests: ['技术', '阅读'], city: '深圳', followers: 620, following: 150, community: 1},
        {id: 3, name: '小丽', age: 30, profession: '教师', interests: ['教育', '旅行'], city: '杭州', followers: 450, following: 320, community: 1},
        {id: 4, name: '小强', age: 26, profession: '医生', interests: ['医学', '运动'], city: '广州', followers: 680, following: 200, community: 1},
        {id: 5, name: '小华', age: 24, profession: '学生', interests: ['音乐', '电影'], city: '北京', followers: 290, following: 410, community: 0},
        {id: 6, name: '小芳', age: 27, profession: '记者', interests: ['新闻', '写作'], city: '上海', followers: 730, following: 180, community: 2}
    ];
    
    var links = [
        {source: 0, target: 1, relationship: '同学', strength: 0.8, type: 'friendship'},
        {source: 1, target: 2, relationship: '同事', strength: 0.6, type: 'work'},
        {source: 2, target: 3, relationship: '朋友', strength: 0.9, type: 'friendship'},
        {source: 3, target: 4, relationship: '邻居', strength: 0.4, type: 'neighbor'},
        {source: 0, target: 4, relationship: '朋友', strength: 0.7, type: 'friendship'},
        {source: 0, target: 5, relationship: '室友', strength: 0.95, type: 'family'},
        {source: 1, target: 6, relationship: '合作伙伴', strength: 0.75, type: 'work'},
        {source: 5, target: 6, relationship: '朋友', strength: 0.6, type: 'friendship'}
    ];
    
    var colorScale = d3.scaleOrdinal()
        .domain(['学生', '设计师', '工程师', '教师', '医生', '记者'])
        .range(['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4']);
    
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
                   d.type === 'work' ? '#3b82f6' : 
                   d.type === 'family' ? '#ef4444' : '#f59e0b';
        })
        .attr('stroke-width', function(d) { return Math.sqrt(d.strength * 6); })
        .attr('stroke-opacity', 0.6)
        .attr('marker-end', 'url(#social-arrow)')
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
            d3.select(this)
                .attr('stroke-width', Math.sqrt(d.strength * 8))
                .attr('stroke-opacity', 1)
                .attr('marker-end', 'url(#social-arrow-highlight)');
            
            showTooltip(event, `关系: ${d.relationship}<br/>强度: ${d.strength}<br/>类型: ${d.type}`);
        })
        .on('mouseout', function(event, d) {
            d3.select(this)
                .attr('stroke-width', Math.sqrt(d.strength * 6))
                .attr('stroke-opacity', 0.6)
                .attr('marker-end', 'url(#social-arrow)');
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
        .attr('r', function(d) { return 12 + Math.log(d.followers) * 2; })
        .attr('fill', function(d) { return colorScale(d.profession); })
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
            d3.select(this)
                .attr('filter', 'url(#social-glow)')
                .transition()
                .duration(150)
                .attr('r', function(d) { return 15 + Math.log(d.followers) * 2; });
            
            // 高亮相关连接
            link.style('opacity', function(l) {
                return l.source.id === d.id || l.target.id === d.id ? 1 : 0.1;
            });
            
            var tooltip = `<strong>${d.name}</strong><br/>
                          年龄: ${d.age}<br/>
                          职业: ${d.profession}<br/>
                          城市: ${d.city}<br/>
                          关注: ${d.following} | 粉丝: ${d.followers}<br/>
                          兴趣: ${d.interests.join(', ')}`;
            showTooltip(event, tooltip);
        })
        .on('mouseout', function(event, d) {
            d3.select(this)
                .attr('filter', null)
                .transition()
                .duration(150)
                .attr('r', function(d) { return 12 + Math.log(d.followers) * 2; });
            
            link.style('opacity', 1);
            hideTooltip();
        })
        .on('click', function(event, d) {
            showUserDetailModal(d);
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

    simulation.on('tick', function() {
        link.attr('x1', function(d) { return d.source.x; })
            .attr('y1', function(d) { return d.source.y; })
            .attr('x2', function(d) { return d.target.x; })
            .attr('y2', function(d) { return d.target.y; });
        node.attr('cx', function(d) { return d.x; }).attr('cy', function(d) { return d.y; });
        labels.attr('x', function(d) { return d.x; }).attr('y', function(d) { return d.y; });
    });
    
    // 工具函数
    function detectCommunities() {
        // 简单的社区检测算法（基于连接密度）
        var communityColors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'];
        
        node.transition()
            .duration(1000)
            .attr('fill', function(d) {
                return communityColors[d.community % communityColors.length];
            });
        
        // 显示社区信息
        var communities = {};
        nodes.forEach(function(n) {
            if (!communities[n.community]) communities[n.community] = [];
            communities[n.community].push(n.name);
        });
        
        var communityInfo = '检测到的社区:\n';
        Object.keys(communities).forEach(function(key) {
            communityInfo += `社区 ${parseInt(key) + 1}: ${communities[key].join(', ')}\n`;
        });
        
        setTimeout(function() { alert(communityInfo); }, 1000);
    }
    
    function restartSimulation() {
        simulation.alpha(1).restart();
    }
    
    function showShortestPaths() {
        // 高亮显示最短路径
        var paths = findAllShortestPaths();
        var pathInfo = '最短路径分析:\n';
        paths.forEach(function(path) {
            pathInfo += `${path.from} → ${path.to}: ${path.distance} 步\n`;
        });
        alert(pathInfo);
    }
    
    function searchUsers() {
        var searchTerm = d3.select(this).property('value').toLowerCase();
        
        node.style('opacity', function(d) {
            return searchTerm === '' || d.name.toLowerCase().includes(searchTerm) || 
                   d.profession.toLowerCase().includes(searchTerm) ? 1 : 0.3;
        });
        
        labels.style('opacity', function(d) {
            return searchTerm === '' || d.name.toLowerCase().includes(searchTerm) || 
                   d.profession.toLowerCase().includes(searchTerm) ? 1 : 0.3;
        });
    }
    
    function highlightPath(sourceId, targetId) {
        // 重置所有样式
        link.attr('stroke-opacity', 0.2).attr('stroke-width', 1);
        node.style('opacity', 0.3);
        
        // 高亮路径上的节点和边
        var path = findShortestPath(sourceId, targetId);
        if (path.length > 1) {
            for (var i = 0; i < path.length - 1; i++) {
                link.filter(function(d) {
                    return (d.source.id === path[i] && d.target.id === path[i + 1]) ||
                           (d.target.id === path[i] && d.source.id === path[i + 1]);
                }).attr('stroke-opacity', 1).attr('stroke-width', 4);
            }
            
            path.forEach(function(nodeId) {
                node.filter(function(d) { return d.id === nodeId; }).style('opacity', 1);
            });
        }
        
        // 3秒后恢复
        setTimeout(function() {
            link.attr('stroke-opacity', 0.6).attr('stroke-width', function(d) { 
                return Math.sqrt(d.strength * 6); 
            });
            node.style('opacity', 1);
        }, 3000);
    }
    
    function findShortestPath(sourceId, targetId) {
        // 简单的BFS最短路径算法
        var queue = [[sourceId]];
        var visited = new Set([sourceId]);
        
        while (queue.length > 0) {
            var path = queue.shift();
            var node = path[path.length - 1];
            
            if (node === targetId) {
                return path;
            }
            
            links.forEach(function(link) {
                var neighbor = link.source.id === node ? link.target.id : 
                              link.target.id === node ? link.source.id : null;
                
                if (neighbor !== null && !visited.has(neighbor)) {
                    visited.add(neighbor);
                    queue.push([...path, neighbor]);
                }
            });
        }
        
        return [];
    }
    
    function findAllShortestPaths() {
        var paths = [];
        for (var i = 0; i < nodes.length; i++) {
            for (var j = i + 1; j < nodes.length; j++) {
                var path = findShortestPath(nodes[i].id, nodes[j].id);
                if (path.length > 0) {
                    paths.push({
                        from: nodes[i].name,
                        to: nodes[j].name,
                        distance: path.length - 1
                    });
                }
            }
        }
        return paths;
    }
    
    function showUserDetailModal(userData) {
        // 创建模态弹窗
        var modal = d3.select('body').append('div')
            .style('position', 'fixed')
            .style('top', '0')
            .style('left', '0')
            .style('width', '100%')
            .style('height', '100%')
            .style('background-color', 'rgba(0,0,0,0.5)')
            .style('z-index', '1000')
            .style('display', 'flex')
            .style('align-items', 'center')
            .style('justify-content', 'center');
        
        var modalContent = modal.append('div')
            .style('background-color', 'white')
            .style('padding', '20px')
            .style('border-radius', '10px')
            .style('box-shadow', '0 4px 6px rgba(0, 0, 0, 0.1)')
            .style('max-width', '400px')
            .style('width', '90%');
        
        modalContent.append('h3')
            .style('margin-top', '0')
            .style('color', '#333')
            .text(userData.name + ' 的详细信息');
        
        var infoList = modalContent.append('div')
            .style('margin', '15px 0');
        
        var infoItems = [
            {label: '年龄', value: userData.age + ' 岁'},
            {label: '职业', value: userData.profession},
            {label: '所在城市', value: userData.city},
            {label: '关注数', value: userData.following},
            {label: '粉丝数', value: userData.followers},
            {label: '兴趣爱好', value: userData.interests.join(', ')},
            {label: '社区编号', value: '社区 ' + (userData.community + 1)}
        ];
        
        infoItems.forEach(function(item) {
            var itemDiv = infoList.append('div')
                .style('margin', '8px 0')
                .style('display', 'flex')
                .style('justify-content', 'space-between');
            
            itemDiv.append('span')
                .style('font-weight', 'bold')
                .style('color', '#555')
                .text(item.label + ':');
            
            itemDiv.append('span')
                .style('color', '#333')
                .text(item.value);
        });
        
        var buttonContainer = modalContent.append('div')
            .style('text-align', 'center')
            .style('margin-top', '20px');
        
        buttonContainer.append('button')
            .style('background-color', '#3b82f6')
            .style('color', 'white')
            .style('border', 'none')
            .style('padding', '10px 20px')
            .style('border-radius', '5px')
            .style('cursor', 'pointer')
            .text('关闭')
            .on('click', function() {
                modal.remove();
            });
        
        // 点击背景关闭
        modal.on('click', function(event) {
            if (event.target === modal.node()) {
                modal.remove();
            }
        });
    }
    
    // 工具提示函数
    function showTooltip(event, content) {
        var tooltip = d3.select('body').selectAll('.social-tooltip').data([0]);
        tooltip = tooltip.enter().append('div')
            .attr('class', 'social-tooltip')
            .style('position', 'absolute')
            .style('background-color', 'rgba(0,0,0,0.8)')
            .style('color', 'white')
            .style('padding', '8px')
            .style('border-radius', '4px')
            .style('font-size', '12px')
            .style('pointer-events', 'none')
            .style('z-index', '1000')
            .merge(tooltip);
        
        tooltip.html(content)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .style('opacity', 1);
    }
    
    function hideTooltip() {
        d3.select('.social-tooltip').style('opacity', 0);
    }
}

// 创建消息传递演示
function initMessagePassingDemo() {
    setTimeout(function() {
        createMessagePassingDemo('message-passing-demo');
    }, 500);
}

function createMessagePassingDemo(containerId) {
    var container = document.getElementById(containerId);
    if (!container || !window.d3) return;
    
    container.innerHTML = '';
    var width = container.clientWidth || 400;
    var height = 300;
    
    // 创建控制面板
    var controls = d3.select(container)
        .append('div')
        .attr('class', 'message-controls mb-3')
        .style('margin-bottom', '10px')
        .style('text-align', 'center');
    
    controls.append('button')
        .attr('id', 'start-message-btn')
        .style('background-color', '#10b981')
        .style('color', 'white')
        .style('border', 'none')
        .style('padding', '8px 16px')
        .style('border-radius', '4px')
        .style('cursor', 'pointer')
        .style('margin', '0 5px')
        .text('开始消息传递')
        .on('click', startMessagePassing);
    
    controls.append('button')
        .style('background-color', '#f59e0b')
        .style('color', 'white')
        .style('border', 'none')
        .style('padding', '8px 16px')
        .style('border-radius', '4px')
        .style('cursor', 'pointer')
        .style('margin', '0 5px')
        .text('暂停/继续')
        .on('click', toggleAnimation);
    
    controls.append('button')
        .style('background-color', '#ef4444')
        .style('color', 'white')
        .style('border', 'none')
        .style('padding', '8px 16px')
        .style('border-radius', '4px')
        .style('cursor', 'pointer')
        .style('margin', '0 5px')
        .text('重置')
        .on('click', resetAnimation);
    
    controls.append('label')
        .style('margin-left', '15px')
        .style('color', '#666')
        .text('速度: ');
    
    controls.append('input')
        .attr('type', 'range')
        .attr('min', '1')
        .attr('max', '5')
        .attr('value', '3')
        .style('margin-left', '5px')
        .on('input', function() {
            animationSpeed = parseInt(this.value);
        });
    
    // 步骤显示
    var stepDisplay = d3.select(container)
        .append('div')
        .attr('class', 'step-display')
        .style('text-align', 'center')
        .style('margin-bottom', '10px')
        .style('font-weight', 'bold')
        .style('color', '#333')
        .text('点击"开始消息传递"查看演示');
    
    var svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
    
    // 创建定义区域
    var defs = svg.append('defs');
    
    // 定义渐变色彩
    var gradient = defs.append('linearGradient')
        .attr('id', 'messageGradient')
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '100%').attr('y2', '0%');
    gradient.append('stop').attr('offset', '0%').attr('stop-color', '#ff6b6b');
    gradient.append('stop').attr('offset', '100%').attr('stop-color', '#4ecdc4');
    
    // 定义消息粒子效果
    var messageFilter = defs.append('filter').attr('id', 'messageGlow');
    messageFilter.append('feGaussianBlur').attr('stdDeviation', '2').attr('result', 'coloredBlur');
    var feMerge = messageFilter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');
    
    // 扩展节点数据，包含特征向量
    var nodes = [
        {id: 0, x: width/2, y: height/2, name: '中心节点', features: [1, 0, 0], messages: [], newFeatures: [1, 0, 0], active: false},
        {id: 1, x: width/2 - 80, y: height/2 - 60, name: '节点A', features: [0, 1, 0], messages: [], newFeatures: [0, 1, 0], active: false},
        {id: 2, x: width/2 + 80, y: height/2 - 60, name: '节点B', features: [0, 0, 1], messages: [], newFeatures: [0, 0, 1], active: false},
        {id: 3, x: width/2 - 80, y: height/2 + 60, name: '节点C', features: [1, 1, 0], messages: [], newFeatures: [1, 1, 0], active: false},
        {id: 4, x: width/2 + 80, y: height/2 + 60, name: '节点D', features: [0, 1, 1], messages: [], newFeatures: [0, 1, 1], active: false}
    ];
    
    var links = [
        {source: 0, target: 1, weight: 0.8}, 
        {source: 0, target: 2, weight: 0.6}, 
        {source: 0, target: 3, weight: 0.9}, 
        {source: 0, target: 4, weight: 0.7},
        {source: 1, target: 2, weight: 0.5},
        {source: 3, target: 4, weight: 0.4}
    ];
    
    // 创建邻接列表
    var adjacencyList = {};
    nodes.forEach(function(node) {
        adjacencyList[node.id] = [];
    });
    links.forEach(function(link) {
        adjacencyList[link.source].push({neighbor: link.target, weight: link.weight});
        adjacencyList[link.target].push({neighbor: link.source, weight: link.weight});
    });
    
    // 绘制连接线
    var link = svg.selectAll('line')
        .data(links)
        .enter().append('line')
        .attr('x1', function(d) { return nodes[d.source].x; })
        .attr('y1', function(d) { return nodes[d.source].y; })
        .attr('x2', function(d) { return nodes[d.target].x; })
        .attr('y2', function(d) { return nodes[d.target].y; })
        .attr('stroke', '#cbd5e1')
        .attr('stroke-width', function(d) { return d.weight * 3; })
        .attr('stroke-opacity', 0.6);
    
    // 绘制节点
    var nodeGroup = svg.selectAll('.node-group')
        .data(nodes)
        .enter().append('g')
        .attr('class', 'node-group')
        .attr('transform', function(d) { return 'translate(' + d.x + ',' + d.y + ')'; });
    
    var nodeCircle = nodeGroup.append('circle')
        .attr('r', 18)
        .attr('fill', function(d) { 
            return d.id === 0 ? '#ff6b6b' : '#3b82f6'; 
        })
        .attr('stroke', '#fff')
        .attr('stroke-width', 3)
        .style('cursor', 'pointer')
        .on('click', function(event, d) {
            if (d.id === 0) {
                startMessagePassing();
            } else {
                showNodeFeatures(d);
            }
        });
    
    // 添加节点标签
    nodeGroup.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', '.35em')
        .attr('font-size', '10px')
        .attr('fill', 'white')
        .attr('font-weight', 'bold')
        .text(function(d) { return d.id; });
    
    // 添加特征向量显示
    nodeGroup.append('text')
        .attr('class', 'feature-text')
        .attr('text-anchor', 'middle')
        .attr('dy', '30px')
        .attr('font-size', '8px')
        .attr('fill', '#333')
        .text(function(d) { return '[' + d.features.join(',') + ']'; });
    
    // 动画变量
    var animationSpeed = 3;
    var currentStep = 0;
    var isAnimating = false;
    var animationTimer = null;
    var messageParticles = [];
    
    function startMessagePassing() {
        if (isAnimating) return;
        
        resetAnimation();
        isAnimating = true;
        currentStep = 0;
        
        var steps = [
            {
                name: '步骤1: 初始化节点特征',
                action: function() {
                    stepDisplay.text('步骤1: 初始化节点特征');
                    highlightActiveNodes([]);
                }
            },
            {
                name: '步骤2: 中心节点开始发送消息',
                action: function() {
                    stepDisplay.text('步骤2: 中心节点开始发送消息');
                    highlightActiveNodes([0]);
                    sendMessagesFrom(0);
                }
            },
            {
                name: '步骤3: 邻居节点接收消息',
                action: function() {
                    stepDisplay.text('步骤3: 邻居节点接收消息');
                    highlightActiveNodes([1, 2, 3, 4]);
                }
            },
            {
                name: '步骤4: 聚合消息并更新特征',
                action: function() {
                    stepDisplay.text('步骤4: 聚合消息并更新特征');
                    aggregateAndUpdate();
                }
            },
            {
                name: '步骤5: 第二轮消息传递',
                action: function() {
                    stepDisplay.text('步骤5: 第二轮消息传递');
                    secondRoundMessagePassing();
                }
            },
            {
                name: '步骤6: 消息传递完成',
                action: function() {
                    stepDisplay.text('消息传递完成！点击节点查看更新后的特征');
                    highlightActiveNodes([]);
                    showCompletionEffect();
                }
            }
        ];
        
        function runStep() {
            if (currentStep < steps.length && isAnimating) {
                steps[currentStep].action();
                currentStep++;
                animationTimer = setTimeout(runStep, 2000 / animationSpeed);
            } else {
                isAnimating = false;
            }
        }
        
        runStep();
    }
    
    function toggleAnimation() {
        if (!isAnimating && animationTimer) {
            // 暂停中，继续
            isAnimating = true;
            startMessagePassing();
        } else {
            // 运行中，暂停
            isAnimating = false;
            if (animationTimer) {
                clearTimeout(animationTimer);
                animationTimer = null;
            }
        }
    }
    
    function resetAnimation() {
        isAnimating = false;
        currentStep = 0;
        if (animationTimer) {
            clearTimeout(animationTimer);
            animationTimer = null;
        }
        
        // 清除消息粒子
        svg.selectAll('.message-particle').remove();
        
        // 重置节点状态
        nodes.forEach(function(node) {
            node.messages = [];
            node.newFeatures = [...node.features];
            node.active = false;
        });
        
        updateFeatureDisplay();
        highlightActiveNodes([]);
        stepDisplay.text('点击"开始消息传递"查看演示');
    }
    
    function sendMessagesFrom(nodeId) {
        var sourceNode = nodes[nodeId];
        var neighbors = adjacencyList[nodeId];
        
        neighbors.forEach(function(neighbor) {
            createMessageParticle(sourceNode, nodes[neighbor.neighbor], neighbor.weight);
        });
    }
    
    function createMessageParticle(source, target, weight) {
        var particle = svg.append('circle')
            .attr('class', 'message-particle')
            .attr('r', 4)
            .attr('cx', source.x)
            .attr('cy', source.y)
            .attr('fill', 'url(#messageGradient)')
            .attr('filter', 'url(#messageGlow)')
            .style('opacity', 0);
        
        particle.transition()
            .duration(200)
            .style('opacity', 1)
            .transition()
            .duration(1000 / animationSpeed * 1000)
            .attr('cx', target.x)
            .attr('cy', target.y)
            .on('end', function() {
                // 消息到达目标节点
                target.messages.push({
                    from: source.id,
                    features: source.features,
                    weight: weight
                });
                particle.remove();
            });
    }
    
    function highlightActiveNodes(activeNodeIds) {
        nodeCircle.transition()
            .duration(300)
            .attr('fill', function(d) {
                if (activeNodeIds.includes(d.id)) {
                    return d.id === 0 ? '#ff4757' : '#3742fa';
                } else {
                    return d.id === 0 ? '#ff6b6b' : '#3b82f6';
                }
            })
            .attr('filter', function(d) {
                return activeNodeIds.includes(d.id) ? 'url(#messageGlow)' : null;
            });
    }
    
    function aggregateAndUpdate() {
        nodes.forEach(function(node) {
            if (node.messages.length > 0) {
                // 简单的聚合函数：加权平均
                var aggregated = [0, 0, 0];
                var totalWeight = 0;
                
                node.messages.forEach(function(msg) {
                    for (var i = 0; i < 3; i++) {
                        aggregated[i] += msg.features[i] * msg.weight;
                    }
                    totalWeight += msg.weight;
                });
                
                // 归一化
                if (totalWeight > 0) {
                    for (var i = 0; i < 3; i++) {
                        aggregated[i] = (aggregated[i] / totalWeight + node.features[i]) / 2;
                        aggregated[i] = Math.round(aggregated[i] * 100) / 100; // 保留两位小数
                    }
                    node.newFeatures = aggregated;
                }
                
                node.messages = []; // 清空消息
            }
        });
        
        updateFeatureDisplay();
    }
    
    function secondRoundMessagePassing() {
        // 所有节点同时发送消息
        nodes.forEach(function(node) {
            sendMessagesFrom(node.id);
        });
        
        setTimeout(function() {
            aggregateAndUpdate();
        }, 1500 / animationSpeed * 1000);
    }
    
    function updateFeatureDisplay() {
        svg.selectAll('.feature-text')
            .transition()
            .duration(500)
            .style('fill', '#ff6b6b')
            .text(function(d) { return '[' + d.newFeatures.join(',') + ']'; })
            .transition()
            .duration(500)
            .style('fill', '#333');
    }
    
    function showCompletionEffect() {
        // 显示完成效果
        nodeCircle.transition()
            .duration(500)
            .attr('r', 25)
            .transition()
            .duration(500)
            .attr('r', 18);
    }
    
    function showNodeFeatures(node) {
        var featureInfo = `节点 ${node.name}:\n\n` +
                         `初始特征: [${node.features.join(', ')}]\n` +
                         `更新后特征: [${node.newFeatures.join(', ')}]\n\n` +
                         `接收消息数: ${node.messages.length}`;
        
        alert(featureInfo);
    }
}

// 页面加载时初始化演示
document.addEventListener('DOMContentLoaded', function() {
    // 延迟初始化，确保DOM完全加载
    setTimeout(function() {
        if (document.getElementById('graph-demo')) {
            initHomeGraphDemo();
        }
        if (document.getElementById('social-graph-demo')) {
            initSocialGraphDemo();
        }
        if (document.getElementById('message-passing-demo')) {
            initMessagePassingDemo();
        }
    }, 1000);
});

// 导出函数
window.runGCNCode = runGCNCode;
window.runGATCode = runGATCode;
window.runGraphSAGECode = runGraphSAGECode;
window.checkWeiboAnswer = checkWeiboAnswer;
window.switchExecMode = switchExecMode;
window.initHomeGraphDemo = initHomeGraphDemo;
window.initSocialGraphDemo = initSocialGraphDemo;
window.initMessagePassingDemo = initMessagePassingDemo;
window.createSocialNetworkDemo = createSocialNetworkDemo;
window.createMessagePassingDemo = createMessagePassingDemo;