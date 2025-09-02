// 图学习网站主要JavaScript功能

// 全局状态管理
var practiceState = {
    learningRate: 0.01,
    hiddenDim: 16,
    epochs: 200,
    cm: null,
    gatCm: null,
    savedCodeKey: 'graphlearn_gcn_code_v1',
    execMode: 'simulate',
    pyodide: null,
    pyodideReady: false,
    syntaxCheckTimer: null,
    currentAlgorithm: 'GCN',
    selectedAlgorithm: null,
    attentionHead: 1,
    timestep: 0,
    visualizationMode: 'attention',
    attentionWeights: null,
    attentionHistory: null,
    currentDataset: 'synthetic',
    coraTrainingResults: null
};

// 页面导航功能
function showSection(id) {
    document.querySelectorAll('.section').forEach(function(s) {
        s.classList.remove('active');
        s.style.display = 'none';
    });
    var el = document.getElementById(id + '-section');
    if (el) {
        el.classList.add('active');
        el.style.display = 'block';
    }
}

// 教程页面切换
function showTutorial(id) {
    document.querySelectorAll('.tutorial-content').forEach(function(s) {
        s.classList.remove('active');
        s.style.display = 'none';
    });
    var el = document.getElementById(id);
    if (el) {
        el.classList.add('active');
        el.style.display = 'block';
    }
}

// 社区页面切换
function showCommunitySection(id) {
    var sections = ['discussions', 'projects', 'pitfalls', 'challenges', 'resources'];
    sections.forEach(function(s) {
        var el = document.getElementById(s);
        if (el) el.style.display = (s === id) ? 'block' : 'none';
    });
}

// 在线实践环境管理
function loadPracticeEnvironment(key) {
    var placeholder = document.getElementById('env-placeholder');
    if (placeholder) placeholder.style.display = 'none';
    
    document.querySelectorAll('.practice-env').forEach(function(s) {
        s.classList.add('hidden');
    });
    
    if (key === 'gcn-classification') {
        var env = document.getElementById('gcn-classification-env');
        if (env) env.classList.remove('hidden');
        
        // 重置GCN环境的标题
        var title = document.querySelector('#gcn-classification-env h2');
        if (title) title.textContent = 'GCN节点分类实践';
        
        // 重置为默认的GCN代码
        initCodeEditor();
        if (practiceState.currentDataset === 'cora') {
            loadCoraGCNCode();
        } else {
            loadSyntheticGCNCode();
        }
        
        updateLearningRate(practiceState.learningRate);
        updateHiddenDim(practiceState.hiddenDim);
        updateEpochs(practiceState.epochs);
    } else if (key === 'gat-attention') {
        loadGATEnvironment();
    } else if (key === 'graphsage-sampling') {
        loadGraphSAGEEnvironment();
    } else {
        alert('该环境即将上线，敬请期待');
    }
}

function loadGATEnvironment() {
    var env = document.getElementById('gat-attention-env');
    if (env) env.classList.remove('hidden');
    
    initGATCodeEditor();
    practiceState.currentAlgorithm = 'GAT';
    practiceState.attentionHead = 1;
    practiceState.timestep = 0;
    
    var title = document.querySelector('#gat-attention-env h2');
    if (title) title.textContent = 'GAT 注意力机制可视化';
}

function initGATCodeEditor() {
    if (practiceState.gatCm || !window.CodeMirror) return;
    var textarea = document.getElementById('gat-code-editor');
    if (!textarea) return;
    
    practiceState.gatCm = CodeMirror.fromTextArea(textarea, {
        mode: 'python',
        theme: 'monokai',
        lineNumbers: true,
        viewportMargin: Infinity,
        autoCloseBrackets: true,
        matchBrackets: true
    });
}

function compareAlgorithms() {
    var comparisonData = {
        'GCN': {
            complexity: 'O(|E| + |V|D)',
            memory: '低',
            performance: '0.78-0.85',
            features: ['简单高效', '基础模型', '适合入门'],
            use_cases: ['节点分类', '图分类', '基础图学习任务']
        },
        'GAT': {
            complexity: 'O(|V|^2 D)',
            memory: '中等',
            performance: '0.82-0.88',
            features: ['注意力机制', '适应性强', '可解释性'],
            use_cases: ['复杂关系建模', '知识图谱', '信息推荐']
        },
        'GraphSAGE': {
            complexity: 'O(|V| K D)',
            memory: '低',
            performance: '0.80-0.86',
            features: ['归纳学习', '大图支持', '邻居采样'],
            use_cases: ['大规模图', '动态图', '工业级应用']
        }
    };
    
    var comparisonHtml = `
        <div style="background: white; padding: 20px; border-radius: 10px; max-width: 800px; max-height: 600px; overflow-y: auto;">
            <h3 style="margin-top: 0; color: #333; text-align: center;">📊 图神经网络算法比较</h3>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                <thead>
                    <tr style="background-color: #f8f9fa;">
                        <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">算法</th>
                        <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">时间复杂度</th>
                        <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">内存占用</th>
                        <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">性能范围</th>
                    </tr>
                </thead>
                <tbody>`;
    
    Object.keys(comparisonData).forEach(function(alg) {
        var data = comparisonData[alg];
        comparisonHtml += `
            <tr>
                <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold; color: ${alg === 'GCN' ? '#3b82f6' : alg === 'GAT' ? '#8b5cf6' : '#10b981'};">${alg}</td>
                <td style="padding: 12px; border: 1px solid #ddd;">${data.complexity}</td>
                <td style="padding: 12px; border: 1px solid #ddd;">${data.memory}</td>
                <td style="padding: 12px; border: 1px solid #ddd;">${data.performance}</td>
            </tr>`;
    });
    
    comparisonHtml += `
                </tbody>
            </table>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">`;
    
    Object.keys(comparisonData).forEach(function(alg) {
        var data = comparisonData[alg];
        comparisonHtml += `
            <div style="border: 2px solid ${alg === 'GCN' ? '#3b82f6' : alg === 'GAT' ? '#8b5cf6' : '#10b981'}; border-radius: 8px; padding: 15px;">
                <h4 style="margin-top: 0; color: ${alg === 'GCN' ? '#3b82f6' : alg === 'GAT' ? '#8b5cf6' : '#10b981'};">${alg}</h4>
                <div style="margin-bottom: 10px;">
                    <strong>特点:</strong>
                    <ul style="margin: 5px 0; padding-left: 20px;">
                        ${data.features.map(f => `<li style="font-size: 14px; color: #666;">${f}</li>`).join('')}
                    </ul>
                </div>
                <div>
                    <strong>适用场景:</strong>
                    <ul style="margin: 5px 0; padding-left: 20px;">
                        ${data.use_cases.map(u => `<li style="font-size: 14px; color: #666;">${u}</li>`).join('')}
                    </ul>
                </div>
            </div>`;
    });
    
    comparisonHtml += `
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <button onclick="closeComparisonModal()" style="background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">关闭</button>
            </div>
        </div>`;
    
    showModal(comparisonHtml);
}

function selectAlgorithm(algorithm) {
    practiceState.selectedAlgorithm = algorithm;
    
    // 更新按钮样式
    document.querySelectorAll('.algorithm-btn').forEach(function(btn) {
        btn.classList.remove('ring-2', 'ring-offset-2');
    });
    
    event.target.classList.add('ring-2', 'ring-offset-2');
    
    // 显示算法信息
    var info = {
        'GCN': '图卷积网络 - 通过聚合邻居节点信息学习节点表示',
        'GAT': '图注意力网络 - 使用注意力机制动态分配邻居权重',
        'GraphSAGE': '图采样聚合 - 通过邻居采样实现归纳学习'
    };
    
    var comparisonEl = document.getElementById('algorithm-comparison');
    if (comparisonEl) {
        comparisonEl.innerHTML = `<strong>${algorithm}</strong>: ${info[algorithm]}`;
    }
}

function updateAttentionHead(value) {
    practiceState.attentionHead = parseInt(value);
    var el = document.getElementById('head-value');
    if (el) el.textContent = value;
    
    // 更新注意力可视化
    if (practiceState.attentionWeights) {
        updateAttentionVisualization();
    }
}

function updateTimestep(value) {
    practiceState.timestep = parseInt(value);
    var el = document.getElementById('timestep-value');
    if (el) el.textContent = value;
    
    // 更新时间步可视化
    if (practiceState.attentionHistory) {
        updateTimestepVisualization();
    }
}

function changeVisualizationMode(mode) {
    practiceState.visualizationMode = mode;
    
    switch (mode) {
        case 'attention':
            showAttentionWeights();
            break;
        case 'flow':
            showInformationFlow();
            break;
        case 'embedding':
            showNodeEmbeddings();
            break;
    }
}

function updateAttentionVisualization() {
    if (!practiceState.attentionWeights) return;
    
    var weights = practiceState.attentionWeights[practiceState.attentionHead - 1];
    renderAttentionHeatmap(weights);
}

function renderAttentionHeatmap(weights) {
    var container = document.getElementById('attention-heatmap');
    if (!container || !window.d3) return;
    
    container.innerHTML = '';
    var width = container.clientWidth || 300;
    var height = 200;
    var svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
    
    var numNodes = weights.length;
    var cellSize = Math.min(width / numNodes, height / numNodes) - 2;
    
    var colorScale = d3.scaleSequential(d3.interpolateBlues)
        .domain([0, d3.max(weights, function(row) { return d3.max(row); })]);
    
    for (var i = 0; i < numNodes; i++) {
        for (var j = 0; j < numNodes; j++) {
            svg.append('rect')
                .attr('x', j * (cellSize + 2))
                .attr('y', i * (cellSize + 2))
                .attr('width', cellSize)
                .attr('height', cellSize)
                .attr('fill', colorScale(weights[i][j]))
                .attr('stroke', '#fff')
                .attr('stroke-width', 1)
                .on('mouseover', function(event) {
                    var tooltip = d3.select('body').append('div')
                        .style('position', 'absolute')
                        .style('background', 'rgba(0,0,0,0.8)')
                        .style('color', 'white')
                        .style('padding', '5px')
                        .style('border-radius', '3px')
                        .style('font-size', '12px')
                        .style('pointer-events', 'none')
                        .text(`注意力: ${weights[i][j].toFixed(3)}`)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 10) + 'px');
                    
                    setTimeout(function() { tooltip.remove(); }, 2000);
                });
        }
    }
    
    // 添加标签
    svg.selectAll('.row-label')
        .data(d3.range(numNodes))
        .enter().append('text')
        .attr('class', 'row-label')
        .attr('x', -5)
        .attr('y', function(d) { return d * (cellSize + 2) + cellSize / 2; })
        .attr('dy', '.35em')
        .attr('text-anchor', 'end')
        .attr('font-size', '10px')
        .text(function(d) { return d; });
    
    svg.selectAll('.col-label')
        .data(d3.range(numNodes))
        .enter().append('text')
        .attr('class', 'col-label')
        .attr('x', function(d) { return d * (cellSize + 2) + cellSize / 2; })
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .text(function(d) { return d; });
}

function showModal(content) {
    var modal = document.createElement('div');
    modal.id = 'comparison-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.5);
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    
    modal.innerHTML = content;
    document.body.appendChild(modal);
    
    // 点击背景关闭
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            closeComparisonModal();
        }
    });
}

function closeComparisonModal() {
    var modal = document.getElementById('comparison-modal');
    if (modal) {
        modal.remove();
    }
}

function loadGraphSAGEEnvironment() {
    // 显示GraphSAGE即将上线的提示，而不是修改GCN环境
    alert('🚀 GraphSAGE 邻居采样功能正在开发中，敬请期待！\n\n您可以先体验：\n• GCN 节点分类任务（包含Cora数据集）\n• GAT 注意力机制可视化');
}

// 代码编辑器初始化 - 增强版本
function initCodeEditor() {
    if (practiceState.cm || !window.CodeMirror) return;
    var textarea = document.getElementById('code-editor');
    if (!textarea) return;
    
    practiceState.cm = CodeMirror.fromTextArea(textarea, {
        mode: 'python',
        theme: 'monokai',
        lineNumbers: true,
        viewportMargin: Infinity,
        autoCloseBrackets: true,
        matchBrackets: true,
        indentUnit: 4,
        tabSize: 4,
        lineWrapping: false,
        foldGutter: true,
        gutters: ['CodeMirror-linenumbers', 'CodeMirror-foldgutter'],
        extraKeys: {
            'Ctrl-Space': 'autocomplete',
            'F11': function(cm) {
                cm.setOption('fullScreen', !cm.getOption('fullScreen'));
            },
            'Esc': function(cm) {
                if (cm.getOption('fullScreen')) cm.setOption('fullScreen', false);
            },
            'Ctrl-/': 'toggleComment',
            'Ctrl-F': 'findPersistent'
        },
        hintOptions: {
            hint: getCodeHints
        }
    });
    
    // 添加实时语法检查
    practiceState.cm.on('change', function(cm) {
        clearTimeout(practiceState.syntaxCheckTimer);
        practiceState.syntaxCheckTimer = setTimeout(function() {
            checkSyntax(cm);
        }, 1000);
    });
    
    // 添加代码提示
    practiceState.cm.on('cursorActivity', function(cm) {
        showCodeSuggestions(cm);
    });
    
    try {
        var saved = localStorage.getItem(practiceState.savedCodeKey);
        if (saved) {
            practiceState.cm.setValue(saved);
        } else {
            // 没有保存的代码时，根据当前数据集加载默认代码
            if (practiceState.currentDataset === 'cora') {
                loadCoraGCNCode();
            } else {
                loadSyntheticGCNCode();
            }
        }
    } catch (e) {
        // 如果发生错误，加载默认代码
        if (practiceState.currentDataset === 'cora') {
            loadCoraGCNCode();
        } else {
            loadSyntheticGCNCode();
        }
    }
    
    // 添加工具栏
    addCodeEditorToolbar();
    
    initPyodideOnce();
}

function getCodeHints(cm, options) {
    var cursor = cm.getCursor();
    var line = cm.getLine(cursor.line);
    var start = cursor.ch;
    var end = cursor.ch;
    
    // 获取当前单词
    while (start && /\w/.test(line.charAt(start - 1))) --start;
    while (end < line.length && /\w/.test(line.charAt(end))) ++end;
    var word = line.slice(start, end).toLowerCase();
    
    // Python/PyTorch 关键词和函数
    var keywords = [
        'import', 'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim',
        'class', 'def', 'for', 'if', 'else', 'elif', 'while', 'try', 'except',
        'GCN', 'GAT', 'GraphSAGE', 'forward', 'backward', 'zero_grad',
        'Linear', 'ReLU', 'Dropout', 'log_softmax', 'nll_loss',
        'Adam', 'SGD', 'parameters', 'train', 'eval'
    ];
    
    var suggestions = keywords.filter(function(item) {
        return item.toLowerCase().startsWith(word);
    });
    
    if (suggestions.length === 0) return null;
    
    return {
        list: suggestions,
        from: CodeMirror.Pos(cursor.line, start),
        to: CodeMirror.Pos(cursor.line, end)
    };
}

function checkSyntax(cm) {
    var code = cm.getValue();
    var lines = code.split('\n');
    var errors = [];
    
    // 简单的语法检查
    lines.forEach(function(line, i) {
        // 检查缩进
        if (line.trim() && line.match(/^\s*/)[0].length % 4 !== 0) {
            errors.push({
                line: i,
                message: '缩进不是4的倍数',
                severity: 'warning'
            });
        }
        
        // 检查常见错误
        if (line.includes('print(') && !line.includes(')')) {
            errors.push({
                line: i,
                message: '缺少右括号',
                severity: 'error'
            });
        }
    });
    
    // 显示错误
    displaySyntaxErrors(cm, errors);
}

function displaySyntaxErrors(cm, errors) {
    // 清除之前的错误标记
    cm.clearGutter('CodeMirror-lint-markers');
    
    errors.forEach(function(error) {
        var marker = document.createElement('div');
        marker.style.cssText = `
            width: 16px;
            height: 16px;
            background: ${error.severity === 'error' ? '#ef4444' : '#f59e0b'};
            border-radius: 50%;
            color: white;
            text-align: center;
            font-size: 10px;
            line-height: 16px;
            cursor: pointer;
        `;
        marker.textContent = error.severity === 'error' ? '×' : '!';
        marker.title = error.message;
        
        cm.setGutterMarker(error.line, 'CodeMirror-lint-markers', marker);
    });
}

function addCodeEditorToolbar() {
    var editorContainer = practiceState.cm.getWrapperElement().parentNode;
    var toolbar = document.createElement('div');
    toolbar.style.cssText = `
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 10px;
        background: #2f3349;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        font-size: 12px;
        color: #a1a1aa;
    `;
    
    toolbar.innerHTML = `
        <div>
            <span>🐍 Python | PyTorch</span>
            <button onclick="formatCode()" style="margin-left: 10px; padding: 2px 8px; background: #4f46e5; color: white; border: none; border-radius: 3px; cursor: pointer;">格式化</button>
            <button onclick="insertTemplate()" style="margin-left: 5px; padding: 2px 8px; background: #059669; color: white; border: none; border-radius: 3px; cursor: pointer;">插入模板</button>
        </div>
        <div>
            <span id="editor-status">就绪</span>
        </div>
    `;
    
    editorContainer.insertBefore(toolbar, practiceState.cm.getWrapperElement());
}

function formatCode() {
    if (!practiceState.cm) return;
    
    var code = practiceState.cm.getValue();
    var lines = code.split('\n');
    var formatted = [];
    var indentLevel = 0;
    
    lines.forEach(function(line) {
        var trimmed = line.trim();
        if (!trimmed) {
            formatted.push('');
            return;
        }
        
        // 减少缩进
        if (trimmed.startsWith('except') || trimmed.startsWith('elif') || 
            trimmed.startsWith('else') || trimmed.startsWith('finally')) {
            indentLevel = Math.max(0, indentLevel - 1);
        }
        
        formatted.push('    '.repeat(indentLevel) + trimmed);
        
        // 增加缩进
        if (trimmed.endsWith(':')) {
            indentLevel++;
        }
    });
    
    practiceState.cm.setValue(formatted.join('\n'));
}

function insertTemplate() {
    if (!practiceState.cm) return;
    
    var templates = {
        'GCN模板': `# GCN 基础模板
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gc1 = nn.Linear(input_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, adj):
        x = torch.mm(adj, x)
        x = F.relu(self.gc1(x))
        x = self.dropout(x)
        x = torch.mm(adj, x)
        x = self.gc2(x)
        return F.log_softmax(x, dim=1)`,
        
        '训练循环': `# 训练循环模板
model = GCN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(features, adj_matrix)
    loss = F.nll_loss(output, labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        _, pred = output.max(dim=1)
        acc = pred.eq(labels).float().mean()
        print(f'Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}')`
    };
    
    var templateNames = Object.keys(templates);
    var choice = prompt('选择模板：\n' + templateNames.map((t, i) => `${i + 1}. ${t}`).join('\n'));
    var index = parseInt(choice) - 1;
    
    if (index >= 0 && index < templateNames.length) {
        var selectedTemplate = templates[templateNames[index]];
        var cursor = practiceState.cm.getCursor();
        practiceState.cm.replaceRange(selectedTemplate, cursor);
    }
}

function showCodeSuggestions(cm) {
    var cursor = cm.getCursor();
    var line = cm.getLine(cursor.line);
    var status = document.getElementById('editor-status');
    
    if (status) {
        var suggestions = [];
        
        if (line.includes('torch.') && !line.includes('import')) {
            suggestions.push('提示：使用 import torch 导入');
        }
        
        if (line.includes('def ') && !line.endsWith(':')) {
            suggestions.push('提示：函数定义需要凒号');
        }
        
        status.textContent = suggestions.length > 0 ? suggestions[0] : '就绪';
    }
}

// Pyodide初始化
async function initPyodideOnce() {
    if (practiceState.pyodide || !window.loadPyodide) return;
    var consoleEl = document.getElementById('console-output');
    try {
        if (consoleEl) consoleEl.textContent = '正在加载 Pyodide (首次较慢)...';
        practiceState.pyodide = await loadPyodide({ 
            indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/' 
        });
        practiceState.pyodideReady = true;
        if (consoleEl) consoleEl.textContent = 'Pyodide 加载完成，可切换为 Pyodide 执行模式';
    } catch (e) {
        practiceState.pyodideReady = false;
        if (consoleEl) consoleEl.textContent = 'Pyodide 加载失败：' + e;
    }
}

// 参数调整功能 - 增强版本
function updateLearningRate(v) {
    var f = parseFloat(v);
    if (!isNaN(f)) practiceState.learningRate = f;
    var el = document.getElementById('lr-value');
    if (el) el.textContent = String(practiceState.learningRate);
    
    // 实时反馈
    showParameterFeedback('learning-rate', f);
    updateParameterRecommendation();
}

function updateHiddenDim(v) {
    var i = parseInt(v);
    if (!isNaN(i)) practiceState.hiddenDim = i;
    var el = document.getElementById('hidden-value');
    if (el) el.textContent = String(practiceState.hiddenDim);
    
    // 实时反馈
    showParameterFeedback('hidden-dim', i);
    updateParameterRecommendation();
}

function updateEpochs(v) {
    var i = parseInt(v);
    if (!isNaN(i)) practiceState.epochs = i;
    var el = document.getElementById('epochs-value');
    if (el) el.textContent = String(practiceState.epochs);
    
    // 实时反馈
    showParameterFeedback('epochs', i);
    updateParameterRecommendation();
}

function showParameterFeedback(paramType, value) {
    var feedbackId = paramType + '-feedback';
    var feedbackEl = document.getElementById(feedbackId);
    
    if (!feedbackEl) {
        // 创建反馈元素
        var paramContainer = document.querySelector(`[onchange*="${paramType.replace('-', '')}"]`).parentNode;
        feedbackEl = document.createElement('div');
        feedbackEl.id = feedbackId;
        feedbackEl.style.cssText = `
            font-size: 11px;
            margin-top: 3px;
            padding: 2px 5px;
            border-radius: 3px;
            transition: all 0.3s ease;
        `;
        paramContainer.appendChild(feedbackEl);
    }
    
    var feedback = getParameterFeedback(paramType, value);
    feedbackEl.textContent = feedback.message;
    feedbackEl.style.backgroundColor = feedback.color;
    feedbackEl.style.color = '#fff';
}

function getParameterFeedback(paramType, value) {
    switch (paramType) {
        case 'learning-rate':
            if (value > 0.1) return { message: '过高！可能发散', color: '#ef4444' };
            if (value > 0.05) return { message: '较高，注意稳定性', color: '#f59e0b' };
            if (value < 0.001) return { message: '过住！收敛慢', color: '#ef4444' };
            if (value < 0.005) return { message: '较低，收敛较慢', color: '#f59e0b' };
            return { message: '合理范围 ✅', color: '#10b981' };
            
        case 'hidden-dim':
            if (value < 8) return { message: '过小！表达能力不足', color: '#ef4444' };
            if (value > 128) return { message: '过大！可能过拟合', color: '#ef4444' };
            if (value > 64) return { message: '较大，注意过拟合', color: '#f59e0b' };
            return { message: '合理选择 ✅', color: '#10b981' };
            
        case 'epochs':
            if (value < 50) return { message: '过少！可能欠拟合', color: '#f59e0b' };
            if (value > 1000) return { message: '过多！耗时且容易过拟合', color: '#f59e0b' };
            return { message: '合理范围 ✅', color: '#10b981' };
            
        default:
            return { message: '', color: '#6b7280' };
    }
}

function updateParameterRecommendation() {
    var recEl = document.getElementById('param-recommendation');
    if (!recEl) {
        // 创建推荐区域
        var paramPanel = document.querySelector('.bg-gray-50');
        if (paramPanel) {
            recEl = document.createElement('div');
            recEl.id = 'param-recommendation';
            recEl.style.cssText = `
                margin-top: 15px;
                padding: 10px;
                background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                border-radius: 6px;
                color: white;
                font-size: 12px;
                line-height: 1.4;
            `;
            paramPanel.appendChild(recEl);
        }
    }
    
    if (recEl) {
        var recommendation = generateParameterRecommendation();
        recEl.innerHTML = `
            <div style="font-weight: bold; margin-bottom: 5px;">💡 智能推荐</div>
            <div>${recommendation}</div>
        `;
    }
}

function generateParameterRecommendation() {
    var lr = practiceState.learningRate;
    var hiddenDim = practiceState.hiddenDim;
    var epochs = practiceState.epochs;
    
    var recommendations = [];
    
    if (lr > 0.05 && hiddenDim < 32) {
        recommendations.push('高学习率 + 小模型：建议降低学习率至 0.01-0.03');
    } else if (lr < 0.01 && epochs < 200) {
        recommendations.push('低学习率 + 少轮次：建议增加训练轮数至 300-500');
    }
    
    if (hiddenDim >= 64 && epochs >= 300) {
        recommendations.push('大模型 + 多轮次：注意过拟合，可适当减少参数');
    }
    
    if (recommendations.length === 0) {
        var expectedAcc = estimatePerformance(lr, hiddenDim, epochs);
        recommendations.push(`当前参数配置预期准确率：${(expectedAcc * 100).toFixed(1)}%`);
    }
    
    return recommendations.join('<br/>');
}

function estimatePerformance(lr, hiddenDim, epochs) {
    // 简单的性能估算模型
    var baseCap = Math.min(0.95, 0.6 + Math.log10(1 + lr * 50) * 0.1 + (hiddenDim - 16) / 100);
    var epochsFactor = Math.min(1, epochs / 200);
    return baseCap * epochsFactor;
}

// 代码运行功能
async function runCode() {
    var consoleEl = document.getElementById('console-output') || document.getElementById('gat-console-output');
    if (consoleEl) {
        consoleEl.textContent = '';
        consoleEl.style.color = '#00ff00';
    }
    
    // 显示实时参数
    if (document.getElementById('console-output')) {
        showCurrentParameters();
    }

    // 检查是否为Cora GCN模式
    if (practiceState.execMode === 'cora-gcn') {
        await runCoraGCN();
        return;
    }

    if (practiceState.execMode === 'pyodide') {
        await runCodeWithPyodide();
        return;
    }

    // 检查当前是否是GAT环境
    var isGATEnvironment = !document.getElementById('gat-attention-env').classList.contains('hidden');
    
    if (isGATEnvironment) {
        await runGATSimulation();
        return;
    }

    // 前端模拟模式 - 增强版本
    var epochs = Math.max(50, practiceState.epochs);
    var lr = practiceState.learningRate;
    var hiddenDim = practiceState.hiddenDim;
    
    // 更真实的模拟算法
    var baseCap = Math.min(0.95, 0.6 + Math.log10(1 + lr * 50) * 0.1 + (hiddenDim - 16) / 100);
    var losses = [], accs = [];
    var loss0 = 1.5 - Math.min(0.8, Math.log10(1 + lr * 100) * 0.2);
    var convergenceRate = 0.85 + (hiddenDim / 64) * 0.1;
    
    // 创建训练进度指示器
    createTrainingProgressIndicator();
    
    for (var t = 0; t < epochs; t++) {
        var progress = t / (epochs - 1);
        
        // 更真实的损失下降曲线
        var noiseFactor = 1 + (Math.random() - 0.5) * 0.1; // 添加噪声
        var loss = loss0 * Math.pow(convergenceRate, progress * 8) * noiseFactor;
        loss = Math.max(0.01, loss); // 防止损失为负
        
        // 更真实的准确率上升曲线
        var accBase = (0.3 + 0.7 * Math.pow(progress, 0.7)) * baseCap;
        var acc = Math.min(baseCap, accBase * (1 + (Math.random() - 0.5) * 0.05));
        
        losses.push(parseFloat(loss.toFixed(4)));
        accs.push(parseFloat(acc.toFixed(4)));
        
        // 实时更新进度
        updateTrainingProgress(t, epochs, loss, acc);
        
        if (consoleEl && (t % Math.max(1, Math.floor(epochs / 10)) === 0)) {
            var logMsg = `Epoch ${t}: Loss=${losses[t].toFixed(4)}, Acc=${accs[t].toFixed(4)}`;
            consoleEl.textContent += logMsg + '\n';
            consoleEl.scrollTop = consoleEl.scrollHeight;
            
            // 实时更新图表
            if (t > 0 && t % 5 === 0) {
                renderCharts(losses.slice(0, t + 1), accs.slice(0, t + 1));
            }
        }
        
        // 添加延迟以模拟真实训练过程
        if (t < epochs - 1) {
            await new Promise(resolve => setTimeout(resolve, 50));
        }
    }
    
    if (consoleEl) {
        consoleEl.textContent += '\n🎉 训练完成！';
        consoleEl.textContent += `\n📊 最终准确率: ${accs[accs.length - 1].toFixed(4)}`;
        consoleEl.textContent += `\n📉 最终损失: ${losses[losses.length - 1].toFixed(4)}`;
        
        // 显示参数影响分析
        if (document.getElementById('console-output')) {
            var analysisText = analyzeParameters(lr, hiddenDim, accs[accs.length - 1]);
            consoleEl.textContent += '\n\n' + analysisText;
        }
    }

    // 最终渲染
    renderCharts(losses, accs);
    renderGraph(accs[accs.length - 1]);
    
    // 显示训练完成动画
    showTrainingCompleteAnimation();
}

// GAT模型模拟运行
async function runGATSimulation() {
    var consoleEl = document.getElementById('gat-console-output');
    if (consoleEl) {
        consoleEl.textContent = '';
        consoleEl.style.color = '#00ff00';
    }
    
    // 模拟GAT训练过程
    var epochs = 200;
    var losses = [], accs = [];
    var attentionWeights = [];
    
    if (consoleEl) {
        consoleEl.textContent = '🚀 开始训练GAT模型...\n\n';
    }
    
    for (var epoch = 0; epoch < epochs; epoch++) {
        var progress = epoch / (epochs - 1);
        
        // GAT损失函数模拟（通常收敛更快）
        var loss = 1.2 * Math.pow(0.9, progress * 6) * (1 + (Math.random() - 0.5) * 0.1);
        loss = Math.max(0.01, loss);
        
        // GAT准确率模拟（通常性能更好）
        var acc = Math.min(0.92, 0.25 + 0.7 * Math.pow(progress, 0.6) * (1 + (Math.random() - 0.5) * 0.05));
        
        losses.push(parseFloat(loss.toFixed(4)));
        accs.push(parseFloat(acc.toFixed(4)));
        
        // 模拟注意力权重
        if (epoch % 40 === 0) {
            var attentionMatrix = [];
            for (var i = 0; i < 7; i++) {
                var row = [];
                for (var j = 0; j < 7; j++) {
                    row.push(Math.random() * 0.8 + 0.2); // 0.2-1.0之间的随机权重
                }
                attentionMatrix.push(row);
            }
            attentionWeights.push(attentionMatrix);
            
            if (consoleEl) {
                consoleEl.textContent += `Epoch ${epoch}: Loss=${loss.toFixed(4)}, Acc=${acc.toFixed(4)}\n`;
                consoleEl.textContent += `注意力权重已更新 (${attentionWeights.length}/5)\n`;
            }
        }
        
        // 实时更新图表
        if (epoch > 0 && epoch % 10 === 0) {
            renderGATCharts(losses.slice(0, epoch + 1), accs.slice(0, epoch + 1));
            
            // 更新注意力权重可视化
            if (attentionWeights.length > 0) {
                practiceState.attentionWeights = attentionWeights;
                renderAttentionHeatmap(attentionWeights[attentionWeights.length - 1]);
                renderAttentionDistribution(attentionWeights[attentionWeights.length - 1]);
            }
        }
        
        await new Promise(resolve => setTimeout(resolve, 20));
    }
    
    if (consoleEl) {
        consoleEl.textContent += '\n🎉 GAT训练完成！\n';
        consoleEl.textContent += `📊 最终准确率: ${accs[accs.length - 1].toFixed(4)}\n`;
        consoleEl.textContent += `🎯 注意力机制成功收敛\n`;
        consoleEl.textContent += `📈 共生成 ${attentionWeights.length} 个注意力权重快照`;
    }
    
    // 保存注意力权重用于可视化
    practiceState.attentionWeights = attentionWeights;
    practiceState.attentionHistory = attentionWeights;
}

// GAT专用图表渲染
function renderGATCharts(losses, accs) {
    if (!window.Plotly) return;
    
    // GAT损失曲线
    var lossChart = document.getElementById('gat-loss-chart');
    if (lossChart) {
        Plotly.newPlot('gat-loss-chart', [{
            y: losses, 
            type: 'scatter', 
            name: 'GAT Loss', 
            line: { color: '#8b5cf6' }
        }], {
            margin: { t: 10, b: 30, l: 40, r: 10 }, 
            yaxis: { title: 'Loss' }
        });
    }
}

// 注意力分布渲染
function renderAttentionDistribution(weights) {
    if (!window.Plotly || !weights) return;
    
    var container = document.getElementById('attention-distribution');
    if (!container) return;
    
    // 计算注意力权重分布
    var flatWeights = weights.flat();
    
    Plotly.newPlot('attention-distribution', [{
        x: flatWeights,
        type: 'histogram',
        name: '注意力权重分布',
        marker: { color: '#8b5cf6' }
    }], {
        margin: { t: 10, b: 30, l: 40, r: 10 },
        xaxis: { title: '权重值' },
        yaxis: { title: '频次' }
    });
}

function showCurrentParameters() {
    var consoleEl = document.getElementById('console-output');
    if (consoleEl) {
        consoleEl.textContent = '📋 当前训练参数:\n';
        consoleEl.textContent += `  • 学习率: ${practiceState.learningRate}\n`;
        consoleEl.textContent += `  • 隐藏维度: ${practiceState.hiddenDim}\n`;
        consoleEl.textContent += `  • 训练轮数: ${practiceState.epochs}\n`;
        consoleEl.textContent += `  • 执行模式: ${practiceState.execMode === 'pyodide' ? 'Pyodide真实执行' : '前端模拟'}\n\n`;
        consoleEl.textContent += '🚀 开始训练...\n\n';
    }
}

function createTrainingProgressIndicator() {
    var container = document.getElementById('console-output');
    if (!container) return;
    
    // 创建进度条容器
    var progressContainer = document.createElement('div');
    progressContainer.id = 'training-progress';
    progressContainer.style.cssText = `
        margin: 10px 0;
        padding: 10px;
        background: rgba(16, 185, 129, 0.1);
        border-radius: 5px;
        border-left: 4px solid #10b981;
    `;
    
    progressContainer.innerHTML = `
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span id="epoch-info">Epoch 0 / 0</span>
            <span id="metrics-info">Loss: -, Acc: -</span>
        </div>
        <div style="background: #e5e7eb; height: 8px; border-radius: 4px; overflow: hidden;">
            <div id="progress-bar" style="width: 0%; height: 100%; background: linear-gradient(90deg, #10b981, #3b82f6); transition: width 0.3s ease;"></div>
        </div>
    `;
    
    container.parentNode.insertBefore(progressContainer, container.nextSibling);
}

function updateTrainingProgress(epoch, totalEpochs, loss, acc) {
    var progressBar = document.getElementById('progress-bar');
    var epochInfo = document.getElementById('epoch-info');
    var metricsInfo = document.getElementById('metrics-info');
    
    if (progressBar && epochInfo && metricsInfo) {
        var progress = ((epoch + 1) / totalEpochs) * 100;
        progressBar.style.width = progress + '%';
        epochInfo.textContent = `Epoch ${epoch + 1} / ${totalEpochs}`;
        metricsInfo.textContent = `Loss: ${loss.toFixed(4)}, Acc: ${acc.toFixed(4)}`;
    }
}

function analyzeParameters(lr, hiddenDim, finalAcc) {
    var analysis = '🔍 参数影响分析:\n';
    
    if (lr > 0.05) {
        analysis += '  ⚠️  学习率偏高，可能导致训练不稳定\n';
    } else if (lr < 0.005) {
        analysis += '  ⚠️  学习率偏低，收敛可能较慢\n';
    } else {
        analysis += '  ✅ 学习率设置合理\n';
    }
    
    if (hiddenDim < 8) {
        analysis += '  ⚠️  隐藏维度过小，模型表达能力有限\n';
    } else if (hiddenDim > 64) {
        analysis += '  ⚠️  隐藏维度过大，可能过拟合\n';
    } else {
        analysis += '  ✅ 隐藏维度设置合理\n';
    }
    
    if (finalAcc > 0.85) {
        analysis += '  🎯 训练效果优秀！\n';
    } else if (finalAcc > 0.7) {
        analysis += '  👍 训练效果良好\n';
    } else {
        analysis += '  📈 建议调整参数以提升性能\n';
    }
    
    return analysis;
}

function showTrainingCompleteAnimation() {
    // 为训练进度容器添加完成动画
    var progressContainer = document.getElementById('training-progress');
    if (progressContainer) {
        progressContainer.style.background = 'rgba(16, 185, 129, 0.2)';
        progressContainer.style.borderLeftColor = '#059669';
        
        // 添加成功图标
        var successIcon = document.createElement('div');
        successIcon.innerHTML = '✅ 训练完成';
        successIcon.style.cssText = `
            color: #059669;
            font-weight: bold;
            text-align: center;
            margin-top: 10px;
            animation: pulse 1s ease-in-out;
        `;
        progressContainer.appendChild(successIcon);
        
        // 添加CSS动画
        if (!document.getElementById('training-animations')) {
            var style = document.createElement('style');
            style.id = 'training-animations';
            style.textContent = `
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.05); }
                    100% { transform: scale(1); }
                }
            `;
            document.head.appendChild(style);
        }
    }
}

async function runCodeWithPyodide() {
    if (!practiceState.pyodideReady) {
        var consoleEl = document.getElementById('console-output');
        if (consoleEl) consoleEl.textContent = 'Pyodide 正在加载或不可用，请稍候或切换到前端模拟模式。';
        return;
    }
    
    var code = practiceState.cm ? practiceState.cm.getValue() : 
               (document.getElementById('code-editor') || {}).value || '';
    
    try {
        practiceState.pyodide.runPython(`
import sys
from js import console
class _C:
  def write(self,s):
    if s: console.log(s)
  def flush(self):
    pass
sys.stdout=_C()
sys.stderr=_C()`);
        
        await practiceState.pyodide.runPythonAsync(code);
        var result = await extractPyResults();
        var consoleEl = document.getElementById('console-output');
        if (consoleEl) consoleEl.textContent = '执行完成（Pyodide）。';
        
        renderCharts(result.losses || [1, 0.8], result.accuracies || [0.3, 0.7]);
        renderGraph((result.accuracies || []).slice(-1)[0] || 0.8, 
                   result.pred_correct || [], result.neighbors_sampled || {});
    } catch (e) {
        var consoleEl = document.getElementById('console-output');
        if (consoleEl) consoleEl.textContent = '执行失败（Pyodide）：' + e;
        renderCharts([1.0, 0.8, 0.6, 0.5, 0.45, 0.4], [0.2, 0.35, 0.5, 0.62, 0.73, 0.8]);
        renderGraph(0.8);
    }
}

// 其他工具函数
function resetCode() {
    var consoleEl = document.getElementById('console-output');
    if (consoleEl) consoleEl.textContent = '点击"运行代码"查看结果...';
    
    // 清除训练进度指示器
    var progressContainer = document.getElementById('training-progress');
    if (progressContainer) {
        progressContainer.remove();
    }
    
    if (window.Plotly) {
        try { Plotly.purge('loss-chart'); } catch(e) {}
        try { Plotly.purge('accuracy-chart'); } catch(e) {}
    }
    
    var gv = document.getElementById('graph-visualization');
    if (gv) gv.innerHTML = '<div class="text-center"><p>图结构可视化</p><p class="text-sm">运行代码后显示节点分类结果</p></div>';
    
    // 重置 Cora 相关状态
    practiceState.coraTrainingResults = null;
    
    // 根据当前数据集重置代码
    if (practiceState.currentDataset === 'cora') {
        loadCoraGCNCode();
    } else {
        loadSyntheticGCNCode();
    }
}

function saveCode() {
    if (practiceState.cm) {
        try {
            localStorage.setItem(practiceState.savedCodeKey, practiceState.cm.getValue());
            alert('已保存到本地浏览器');
        } catch(e) { 
            alert('保存失败：浏览器不支持或存储空间不足'); 
        }
    }
}

// 页面加载完成后的初始化
document.addEventListener('DOMContentLoaded', function() {
    renderAuthState();
    showCommunitySection('discussions');
    showSection('home'); // 默认显示首页
});

// 导出主要函数到全局作用域
window.showSection = showSection;
window.showTutorial = showTutorial;
window.showCommunitySection = showCommunitySection;
window.loadPracticeEnvironment = loadPracticeEnvironment;
window.updateLearningRate = updateLearningRate;
window.updateHiddenDim = updateHiddenDim;
window.updateEpochs = updateEpochs;
window.runCode = runCode;
window.resetCode = resetCode;
window.saveCode = saveCode;
window.compareAlgorithms = compareAlgorithms;
window.selectAlgorithm = selectAlgorithm;
window.updateAttentionHead = updateAttentionHead;
window.updateTimestep = updateTimestep;
window.changeVisualizationMode = changeVisualizationMode;
window.closeComparisonModal = closeComparisonModal;
window.formatCode = formatCode;
window.insertTemplate = insertTemplate;
window.switchDataset = switchDataset;
window.switchExecMode = switchExecMode;
window.runCoraGCN = runCoraGCN;
window.practiceState = practiceState;

// Cora数据集相关函数
function switchDataset(datasetType) {
    practiceState.currentDataset = datasetType;
    var coraInfo = document.getElementById('cora-dataset-info');
    
    if (datasetType === 'cora') {
        if (coraInfo) coraInfo.classList.remove('hidden');
        loadCoraGCNCode();
        console.log('已切换到Cora数据集');
    } else {
        if (coraInfo) coraInfo.classList.add('hidden');
        loadSyntheticGCNCode();
        console.log('已切换到简单示例数据集');
    }
}

function loadCoraGCNCode() {
    var coraCode = `# Cora数据集 GCN 节点分类完整实现
# 这是一个真实的图学习任务，使用著名Cora论文引用数据集

# 数据集信息：
# - 2708个论文节点
# - 1433维特征向量（词袋表示）
# - 5429条引用连接
# - 7个分类类别

# 点击下面的“运行代码”按钮开始训练！
# 注意：需要将执行模式切换到“Cora GCN 实践”

print("欢迎使用Cora数据集GCN节点分类功能！")
print("\n数据集信息：")
print("- 论文节点数：2708")
print("- 特征维度：1433")
print("- 引用连接数：5429")
print("- 分类类别：7种")
print("\n请将执行模式切换到 'Cora GCN 实践' 然后点击运行按钮！")
print("训练过程中您可以实时观察：")
print("1. 损失函数和准确率变化")
print("2. 实时的训练进度")
print("3. Cora数据集的图结构可视化")
print("4. 节点分类结果展示")

# 这里显示的是接口代码，实际GCN实现在cora-gcn.js中
print("\n🚀 准备开始Cora GCN节点分类任务...")`;
    
    if (practiceState.cm) {
        practiceState.cm.setValue(coraCode);
    }
}

function loadSyntheticGCNCode() {
    var syntheticCode = `# GCN 节点分类示例
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gc1 = nn.Linear(input_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, adj):
        # 第一层GCN
        x = torch.mm(adj, x)  # 聚合邻居信息
        x = self.gc1(x)       # 线性变换
        x = F.relu(x)         # 激活函数
        x = self.dropout(x)   # 防止过拟合
        
        # 第二层GCN
        x = torch.mm(adj, x)
        x = self.gc2(x)
        
        return F.log_softmax(x, dim=1)

# 创建示例数据
features = torch.randn(7, 5)  # 7个节点，每个5维特征
adj_matrix = torch.eye(7) + torch.randn(7, 7).abs() * 0.3  # 邻接矩阵
labels = torch.tensor([0, 0, 1, 1, 2, 2, 2])  # 节点标签

# 创建和训练模型
model = GCN(input_dim=5, hidden_dim=16, output_dim=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("开始训练GCN模型...")
for epoch in range(100):
    optimizer.zero_grad()
    output = model(features, adj_matrix)
    loss = F.nll_loss(output, labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        _, pred = output.max(dim=1)
        acc = pred.eq(labels).float().mean()
        print(f'Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}')

print("训练完成！")`;
    
    if (practiceState.cm) {
        practiceState.cm.setValue(syntheticCode);
    }
}

function switchExecMode(mode) {
    practiceState.execMode = mode;
    console.log('已切换执行模式到:', mode);
    
    var consoleEl = document.getElementById('console-output');
    if (consoleEl) {
        if (mode === 'cora-gcn') {
            consoleEl.textContent = '🎆 已切换到Cora GCN实践模式！\n点击运行按钮开始完整的Cora数据集GCN训练...';
        } else if (mode === 'pyodide') {
            consoleEl.textContent = '🔍 已切换到Pyodide模式！\n将使用真实的Python环境执行代码...';
        } else {
            consoleEl.textContent = '🎮 已切换到前端模拟模式！\n点击运行按钮查看结果...';
        }
    }
}

async function runCoraGCN() {
    if (!window.trainCoraGCN) {
        var consoleEl = document.getElementById('console-output');
        if (consoleEl) {
            consoleEl.textContent = '❗ Cora GCN模块未加载，请检查cora-gcn.js文件是否正确引入。';
        }
        return;
    }
    
    var consoleEl = document.getElementById('console-output');
    if (consoleEl) {
        consoleEl.textContent = '🚀 初始化Cora数据集...\n';
    }
    
    // 创建训练进度指示器
    createTrainingProgressIndicator();
    
    try {
        // 调用cora-gcn.js中的训练函数
        var results = await window.trainCoraGCN({
            hiddenDim: practiceState.hiddenDim,
            learningRate: practiceState.learningRate,
            epochs: practiceState.epochs,
            onProgress: function(progress) {
                // 更新进度显示
                updateTrainingProgress(
                    progress.epoch, 
                    practiceState.epochs, 
                    progress.loss, 
                    progress.trainAcc
                );
                
                // 更新控制台输出
                if (consoleEl) {
                    consoleEl.textContent += `Epoch ${progress.epoch}: Loss=${progress.loss.toFixed(4)}, TrainAcc=${progress.trainAcc.toFixed(4)}, ValAcc=${progress.valAcc.toFixed(4)}, TestAcc=${progress.testAcc.toFixed(4)}\n`;
                    consoleEl.scrollTop = consoleEl.scrollHeight;
                }
            },
            onComplete: function(results) {
                practiceState.coraTrainingResults = results;
                
                if (consoleEl) {
                    consoleEl.textContent += `\n✨ Cora GCN训练完成！\n`;
                    consoleEl.textContent += `最终测试准确率: ${results.testAccuracy.toFixed(4)}\n`;
                    consoleEl.textContent += analyzeParameters(
                        practiceState.learningRate, 
                        practiceState.hiddenDim, 
                        results.testAccuracy
                    );
                }
                
                // 渲染图表
                renderCoraCharts(results.history);
                
                // 可视化Cora图结构
                renderCoraGraph(results);
                
                // 显示完成动画
                showTrainingCompleteAnimation();
            }
        });
    } catch (error) {
        console.error('Cora GCN训练失败:', error);
        if (consoleEl) {
            consoleEl.textContent += `\n❗ 训练失败: ${error.message}\n`;
        }
    }
}

function renderCoraCharts(history) {
    if (!window.Plotly || !history) return;
    
    // 渲染损失函数图表
    var lossData = {
        x: Array.from({length: history.losses.length}, (_, i) => i),
        y: history.losses,
        type: 'scatter',
        mode: 'lines',
        name: '训练损失',
        line: { color: '#ef4444', width: 2 }
    };
    
    Plotly.newPlot('loss-chart', [lossData], {
        title: 'Cora GCN 损失函数',
        xaxis: { title: 'Epoch' },
        yaxis: { title: 'Loss' },
        margin: { t: 30, r: 20, b: 30, l: 40 }
    }, { responsive: true });
    
    // 渲染准确率图表
    var accData = [
        {
            x: Array.from({length: history.trainAccs.length}, (_, i) => i),
            y: history.trainAccs,
            type: 'scatter',
            mode: 'lines',
            name: '训练准确率',
            line: { color: '#3b82f6', width: 2 }
        },
        {
            x: Array.from({length: history.valAccs.length}, (_, i) => i),
            y: history.valAccs,
            type: 'scatter',
            mode: 'lines',
            name: '验证准确率',
            line: { color: '#10b981', width: 2 }
        },
        {
            x: Array.from({length: history.testAccs.length}, (_, i) => i),
            y: history.testAccs,
            type: 'scatter',
            mode: 'lines',
            name: '测试准确率',
            line: { color: '#f59e0b', width: 2 }
        }
    ];
    
    Plotly.newPlot('accuracy-chart', accData, {
        title: 'Cora GCN 准确率',
        xaxis: { title: 'Epoch' },
        yaxis: { title: 'Accuracy' },
        margin: { t: 30, r: 20, b: 30, l: 40 }
    }, { responsive: true });
}

function renderCoraGraph(results) {
    var graphContainer = document.getElementById('graph-visualization');
    if (!graphContainer || !window.visualizeCoraSample) return;
    
    // 清空容器
    graphContainer.innerHTML = '';
    
    // 使用cora-gcn.js中的可视化函数
    window.visualizeCoraSample('graph-visualization', 30);
    
    // 添加结果信息
    setTimeout(function() {
        var infoDiv = document.createElement('div');
        infoDiv.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            z-index: 10;
        `;
        infoDiv.innerHTML = `
            <div><strong>Cora 数据集</strong></div>
            <div>节点: ${results.dataset.numNodes}</div>
            <div>连接: ${results.dataset.numEdges || 5429}</div>
            <div>类别: ${results.dataset.numClasses}</div>
            <div>测试准确率: ${results.testAccuracy.toFixed(3)}</div>
        `;
        
        if (graphContainer.style.position !== 'relative') {
            graphContainer.style.position = 'relative';
        }
        graphContainer.appendChild(infoDiv);
    }, 100);
}