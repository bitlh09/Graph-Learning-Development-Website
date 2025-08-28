// 在线实践环境模块
const Playground = {
    // 实践环境状态
    state: {
        learningRate: 0.01,
        hiddenDim: 16,
        epochs: 200,
        cm: null,
        savedCodeKey: 'graphlearn_gcn_code_v1',
        execMode: 'simulate',
        pyodide: null,
        pyodideReady: false,
        currentEnvironment: null
    },

    // 初始化实践环境
    init() {
        this.bindEvents();
        this.initCodeEditor();
        this.initPyodide();
    },

    // 绑定事件
    bindEvents() {
        // 环境选择事件
        document.addEventListener('click', (e) => {
            if (e.target.closest('[onclick*="loadPracticeEnvironment"]')) {
                e.preventDefault();
                const match = e.target.closest('[onclick*="loadPracticeEnvironment"]')
                    .getAttribute('onclick').match(/loadPracticeEnvironment\('([^']+)'\)/);
                if (match) {
                    this.loadPracticeEnvironment(match[1]);
                }
            }
        });

        // 代码运行事件
        document.addEventListener('click', (e) => {
            if (e.target.matches('[onclick*="runCode"]')) {
                e.preventDefault();
                this.runCode();
            }
        });

        // 代码重置事件
        document.addEventListener('click', (e) => {
            if (e.target.matches('[onclick*="resetCode"]')) {
                e.preventDefault();
                this.resetCode();
            }
        });

        // 代码保存事件
        document.addEventListener('click', (e) => {
            if (e.target.matches('[onclick*="saveCode"]')) {
                e.preventDefault();
                this.saveCode();
            }
        });
    },

    // 初始化代码编辑器
    initCodeEditor() {
        if (this.state.cm || !window.CodeMirror) return;
        
        const textarea = document.getElementById('code-editor');
        if (!textarea) return;

        this.state.cm = CodeMirror.fromTextArea(textarea, {
            mode: 'python',
            theme: 'monokai',
            lineNumbers: true,
            viewportMargin: Infinity,
            autoCloseBrackets: true,
            matchBrackets: true,
            indentUnit: 4,
            tabSize: 4,
            lineWrapping: true
        });

        // 加载历史保存的代码
        this.loadSavedCode();
    },

    // 初始化Pyodide
    async initPyodide() {
        if (this.state.pyodide || !window.loadPyodide) return;
        
        const consoleEl = document.getElementById('console-output');
        try {
            if (consoleEl) consoleEl.textContent = '正在加载 Pyodide (首次较慢)...';
            this.state.pyodide = await loadPyodide({ 
                indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/' 
            });
            this.state.pyodideReady = true;
            if (consoleEl) consoleEl.textContent = 'Pyodide 加载完成，可切换为 Pyodide 执行模式';
        } catch (e) {
            this.state.pyodideReady = false;
            if (consoleEl) consoleEl.textContent = 'Pyodide 加载失败：' + e;
        }
    },

    // 加载实践环境
    loadPracticeEnvironment(key) {
        const placeholder = document.getElementById('env-placeholder');
        if (placeholder) placeholder.style.display = 'none';
        
        document.querySelectorAll('.practice-env').forEach(env => {
            env.classList.add('hidden');
        });

        this.state.currentEnvironment = key;

        if (key === 'gcn-classification') {
            this.loadGCNEnvironment();
        } else if (key === 'graphsage-sampling') {
            this.loadGraphSAGEEnvironment();
        } else if (key === 'gat-attention') {
            this.loadGATEnvironment();
        } else {
            alert('该环境即将上线，敬请期待');
        }
    },

    // 加载GCN环境
    loadGCNEnvironment() {
        const env = document.getElementById('gcn-classification-env');
        if (env) env.classList.remove('hidden');
        
        this.initCodeEditor();
        this.loadGCNCode();
        this.updateEnvironmentTitle('GCN节点分类实践');
    },

    // 加载GraphSAGE环境
    loadGraphSAGEEnvironment() {
        const env = document.getElementById('gcn-classification-env');
        if (env) env.classList.remove('hidden');
        
        this.initCodeEditor();
        this.loadGraphSAGECode();
        this.updateEnvironmentTitle('GraphSAGE 邻居采样实践');
    },

    // 加载GAT环境
    loadGATEnvironment() {
        const env = document.getElementById('gcn-classification-env');
        if (env) env.classList.remove('hidden');
        
        this.initCodeEditor();
        this.loadGATCode();
        this.updateEnvironmentTitle('GAT注意力可视化实践');
    },

    // 更新环境标题
    updateEnvironmentTitle(title) {
        const titleEl = document.querySelector('#gcn-classification-env h2');
        if (titleEl) titleEl.textContent = title;
    },

    // 加载GCN代码
    loadGCNCode() {
        const gcnCode = `import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 生成示例图数据
def generate_sample_data():
    num_nodes = 7
    features = torch.randn(num_nodes, 5)
    adj_matrix = torch.tensor([
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 2])
    return features, adj_matrix, labels

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gc1 = nn.Linear(input_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, adj):
        x = torch.mm(adj, x)
        x = self.gc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = torch.mm(adj, x)
        x = self.gc2(x)
        return F.log_softmax(x, dim=1)

def train_gcn():
    features, adj_matrix, labels = generate_sample_data()
    model = GCN(input_dim=5, hidden_dim=16, output_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    losses = []
    accuracies = []
    
    print("开始训练GCN模型...")
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_matrix)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        
        _, pred = output.max(dim=1)
        acc = pred.eq(labels).float().mean()
        losses.append(loss.item())
        accuracies.append(acc.item())
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}')
    
    return model, features, adj_matrix, labels, losses, accuracies

if __name__ == "__main__":
    model, features, adj_matrix, labels, losses, accuracies = train_gcn()
    print(f"最终准确率: {accuracies[-1]:.4f}")`;

        if (this.state.cm) {
            this.state.cm.setValue(gcnCode);
        }
    },

    // 加载GraphSAGE代码
    loadGraphSAGECode() {
        const graphsageCode = `import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 生成示例图数据
def generate_sample_data():
    num_nodes = 10
    features = torch.randn(num_nodes, 5)
    adj_matrix = torch.tensor([
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    return features, adj_matrix, labels

# 邻居采样函数
def sample_neighbors(adj_matrix, node_idx, num_samples):
    neighbors = torch.where(adj_matrix[node_idx] > 0)[0]
    if len(neighbors) > num_samples:
        sampled_indices = torch.randperm(len(neighbors))[:num_samples]
        return neighbors[sampled_indices]
    else:
        if len(neighbors) == 0:
            return torch.tensor([node_idx])
        repeated = neighbors.repeat((num_samples + len(neighbors) - 1) // len(neighbors))
        return repeated[:num_samples]

class MeanAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MeanAggregator, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, node_features, neighbor_features):
        aggregated = torch.mean(neighbor_features, dim=0)
        return self.linear(aggregated)

class GraphSAGELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_samples=3):
        super(GraphSAGELayer, self).__init__()
        self.num_samples = num_samples
        self.aggregator = MeanAggregator(input_dim, output_dim)
        self.linear = nn.Linear(input_dim + output_dim, output_dim)
        
    def forward(self, node_features, adj_matrix):
        batch_size = node_features.size(0)
        aggregated_features = []
        
        for i in range(batch_size):
            neighbors = sample_neighbors(adj_matrix, i, self.num_samples)
            neighbor_features = node_features[neighbors]
            aggregated = self.aggregator(node_features[i], neighbor_features)
            aggregated_features.append(aggregated)
        
        aggregated_features = torch.stack(aggregated_features)
        combined = torch.cat([node_features, aggregated_features], dim=1)
        output = self.linear(combined)
        return F.relu(output)

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_samples=3):
        super(GraphSAGE, self).__init__()
        self.sage1 = GraphSAGELayer(input_dim, hidden_dim, num_samples)
        self.sage2 = GraphSAGELayer(hidden_dim, output_dim, num_samples)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, adj):
        x = self.sage1(x, adj)
        x = self.dropout(x)
        x = self.sage2(x, adj)
        return F.log_softmax(x, dim=1)

def train_graphsage():
    features, adj_matrix, labels = generate_sample_data()
    model = GraphSAGE(input_dim=5, hidden_dim=16, output_dim=3, num_samples=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    losses = []
    accuracies = []
    
    print("开始训练 GraphSAGE 模型...")
    print(f"图规模: {features.size(0)} 个节点")
    print(f"邻居采样数: 3")
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_matrix)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        
        _, pred = output.max(dim=1)
        acc = pred.eq(labels).float().mean()
        losses.append(loss.item())
        accuracies.append(acc.item())
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}')
    
    return model, features, adj_matrix, labels, losses, accuracies

if __name__ == "__main__":
    model, features, adj_matrix, labels, losses, accuracies = train_graphsage()
    print(f"最终准确率: {accuracies[-1]:.4f}")`;

        if (this.state.cm) {
            this.state.cm.setValue(graphsageCode);
        }
    },

    // 加载GAT代码
    loadGATCode() {
        const gatCode = `import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 生成示例图数据
def generate_sample_data():
    num_nodes = 7
    features = torch.randn(num_nodes, 5)
    adj_matrix = torch.tensor([
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 2])
    return features, adj_matrix, labels

class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super(GATLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        self.W = nn.Linear(input_dim, output_dim)
        self.attention = nn.Linear(2 * self.head_dim, 1)
        
    def forward(self, x, adj):
        batch_size = x.size(0)
        x = self.W(x)
        x = x.view(batch_size, self.num_heads, self.head_dim)
        
        attention_scores = []
        for i in range(batch_size):
            for j in range(batch_size):
                if adj[i, j] > 0:
                    concat_features = torch.cat([x[i], x[j]], dim=0)
                    score = self.attention(concat_features)
                    attention_scores.append(score)
                else:
                    attention_scores.append(torch.tensor(-1e9))
        
        attention_scores = torch.stack(attention_scores).view(batch_size, batch_size)
        attention_probs = F.softmax(attention_scores, dim=1)
        output = torch.mm(attention_probs, x.view(batch_size, -1))
        
        return output, attention_probs

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(GAT, self).__init__()
        self.gat1 = GATLayer(input_dim, hidden_dim, num_heads)
        self.gat2 = GATLayer(hidden_dim, output_dim, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, adj):
        x, attn1 = self.gat1(x, adj)
        x = F.relu(x)
        x = self.dropout(x)
        x, attn2 = self.gat2(x, adj)
        return F.log_softmax(x, dim=1), attn1, attn2

def train_gat():
    features, adj_matrix, labels = generate_sample_data()
    model = GAT(input_dim=5, hidden_dim=16, output_dim=3, num_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    losses = []
    accuracies = []
    
    print("开始训练GAT模型...")
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output, attn1, attn2 = model(features, adj_matrix)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        
        _, pred = output.max(dim=1)
        acc = pred.eq(labels).float().mean()
        losses.append(loss.item())
        accuracies.append(acc.item())
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}')
    
    return model, features, adj_matrix, labels, losses, accuracies, attn1, attn2

if __name__ == "__main__":
    model, features, adj_matrix, labels, losses, accuracies, attn1, attn2 = train_gat()
    print(f"最终准确率: {accuracies[-1]:.4f}")`;

        if (this.state.cm) {
            this.state.cm.setValue(gatCode);
        }
    },

    // 运行代码
    async runCode() {
        const consoleEl = document.getElementById('console-output');
        if (consoleEl) consoleEl.textContent = '';

        if (this.state.execMode === 'pyodide') {
            await this.runCodeWithPyodide();
        } else {
            this.runCodeSimulation();
        }
    },

    // 使用Pyodide运行代码
    async runCodeWithPyodide() {
        if (!this.state.pyodideReady) {
            const consoleEl = document.getElementById('console-output');
            if (consoleEl) consoleEl.textContent = 'Pyodide 正在加载或不可用，请稍候或切换到前端模拟模式。';
            return;
        }

        const code = this.state.cm ? this.state.cm.getValue() : '';
        const consoleEl = document.getElementById('console-output');
        
        try {
            await this.state.pyodide.runPythonAsync(`
import sys
from js import console
class _C:
    def write(self,s):
        if s: console.log(s)
    def flush(self):
        pass
sys.stdout=_C()
sys.stderr=_C()
            `);
            
            await this.state.pyodide.runPythonAsync(code);
            
            if (consoleEl) consoleEl.textContent = '执行完成（Pyodide）。';
            this.renderCharts([1.0,0.8,0.6,0.5,0.45,0.4],[0.2,0.35,0.5,0.62,0.73,0.8]);
            this.renderGraph(0.8);
        } catch (e) {
            if (consoleEl) consoleEl.textContent = '执行失败（Pyodide）：' + e;
            this.renderCharts([1.0,0.8,0.6,0.5,0.45,0.4],[0.2,0.35,0.5,0.62,0.73,0.8]);
            this.renderGraph(0.8);
        }
    },

    // 模拟运行代码
    runCodeSimulation() {
        const consoleEl = document.getElementById('console-output');
        const epochs = Math.max(50, this.state.epochs);
        const lr = this.state.learningRate;
        const cap = Math.min(0.9, 0.5 + Math.log10(1 + lr*100) + (this.state.hiddenDim-16)/200);
        const losses = [], accs = [];
        const loss0 = 1.2 - Math.min(0.6, Math.log10(1 + lr*100));
        
        for (let t = 0; t < epochs; t++) {
            const progress = t/(epochs-1);
            const loss = loss0 * Math.pow(0.85 + (0.05*(16/this.state.hiddenDim)), progress*5);
            const acc = Math.min(cap, (0.2 + 0.8*progress) * cap);
            losses.push(parseFloat(loss.toFixed(4)));
            accs.push(parseFloat(acc.toFixed(4)));
            
            if (consoleEl && (t % Math.max(1, Math.floor(epochs/4)) === 0)) {
                consoleEl.textContent += 'Epoch '+t+': Loss='+losses[t].toFixed(4)+', Acc='+accs[t].toFixed(4)+'\n';
            }
        }
        
        if (consoleEl) {
            consoleEl.textContent += '训练完成\n最终准确率: '+accs[accs.length-1].toFixed(4);
        }

        this.renderCharts(losses, accs);
        this.renderGraph(accs[accs.length-1]);
    },

    // 重置代码
    resetCode() {
        const consoleEl = document.getElementById('console-output');
        if (consoleEl) consoleEl.textContent = '点击"运行代码"查看结果...';
        
        if (window.Plotly) {
            try { Plotly.purge('loss-chart'); } catch(e){}
            try { Plotly.purge('accuracy-chart'); } catch(e){}
        }
        
        const gv = document.getElementById('graph-visualization');
        if (gv) {
            gv.innerHTML = `<div class="text-center">
                <svg class="w-12 h-12 text-gray-400 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z"></path>
                </svg>
                <p>图结构可视化</p><p class="text-sm">运行代码后显示节点分类结果</p></div>`;
        }
    },

    // 保存代码
    saveCode() {
        if (this.state.cm) {
            try {
                localStorage.setItem(this.state.savedCodeKey, this.state.cm.getValue());
                alert('已保存到本地浏览器');
            } catch(e) {
                alert('保存失败：浏览器不支持或存储空间不足');
            }
        }
    },

    // 加载保存的代码
    loadSavedCode() {
        try {
            const saved = localStorage.getItem(this.state.savedCodeKey);
            if (saved && this.state.cm) {
                this.state.cm.setValue(saved);
            }
        } catch (e) {
            console.error('加载保存的代码失败:', e);
        }
    },

    // 渲染图表
    renderCharts(losses, accs) {
        if (!window.Plotly) return;
        
        Plotly.newPlot('loss-chart', [{
            y: losses, 
            type: 'scatter', 
            name: 'Loss', 
            line: {color:'#ef4444'}
        }], {
            margin: {t:10,b:30,l:40,r:10}, 
            yaxis: {title:'Loss'}
        });
        
        Plotly.newPlot('accuracy-chart', [{
            y: accs, 
            type: 'scatter', 
            name: 'Acc', 
            line: {color:'#10b981'}
        }], {
            margin: {t:10,b:30,l:40,r:10}, 
            yaxis: {title:'Accuracy', range:[0,1]}
        });
    },

    // 渲染图结构
    renderGraph(finalAcc, correctFlags, neighborsInfo) {
        const container = document.getElementById('graph-visualization');
        if (!container || !window.d3) return;
        
        container.innerHTML = '';
        const width = container.clientWidth || 400;
        const height = 220;
        const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
        
        const isGraphSAGE = neighborsInfo && Object.keys(neighborsInfo).length > 0;
        
        let nodes, links;
        
        if (isGraphSAGE) {
            nodes = d3.range(10).map(function(i){ 
                return { 
                    id: i, 
                    label: ['类别A','类别A','类别A','类别B','类别B','类别B','类别C','类别C','类别C','类别C'][i] 
                }; 
            });
            links = [
                {source:0,target:1}, {source:0,target:3}, {source:1,target:2}, {source:1,target:4},
                {source:2,target:5}, {source:3,target:4}, {source:3,target:6}, {source:4,target:5},
                {source:4,target:7}, {source:5,target:8}, {source:6,target:7}, {source:6,target:9},
                {source:7,target:8}, {source:8,target:9}
            ];
        } else {
            nodes = d3.range(7).map(function(i){ 
                return { 
                    id: i, 
                    label: ['AI','AI','CV','CV','NLP','NLP','NLP'][i] 
                }; 
            });
            links = [
                {source:0,target:1}, {source:1,target:2}, {source:2,target:3}, 
                {source:3,target:4}, {source:4,target:5}, {source:5,target:6}
            ];
        }
        
        if (Array.isArray(correctFlags) && correctFlags.length) {
            nodes.forEach(function(n, idx) { n.correct = !!correctFlags[idx]; });
        } else {
            const correctCount = Math.round(finalAcc * nodes.length);
            nodes.forEach(function(n, idx) { n.correct = idx < correctCount; });
        }
        
        const color = function(n) { return n.correct ? '#67C23A' : '#F56C6C'; };

        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(function(d){return d.id;}).distance(50))
            .force('charge', d3.forceManyBody().strength(-150))
            .force('center', d3.forceCenter(width/2, height/2));

        const link = svg.append('g').selectAll('line').data(links).enter().append('line')
            .attr('stroke', '#cbd5e1').attr('stroke-width', 1.5);

        const node = svg.append('g').selectAll('circle').data(nodes).enter().append('circle')
            .attr('r', isGraphSAGE ? 8 : 10).attr('fill', function(d){return color(d);}).attr('class', 'graph-node')
            .call(d3.drag()
                .on('start', function(event,d){ if(!event.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
                .on('drag', function(event,d){ d.fx=event.x; d.fy=event.y; })
                .on('end', function(event,d){ if(!event.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; })
            );
        
        const labels = svg.append('g').selectAll('text').data(nodes).enter().append('text')
            .text(function(d) { return d.id; })
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em')
            .attr('font-size', '10px')
            .attr('fill', 'white')
            .attr('font-weight', 'bold');
        
        node.append('title').text(function(d){ 
            let base = '节点 '+d.id+'\n预测: '+(d.correct?'正确':'错误');
            if (isGraphSAGE && neighborsInfo && neighborsInfo[d.id]) {
                base += '\n采样邻居: [' + neighborsInfo[d.id].join(',') + ']';
            }
            return base;
        });

        simulation.on('tick', function(){
            link
                .attr('x1', function(d){return d.source.x;})
                .attr('y1', function(d){return d.source.y;})
                .attr('x2', function(d){return d.target.x;})
                .attr('y2', function(d){return d.target.y;});
            node.attr('cx', function(d){return d.x;}).attr('cy', function(d){return d.y;});
            labels.attr('x', function(d){return d.x;}).attr('y', function(d){return d.y;});
        });
    },

    // 切换执行模式
    switchExecMode(mode) {
        this.state.execMode = mode;
        const consoleEl = document.getElementById('console-output');
        
        if (mode === 'pyodide' && !this.state.pyodideReady) {
            if (consoleEl) consoleEl.textContent = 'Pyodide 尚未就绪，正在加载...';
            this.initPyodide();
        } else {
            if (consoleEl) consoleEl.textContent = '已切换执行模式：' + (mode === 'pyodide' ? 'Pyodide' : '前端模拟');
        }
    },

    // 更新参数
    updateLearningRate(value) {
        const f = parseFloat(value);
        if (!isNaN(f)) this.state.learningRate = f;
        const el = document.getElementById('lr-value');
        if (el) el.textContent = String(this.state.learningRate);
    },

    updateHiddenDim(value) {
        const i = parseInt(value);
        if (!isNaN(i)) this.state.hiddenDim = i;
        const el = document.getElementById('hidden-value');
        if (el) el.textContent = String(this.state.hiddenDim);
    },

    updateEpochs(value) {
        const i = parseInt(value);
        if (!isNaN(i)) this.state.epochs = i;
        const el = document.getElementById('epochs-value');
        if (el) el.textContent = String(this.state.epochs);
    }
};

// 全局函数
window.loadPracticeEnvironment = (key) => Playground.loadPracticeEnvironment(key);
window.runCode = () => Playground.runCode();
window.resetCode = () => Playground.resetCode();
window.saveCode = () => Playground.saveCode();
window.switchExecMode = (mode) => Playground.switchExecMode(mode);
window.updateLearningRate = (value) => Playground.updateLearningRate(value);
window.updateHiddenDim = (value) => Playground.updateHiddenDim(value);
window.updateEpochs = (value) => Playground.updateEpochs(value);

// 页面加载完成后初始化
// Initialization moved to `GraphLearn.init()` in `js/core.js` to ensure a single, predictable startup sequence.
// Playground.init() will be called from there.
