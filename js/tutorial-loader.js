// 教程内容加载器
const TutorialLoader = {
    // 教程内容缓存
    cache: {},
    
    // 当前显示的教程
    currentTutorial: null,
    
    // 教程文件映射
    tutorialFiles: {
        'intro-graphs': 'pages/tutorials/intro-graphs.html',
        'social-network': 'pages/tutorials/social-network.html',
        'graph-basics': 'pages/tutorials/graph-basics.html',
        'graph-types': 'pages/tutorials/graph-types.html',
        'nn-review': 'pages/tutorials/nn-review.html',
        'why-gnn': 'pages/tutorials/why-gnn.html',
        'message-passing': 'pages/tutorials/message-passing.html',
        'gnn-framework': 'pages/tutorials/gnn-framework.html',
        'gcn': 'pages/tutorials/gcn.html',
        'gat': 'pages/tutorials/gat.html',
        'graphsage': 'pages/tutorials/graphsage.html',
        'advanced-topics': 'pages/tutorials/advanced-topics.html'
    },
    
    // 初始化
    init() {
        this.bindEvents();
    },
    
    // 绑定事件
    bindEvents() {
        // 监听教程导航点击事件
        document.addEventListener('click', (e) => {
            const tutorialLink = e.target.closest('[data-action="show-tutorial"]');
            if (tutorialLink) {
                e.preventDefault();
                const tutorialId = tutorialLink.getAttribute('data-tutorial');
                if (tutorialId) {
                    this.loadTutorial(tutorialId);
                }
            }
        });
    },
    
    // 加载教程内容
    async loadTutorial(tutorialId) {
        const container = document.getElementById('tutorial-sections');
        const defaultContent = document.getElementById('default-tutorial-content');
        
        if (!container) return;
        
        // 隐藏默认内容
        if (defaultContent) {
            defaultContent.style.display = 'none';
        }
        
        // 如果已经是当前教程，不重复加载
        if (this.currentTutorial === tutorialId) {
            return;
        }
        
        // 显示加载状态
        container.innerHTML = '<div class="text-center py-8"><div class="loading mx-auto"></div><p class="mt-4 text-gray-500">加载教程内容中...</p></div>';
        
        try {
            let content;
            
            // 检查缓存
            if (this.cache[tutorialId]) {
                content = this.cache[tutorialId];
            } else {
                // 从文件加载
                const filePath = this.tutorialFiles[tutorialId];
                if (filePath) {
                    const response = await fetch(filePath);
                    if (response.ok) {
                        content = await response.text();
                        this.cache[tutorialId] = content; // 缓存内容
                    } else {
                        throw new Error(`Failed to load tutorial: ${response.status}`);
                    }
                } else {
                    // 生成默认内容
                    content = this.generateDefaultTutorialContent(tutorialId);
                }
            }
            
            // 显示内容
            container.innerHTML = content;
            this.currentTutorial = tutorialId;
            
            // 更新导航状态
            this.updateNavigationState(tutorialId);
            
            // 执行内容中的脚本
            this.executeScripts(container);
            
            // 滚动到顶部
            container.scrollIntoView({ behavior: 'smooth', block: 'start' });
            
            // 记录学习进度
            if (window.GraphLearn && window.GraphLearn.showTutorial) {
                window.GraphLearn.showTutorial(tutorialId);
            }
            
        } catch (error) {
            console.error('Failed to load tutorial:', error);
            container.innerHTML = `
                <div class="text-center py-8">
                    <svg class="w-16 h-16 text-red-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.728-.833-2.598 0L4.616 15.5c-.77.833.192 2.5 1.732 2.5z"></path>
                    </svg>
                    <h3 class="text-lg font-medium text-red-900 mb-2">加载失败</h3>
                    <p class="text-red-700">无法加载教程内容，请稍后重试</p>
                    <button onclick="TutorialLoader.loadTutorial('${tutorialId}')" class="mt-4 bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700">
                        重试
                    </button>
                </div>
            `;
        }
    },
    
    // 生成默认教程内容
    generateDefaultTutorialContent(tutorialId) {
        const tutorialNames = {
            'intro-graphs': '什么是图？',
            'social-network': '社交网络案例',
            'graph-basics': '图的基本概念',
            'graph-types': '图的类型与表示',
            'nn-review': '神经网络回顾',
            'why-gnn': '为什么需要GNN？',
            'message-passing': '消息传递机制',
            'gnn-framework': 'GNN通用框架',
            'gcn': '图卷积网络(GCN)',
            'gat': '图注意力网络(GAT)',
            'graphsage': 'GraphSAGE',
            'advanced-topics': '高级主题'
        };
        
        const tutorialName = tutorialNames[tutorialId] || tutorialId;
        
        return `
            <div class="tutorial-content" id="${tutorialId}">
                <div class="mb-6">
                    <h1 class="text-3xl font-bold text-gray-900 mb-2">${tutorialName}</h1>
                    <p class="text-lg text-gray-600">本节内容正在完善中...</p>
                </div>

                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-6">
                    <h3 class="text-lg font-semibold text-yellow-900 mb-2">🚧 内容开发中</h3>
                    <p class="text-yellow-800">
                        这节教程的详细内容正在开发中，敬请期待！
                        您可以先浏览其他已完成的章节。
                    </p>
                </div>

                <div class="bg-blue-50 border rounded-lg p-6">
                    <h3 class="text-lg font-semibold text-blue-900 mb-3">💡 学习建议</h3>
                    <ul class="text-blue-800 space-y-2">
                        <li>• 按顺序学习前面的章节，建立坚实基础</li>
                        <li>• 动手实践，在在线编程环境中运行代码</li>
                        <li>• 参与社区讨论，与其他学习者交流</li>
                        <li>• 关注我们的更新，第一时间获取新内容</li>
                    </ul>
                </div>

                <!-- 导航按钮 -->
                <div class="flex justify-between items-center pt-6 border-t border-gray-200 mt-8">
                    <div>
                        ${this.getPreviousTutorialButton(tutorialId)}
                    </div>
                    <div class="text-center">
                        <span class="text-sm text-gray-500">${this.getTutorialPosition(tutorialId)}</span>
                    </div>
                    <div>
                        ${this.getNextTutorialButton(tutorialId)}
                    </div>
                </div>
            </div>
        `;
    },
    
    // 获取上一节按钮
    getPreviousTutorialButton(currentId) {
        const tutorialOrder = Object.keys(this.tutorialFiles);
        const currentIndex = tutorialOrder.indexOf(currentId);
        
        if (currentIndex > 0) {
            const prevId = tutorialOrder[currentIndex - 1];
            const prevName = this.getTutorialName(prevId);
            return `
                <button data-action="show-tutorial" data-tutorial="${prevId}" class="bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors">
                    ← 上一节：${prevName}
                </button>
            `;
        }
        return '';
    },
    
    // 获取下一节按钮
    getNextTutorialButton(currentId) {
        const tutorialOrder = Object.keys(this.tutorialFiles);
        const currentIndex = tutorialOrder.indexOf(currentId);
        
        if (currentIndex < tutorialOrder.length - 1) {
            const nextId = tutorialOrder[currentIndex + 1];
            const nextName = this.getTutorialName(nextId);
            return `
                <button data-action="show-tutorial" data-tutorial="${nextId}" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                    下一节：${nextName} →
                </button>
            `;
        }
        return '';
    },
    
    // 获取教程名称
    getTutorialName(tutorialId) {
        const names = {
            'intro-graphs': '什么是图？',
            'social-network': '社交网络案例',
            'graph-basics': '图的基本概念',
            'graph-types': '图的类型与表示',
            'nn-review': '神经网络回顾',
            'why-gnn': '为什么需要GNN？',
            'message-passing': '消息传递机制',
            'gnn-framework': 'GNN通用框架',
            'gcn': '图卷积网络(GCN)',
            'gat': '图注意力网络(GAT)',
            'graphsage': 'GraphSAGE',
            'advanced-topics': '高级主题'
        };
        return names[tutorialId] || tutorialId;
    },
    
    // 获取教程位置信息
    getTutorialPosition(tutorialId) {
        const tutorialOrder = Object.keys(this.tutorialFiles);
        const currentIndex = tutorialOrder.indexOf(tutorialId);
        return `第${currentIndex + 1}节 共${tutorialOrder.length}节`;
    },
    
    // 更新导航状态
    updateNavigationState(tutorialId) {
        // 移除所有活跃状态
        document.querySelectorAll('.tutorial-link').forEach(link => {
            link.classList.remove('text-blue-600', 'font-medium');
            link.classList.add('text-gray-600');
        });
        
        // 添加当前教程的活跃状态
        const activeLink = document.querySelector(`[data-tutorial="${tutorialId}"]`);
        if (activeLink) {
            activeLink.classList.remove('text-gray-600');
            activeLink.classList.add('text-blue-600', 'font-medium');
        }
    },
    
    // 执行内容中的脚本
    executeScripts(container) {
        const scripts = container.querySelectorAll('script');
        scripts.forEach(script => {
            try {
                eval(script.textContent);
            } catch (error) {
                console.error('Script execution error:', error);
            }
        });
    }
};

// 全局函数
window.TutorialLoader = TutorialLoader;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    TutorialLoader.init();
});