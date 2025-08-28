// 核心功能模块
const GraphLearn = {
    state: {
        currentSection: 'home',
        currentTutorial: null,
        currentCommunitySection: 'discussions',
        user: null,
        learningProgress: {
            tutorials: {},
            practice: {},
            achievements: [],
            totalTime: 0,
            lastActive: null
        }
    },

    showInitError(moduleName, error) {
        try {
            const id = `init-error-${moduleName}`;
            if (document.getElementById(id)) return;

            const el = document.createElement('div');
            el.id = id;
            el.className = 'fixed top-4 right-4 bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded shadow-lg z-60';
            el.style.maxWidth = '320px';
            el.innerHTML = `
                <div class="flex items-start">
                    <div class="flex-1">
                        <p style="font-weight:600; margin:0">模块初始化失败: ${moduleName}</p>
                        <p style="margin:4px 0 0; font-size:12px; color: #6b7280">${String(error).slice(0, 200)}</p>
                    </div>
                    <button aria-label="关闭" style="margin-left:8px; background:none; border:none; cursor:pointer; color:#7f1d1d; font-weight:700" onclick="this.parentNode && this.parentNode.parentNode && this.parentNode.parentNode.removeChild(this.parentNode.parentNode)">×</button>
                </div>
            `;

            document.body.appendChild(el);

            setTimeout(() => {
                if (el.parentNode) el.parentNode.removeChild(el);
            }, 8000);
        } catch (e) {
            console.error('showInitError failed', e);
        }
    },
    pageCache: {},
    
    init() {
    this.loadUserProgress();
    this.bindEvents();
    // 初始化各模块（统一入口），每个模块初始化包裹 try/catch 并在失败时提示
    try {
        if (window.Navigation && typeof Navigation.init === 'function') {
            Navigation.init();
        }
    } catch (e) {
        this.showInitError('Navigation', e);
        console.error('Navigation.init failed:', e);
    }

    try {
        if (window.Playground && typeof Playground.init === 'function') {
            Playground.init();
        }
    } catch (e) {
        this.showInitError('Playground', e);
        console.error('Playground.init failed:', e);
    }

    try {
        if (window.Visualization && typeof Visualization.init === 'function') {
            Visualization.init();
        }
    } catch (e) {
        this.showInitError('Visualization', e);
        console.error('Visualization.init failed:', e);
    }

    try {
        if (window.Auth && typeof Auth.init === 'function') {
            Auth.init();
        }
    } catch (e) {
        this.showInitError('Auth', e);
        console.error('Auth.init failed:', e);
    }

    // 初始化导航状态并加载首页内容
    try {
        this.updateNavigationState('home');
        this.loadPageContent('home', 'home-content');
    } catch (e) {
        this.showInitError('PageLoader', e);
        console.error('Page load/init failed:', e);
    }

    // 初始化进度追踪与面板
    try {
        this.startProgressTracking();
        this.renderLearningDashboard();
    } catch (e) {
        this.showInitError('Progress', e);
        console.error('Progress init failed:', e);
    }

    // 默认展示社区第一个分区
    try {
        this.showCommunitySection('discussions');
    } catch (e) {
        this.showInitError('Community', e);
        console.error('Community init failed:', e);
    }
    },
    
    bindEvents() {
        // 页面可见性变化时暂停/恢复计时
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseProgressTracking();
            } else {
                this.resumeProgressTracking();
            }
        });
        
        // 定期保存进度
        setInterval(() => {
            this.saveUserProgress();
        }, 30000); // 每30秒保存一次
    },
    
    showSection(sectionName) {
        this.state.currentSection = sectionName;
        this.loadPageContent(sectionName, sectionName + '-content');
        this.updateNavigationState(sectionName);
        
        // 记录学习活动
        this.recordActivity('section_visit', { section: sectionName });
    },
    
    showTutorial(tutorialName) {
        this.state.currentTutorial = tutorialName;
        this.updateTutorialNavigationState(tutorialName);
        
        // 记录教程学习进度
        if (!this.state.learningProgress.tutorials[tutorialName]) {
            this.state.learningProgress.tutorials[tutorialName] = {
                started: new Date().toISOString(),
                completed: false,
                timeSpent: 0,
                exercises: {},
                lastVisit: new Date().toISOString()
            };
        } else {
            this.state.learningProgress.tutorials[tutorialName].lastVisit = new Date().toISOString();
        }
        
        this.recordActivity('tutorial_visit', { tutorial: tutorialName });
        this.updateProgressDisplay();
    },
    
    showCommunitySection(sectionName) {
        this.state.currentCommunitySection = sectionName;
        this.updateCommunityNavigationState(sectionName);
    },
    
    loadPageContent(sectionName, contentId) {
        const contentElement = document.getElementById(contentId);
        if (!contentElement) return;
        
        // 检查缓存
        if (this.pageCache[sectionName]) {
            contentElement.innerHTML = this.pageCache[sectionName];
            return;
        }
        
        // 动态加载页面内容
        const pagePath = `pages/${sectionName}.html`;
        fetch(pagePath)
            .then(response => response.text())
            .then(html => {
                contentElement.innerHTML = html;
                this.pageCache[sectionName] = html;
                
                // 如果是教程页面，初始化进度显示
                if (sectionName === 'tutorials') {
                    this.initializeTutorialProgress();
                }
            })
            .catch(error => {
                console.error('Failed to load page content:', error);
                contentElement.innerHTML = '<div class="text-center py-8"><p class="text-gray-500">页面加载失败</p></div>';
            });
    },
    
    updateNavigationState(sectionName) {
        // 更新导航栏状态
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('bg-blue-700');
            if (link.getAttribute('onclick')?.includes(sectionName)) {
                link.classList.add('bg-blue-700');
            }
        });
    },
    
    updateTutorialNavigationState(tutorialName) {
        // 更新教程导航状态
        document.querySelectorAll('.tutorial-link').forEach(link => {
            link.classList.remove('text-blue-600', 'font-medium');
            if (link.getAttribute('onclick')?.includes(tutorialName)) {
                link.classList.add('text-blue-600', 'font-medium');
            }
        });
    },
    
    updateCommunityNavigationState(sectionName) {
        // 更新社区导航状态
        document.querySelectorAll('.community-nav-link').forEach(link => {
            link.classList.remove('bg-blue-100', 'text-blue-700');
            if (link.getAttribute('onclick')?.includes(sectionName)) {
                link.classList.add('bg-blue-100', 'text-blue-700');
            }
        });
    },
    
    // 学习进度管理
    recordActivity(type, data) {
        const activity = {
            type,
            data,
            timestamp: new Date().toISOString()
        };
        
        // 记录到学习历史
        if (!this.state.learningProgress.activities) {
            this.state.learningProgress.activities = [];
        }
        this.state.learningProgress.activities.push(activity);
        
        // 限制历史记录数量
        if (this.state.learningProgress.activities.length > 100) {
            this.state.learningProgress.activities = this.state.learningProgress.activities.slice(-100);
        }
        
        this.checkAchievements(activity);
    },
    
    startProgressTracking() {
        this.state.learningProgress.lastActive = new Date().toISOString();
        this.progressTimer = setInterval(() => {
            if (!document.hidden) {
                this.state.learningProgress.totalTime += 1;
                this.updateProgressDisplay();
            }
        }, 1000);
    },
    
    pauseProgressTracking() {
        if (this.progressTimer) {
            clearInterval(this.progressTimer);
        }
    },
    
    resumeProgressTracking() {
        this.state.learningProgress.lastActive = new Date().toISOString();
        this.startProgressTracking();
    },
    
    updateProgressDisplay() {
        // 更新总体进度条
        const totalProgress = this.calculateTotalProgress();
        const progressBar = document.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.style.setProperty('--progress', totalProgress + '%');
        }
        
        // 更新进度文本
        const progressText = document.querySelector('.progress-text');
        if (progressText) {
            progressText.textContent = `已完成 ${totalProgress.toFixed(1)}%`;
        }
        
        // 更新学习时间
        const timeSpent = this.formatTime(this.state.learningProgress.totalTime);
        const timeElement = document.querySelector('.time-spent');
        if (timeElement) {
            timeElement.textContent = `学习时间: ${timeSpent}`;
        }
        
        // 更新成就显示
        this.updateAchievementsDisplay();
    },
    
    calculateTotalProgress() {
        const tutorials = Object.keys(this.state.learningProgress.tutorials);
        if (tutorials.length === 0) return 0;
        
        const completedTutorials = tutorials.filter(tutorial => 
            this.state.learningProgress.tutorials[tutorial].completed
        );
        
        const tutorialProgress = (completedTutorials.length / tutorials.length) * 70; // 教程占70%
        const practiceProgress = this.calculatePracticeProgress() * 30; // 实践占30%
        
        return Math.min(100, tutorialProgress + practiceProgress);
    },
    
    calculatePracticeProgress() {
        const practiceSessions = Object.keys(this.state.learningProgress.practice);
        if (practiceSessions.length === 0) return 0;
        
        const totalSessions = practiceSessions.length;
        const completedSessions = practiceSessions.filter(session => 
            this.state.learningProgress.practice[session].completed
        );
        
        return completedSessions.length / totalSessions;
    },
    
    formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        
        if (hours > 0) {
            return `${hours}小时${minutes}分钟`;
        } else {
            return `${minutes}分钟`;
        }
    },
    
    checkAchievements(activity) {
        const achievements = [
            {
                id: 'first_tutorial',
                name: '初学者的第一步',
                description: '完成第一个教程',
                condition: () => Object.keys(this.state.learningProgress.tutorials).length >= 1
            },
            {
                id: 'practice_master',
                name: '实践达人',
                description: '完成5次在线实践',
                condition: () => Object.keys(this.state.learningProgress.practice).length >= 5
            },
            {
                id: 'time_dedication',
                name: '专注学习者',
                description: '累计学习时间超过1小时',
                condition: () => this.state.learningProgress.totalTime >= 3600
            },
            {
                id: 'gcn_expert',
                name: 'GCN专家',
                description: '完成GCN相关教程和实践',
                condition: () => {
                    const gcnTutorials = ['gcn', 'message-passing', 'gnn-framework'];
                    return gcnTutorials.every(tutorial => 
                        this.state.learningProgress.tutorials[tutorial]?.completed
                    );
                }
            },
            {
                id: 'gat_master',
                name: '注意力大师',
                description: '完成GAT相关教程和实践',
                condition: () => {
                    const gatTutorials = ['gat', 'message-passing'];
                    return gatTutorials.every(tutorial => 
                        this.state.learningProgress.tutorials[tutorial]?.completed
                    );
                }
            },
            {
                id: 'graphsage_expert',
                name: 'GraphSAGE专家',
                description: '完成GraphSAGE相关教程和实践',
                condition: () => {
                    return this.state.learningProgress.tutorials['graphsage']?.completed;
                }
            }
        ];
        
        achievements.forEach(achievement => {
            if (!this.state.learningProgress.achievements.includes(achievement.id) && 
                achievement.condition()) {
                this.unlockAchievement(achievement);
            }
        });
    },
    
    unlockAchievement(achievement) {
        this.state.learningProgress.achievements.push(achievement.id);
        this.showAchievementNotification(achievement);
        this.saveUserProgress();
    },
    
    showAchievementNotification(achievement) {
        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 rounded shadow-lg z-50 transform transition-all duration-500 translate-x-full';
        notification.innerHTML = `
            <div class="flex items-center">
                <div class="flex-shrink-0">
                    <svg class="h-5 w-5 text-yellow-500" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                    </svg>
                </div>
                <div class="ml-3">
                    <p class="text-sm font-medium">🏆 解锁成就: ${achievement.name}</p>
                    <p class="text-sm">${achievement.description}</p>
                </div>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // 动画显示
        setTimeout(() => {
            notification.classList.remove('translate-x-full');
        }, 100);
        
        // 自动隐藏
        setTimeout(() => {
            notification.classList.add('translate-x-full');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 500);
        }, 5000);
    },
    
    updateAchievementsDisplay() {
        const achievementsContainer = document.querySelector('.achievements-container');
        if (!achievementsContainer) return;
        
        const achievements = [
            { id: 'first_tutorial', name: '初学者的第一步', icon: '🎯' },
            { id: 'practice_master', name: '实践达人', icon: '💻' },
            { id: 'time_dedication', name: '专注学习者', icon: '⏰' },
            { id: 'gcn_expert', name: 'GCN专家', icon: '🔗' },
            { id: 'gat_master', name: '注意力大师', icon: '👁️' },
            { id: 'graphsage_expert', name: 'GraphSAGE专家', icon: '📊' }
        ];
        
        achievementsContainer.innerHTML = achievements.map(achievement => {
            const unlocked = this.state.learningProgress.achievements.includes(achievement.id);
            return `
                <div class="flex items-center space-x-2 ${unlocked ? 'text-yellow-600' : 'text-gray-400'}">
                    <span class="text-lg">${achievement.icon}</span>
                    <span class="text-sm">${achievement.name}</span>
                    ${unlocked ? '<span class="text-yellow-500">✓</span>' : ''}
                </div>
            `;
        }).join('');
    },
    
    renderLearningDashboard() {
        const dashboardContainer = document.querySelector('.learning-dashboard');
        if (!dashboardContainer) return;
        
        const totalProgress = this.calculateTotalProgress();
        const timeSpent = this.formatTime(this.state.learningProgress.totalTime);
        const achievementsCount = this.state.learningProgress.achievements.length;
        
        dashboardContainer.innerHTML = `
            <div class="bg-white rounded-lg shadow p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">学习进度</h3>
                
                <!-- 总体进度 -->
                <div class="mb-4">
                    <div class="flex justify-between items-center mb-2">
                        <span class="text-sm font-medium text-gray-700">总体进度</span>
                        <span class="text-sm text-gray-500">${totalProgress.toFixed(1)}%</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2">
                        <div class="progress-bar h-2 rounded-full transition-all duration-300" 
                             style="--progress: ${totalProgress}%"></div>
                    </div>
                </div>
                
                <!-- 学习统计 -->
                <div class="grid grid-cols-3 gap-4 mb-4">
                    <div class="text-center">
                        <div class="text-2xl font-bold text-blue-600">${Object.keys(this.state.learningProgress.tutorials).length}</div>
                        <div class="text-xs text-gray-500">已学习教程</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold text-green-600">${Object.keys(this.state.learningProgress.practice).length}</div>
                        <div class="text-xs text-gray-500">实践次数</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold text-yellow-600">${achievementsCount}</div>
                        <div class="text-xs text-gray-500">获得成就</div>
                    </div>
                </div>
                
                <!-- 学习时间 -->
                <div class="text-center text-sm text-gray-600 mb-4">
                    <span class="time-spent">学习时间: ${timeSpent}</span>
                </div>
                
                <!-- 成就展示 -->
                <div class="achievements-container space-y-2"></div>
            </div>
        `;
        
        this.updateAchievementsDisplay();
    },
    
    initializeTutorialProgress() {
        // 为每个教程添加进度指示器
        const tutorialLinks = document.querySelectorAll('.tutorial-link');
        tutorialLinks.forEach(link => {
            const tutorialName = link.getAttribute('onclick')?.match(/showTutorial\('([^']+)'\)/)?.[1];
            if (tutorialName) {
                const progress = this.state.learningProgress.tutorials[tutorialName];
                if (progress) {
                    // 添加进度指示器
                    const indicator = document.createElement('span');
                    indicator.className = 'ml-2 inline-block w-2 h-2 rounded-full';
                    indicator.style.backgroundColor = progress.completed ? '#10b981' : '#f59e0b';
                    link.appendChild(indicator);
                }
            }
        });
    },
    
    // 数据持久化
    saveUserProgress() {
        try {
            localStorage.setItem('graphlearn_progress', JSON.stringify(this.state.learningProgress));
        } catch (error) {
            console.error('Failed to save progress:', error);
        }
    },
    
    loadUserProgress() {
        try {
            const saved = localStorage.getItem('graphlearn_progress');
            if (saved) {
                this.state.learningProgress = { ...this.state.learningProgress, ...JSON.parse(saved) };
            }
        } catch (error) {
            console.error('Failed to load progress:', error);
        }
    },
    
    // 工具函数
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },
    
    deepClone(obj) {
        return JSON.parse(JSON.stringify(obj));
    },
    
    // 存储管理
    storage: {
        get(key) {
            try {
                return JSON.parse(localStorage.getItem(key));
            } catch (error) {
                return null;
            }
        },
        
        set(key, value) {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (error) {
                console.error('Storage set error:', error);
                return false;
            }
        },
        
        remove(key) {
            try {
                localStorage.removeItem(key);
                return true;
            } catch (error) {
                console.error('Storage remove error:', error);
                return false;
            }
        }
    },
    
    // API请求封装
    api: {
        async request(url, options = {}) {
            try {
                const response = await fetch(url, {
                    headers: {
                        'Content-Type': 'application/json',
                        ...options.headers
                    },
                    ...options
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                return await response.json();
            } catch (error) {
                console.error('API request failed:', error);
                throw error;
            }
        }
    },
    
    // 通知系统
    notify: {
        show(message, type = 'info', duration = 3000) {
            const notification = document.createElement('div');
            const colors = {
                info: 'bg-blue-100 border-blue-500 text-blue-700',
                success: 'bg-green-100 border-green-500 text-green-700',
                warning: 'bg-yellow-100 border-yellow-500 text-yellow-700',
                error: 'bg-red-100 border-red-500 text-red-700'
            };
            
            notification.className = `fixed top-4 right-4 border-l-4 p-4 rounded shadow-lg z-50 transform transition-all duration-500 translate-x-full ${colors[type]}`;
            notification.textContent = message;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.classList.remove('translate-x-full');
            }, 100);
            
            setTimeout(() => {
                notification.classList.add('translate-x-full');
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.parentNode.removeChild(notification);
                    }
                }, 500);
            }, duration);
        }
    },
    
    // 加载状态管理
    loading: {
        show(message = '加载中...') {
            const loading = document.getElementById('loading');
            if (loading) {
                loading.classList.remove('hidden');
                const text = loading.querySelector('p');
                if (text) text.textContent = message;
            }
        },
        
        hide() {
            const loading = document.getElementById('loading');
            if (loading) {
                loading.classList.add('hidden');
            }
        }
    }
};

// 全局函数导出
window.showSection = (sectionName) => GraphLearn.showSection(sectionName);
window.showTutorial = (tutorialName) => GraphLearn.showTutorial(tutorialName);
window.showCommunitySection = (sectionName) => GraphLearn.showCommunitySection(sectionName);

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    GraphLearn.init();
});
