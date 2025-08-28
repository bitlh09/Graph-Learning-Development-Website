// 导航功能模块
const Navigation = {
    // 导航状态
    state: {
        currentSection: 'home',
        currentTutorial: 'intro-graphs',
        currentCommunitySection: 'discussions',
        tutorialProgress: 35
    },

    // 初始化导航
    init() {
        this.bindNavigationEvents();
        this.updateActiveNav();
        this.updateProgress();
    },

    // 绑定导航事件（事件委托，优先使用 data-*，保留 href/onclick 兼容性）
    bindNavigationEvents() {
        document.addEventListener('click', (e) => {
            const el = e.target.closest('[data-action], .nav-link, .tutorial-link, .community-nav-link');
            if (!el) return;

            // 如果是可交互元素，阻止默认导航
            if (el.tagName === 'A' || el.tagName === 'BUTTON' || el.hasAttribute('data-action')) {
                e.preventDefault();
            }

            const action = el.dataset.action;
            if (action === 'show-section') {
                const target = el.dataset.target || el.getAttribute('href')?.replace('#', '');
                if (target) this.navigateToSection(target);
                return;
            }

            if (action === 'show-tutorial') {
                const tutorial = el.dataset.tutorial || el.getAttribute('data-tutorial');
                if (tutorial) this.navigateToTutorial(tutorial);
                return;
            }

            if (action === 'show-member') {
                const member = el.dataset.member;
                if (member) {
                    // 切换到 group 区域再加载成员页面片段
                    if (window.GraphLearn && typeof GraphLearn.showSection === 'function') {
                        GraphLearn.showSection('group');
                        // 加载 member 页面（pages/memberX.html）到 group-content
                        if (typeof GraphLearn.loadPageContent === 'function') {
                            GraphLearn.loadPageContent(member, 'group-content');
                        } else {
                            // 后备：直接导航到成员页
                            window.location.href = el.getAttribute('href') || `pages/${member}.html`;
                        }
                    } else {
                        window.location.href = el.getAttribute('href') || `pages/${member}.html`;
                    }
                }
                return;
            }

            if (action === 'show-community') {
                const target = el.dataset.target || el.getAttribute('data-target');
                if (target) this.navigateToCommunitySection(target);
                return;
            }

            // 兼容旧的 href / inline onclick 形式
            if (el.matches('.nav-link')) {
                const section = el.getAttribute('href')?.replace('#', '') ||
                                el.getAttribute('onclick')?.match(/showSection\('([^']+)'\)/)?.[1];
                if (section) this.navigateToSection(section);
            } else if (el.matches('.tutorial-link')) {
                const tutorial = el.dataset.tutorial || el.getAttribute('onclick')?.match(/showTutorial\('([^']+)'\)/)?.[1];
                if (tutorial) this.navigateToTutorial(tutorial);
            } else if (el.matches('.community-nav-link')) {
                const section = el.dataset.target || el.getAttribute('onclick')?.match(/showCommunitySection\('([^']+)'\)/)?.[1];
                if (section) this.navigateToCommunitySection(section);
            }
        });
    },

    // 导航到指定区域
    navigateToSection(sectionId) {
        // 隐藏所有区域
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });

        // 显示目标区域
        const targetSection = document.getElementById(sectionId + '-section');
        if (targetSection) {
            targetSection.classList.add('active');
            this.state.currentSection = sectionId;
            this.updateActiveNav();
            
            // 更新URL（如果支持）
            if (window.history && window.history.pushState) {
                window.history.pushState({ section: sectionId }, '', `#${sectionId}`);
            }
        }
    },

    // 导航到指定教程
    navigateToTutorial(tutorialId) {
        // 隐藏所有教程内容
        document.querySelectorAll('.tutorial-content').forEach(content => {
            content.style.display = 'none';
        });

        // 显示目标教程
        const targetTutorial = document.getElementById(tutorialId);
        if (targetTutorial) {
            targetTutorial.style.display = 'block';
            this.state.currentTutorial = tutorialId;
            this.updateTutorialProgress();
        }
    },

    // 导航到社区分区
    navigateToCommunitySection(sectionId) {
        const sections = ['discussions', 'projects', 'pitfalls', 'challenges', 'resources'];
        sections.forEach(section => {
            const el = document.getElementById(section);
            if (el) {
                el.style.display = (section === sectionId) ? 'block' : 'none';
            }
        });
        this.state.currentCommunitySection = sectionId;
        this.updateCommunityNav();
    },

    // 更新活跃导航状态
    updateActiveNav() {
        // 移除所有活跃状态
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });

        // 优先使用 data-* 标记查找活跃链接，回退到 href/onclick
        let activeLink = document.querySelector(`[data-action="show-section"][data-target="${this.state.currentSection}"]`);
        if (!activeLink) {
            activeLink = document.querySelector(`.nav-link[href="#${this.state.currentSection}"]`) ||
                         document.querySelector(`[onclick*="showSection('${this.state.currentSection}')"]`);
        }
        if (activeLink) activeLink.classList.add('active');
    },

    // 更新教程导航状态
    updateTutorialNav() {
        // 移除所有活跃状态
        document.querySelectorAll('.tutorial-link').forEach(link => {
            link.classList.remove('active');
        });

        // 优先使用 data-tutorial 查找活跃教程链接
        let activeLink = document.querySelector(`[data-action="show-tutorial"][data-tutorial="${this.state.currentTutorial}"]`);
        if (!activeLink) {
            activeLink = document.querySelector(`[data-tutorial="${this.state.currentTutorial}"]`) ||
                         document.querySelector(`[onclick*="showTutorial('${this.state.currentTutorial}')"]`);
        }
        if (activeLink) activeLink.classList.add('active');
    },

    // 更新社区导航状态
    updateCommunityNav() {
        // 移除所有活跃状态
        document.querySelectorAll('.community-nav-link').forEach(link => {
            link.classList.remove('active');
        });

        // 优先使用 data-target 查找社区导航活跃项
        let activeLink = document.querySelector(`[data-action="show-community"][data-target="${this.state.currentCommunitySection}"]`);
        if (!activeLink) {
            activeLink = document.querySelector(`[data-target="${this.state.currentCommunitySection}"]`) ||
                         document.querySelector(`[onclick*="showCommunitySection('${this.state.currentCommunitySection}')"]`);
        }
        if (activeLink) activeLink.classList.add('active');
    },

    // 更新学习进度
    updateProgress() {
        const progressBar = document.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.style.setProperty('--progress', `${this.state.tutorialProgress}%`);
        }

        const progressText = document.querySelector('.progress-text');
        if (progressText) {
            progressText.textContent = `已完成 ${this.state.tutorialProgress}%`;
        }
    },

    // 更新教程进度
    updateTutorialProgress() {
        // 根据当前教程计算进度
        const tutorialOrder = [
            'intro-graphs', 'social-network', 'graph-basics', 'graph-types',
            'nn-review', 'why-gnn', 'message-passing', 'gnn-framework',
            'gcn', 'gat', 'graphsage', 'advanced-topics'
        ];
        
        const currentIndex = tutorialOrder.indexOf(this.state.currentTutorial);
        if (currentIndex !== -1) {
            this.state.tutorialProgress = Math.round(((currentIndex + 1) / tutorialOrder.length) * 100);
            this.updateProgress();
        }
    },

    // 面包屑导航
    updateBreadcrumb() {
        const breadcrumbContainer = document.getElementById('breadcrumb');
        if (!breadcrumbContainer) return;

        const breadcrumbs = [];
        
        // 添加主区域
        const sectionNames = {
            'home': '首页',
            'tutorials': '教程',
            'playground': '在线实践',
            'community': '社区',
            'resources': '资源'
        };
        
        breadcrumbs.push({
            name: sectionNames[this.state.currentSection] || this.state.currentSection,
            active: true
        });

        // 如果是教程页面，添加教程名称
        if (this.state.currentSection === 'tutorials') {
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
            
            breadcrumbs.push({
                name: tutorialNames[this.state.currentTutorial] || this.state.currentTutorial,
                active: true
            });
        }

        // 渲染面包屑
        breadcrumbContainer.innerHTML = breadcrumbs.map((crumb, index) => {
            return `<span class="text-gray-500">${crumb.name}</span>${index < breadcrumbs.length - 1 ? ' / ' : ''}`;
        }).join('');
    },

    // 返回上一页
    goBack() {
        if (window.history && window.history.length > 1) {
            window.history.back();
        } else {
            this.navigateToSection('home');
        }
    },

    // 前进
    goForward() {
        if (window.history && window.history.length > 1) {
            window.history.forward();
        }
    },

    // 处理浏览器前进后退
    handlePopState(event) {
        if (event.state && event.state.section) {
            this.navigateToSection(event.state.section);
        }
    },

    // 滚动到顶部
    scrollToTop() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    },

    // 滚动到指定元素
    scrollToElement(elementId, offset = 0) {
        const element = document.getElementById(elementId);
        if (element) {
            const elementPosition = element.offsetTop - offset;
            window.scrollTo({
                top: elementPosition,
                behavior: 'smooth'
            });
        }
    },

    // 平滑滚动到锚点
    scrollToAnchor(anchor) {
        const element = document.querySelector(anchor);
        if (element) {
            element.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }
};

// 全局导航函数
window.navigateToSection = (sectionId) => Navigation.navigateToSection(sectionId);
window.navigateToTutorial = (tutorialId) => Navigation.navigateToTutorial(tutorialId);
window.navigateToCommunitySection = (sectionId) => Navigation.navigateToCommunitySection(sectionId);
window.goBack = () => Navigation.goBack();
window.goForward = () => Navigation.goForward();
window.scrollToTop = () => Navigation.scrollToTop();
window.scrollToElement = (elementId, offset) => Navigation.scrollToElement(elementId, offset);
window.scrollToAnchor = (anchor) => Navigation.scrollToAnchor(anchor);

// 监听浏览器前进后退
window.addEventListener('popstate', (event) => {
    Navigation.handlePopState(event);
});

// 页面加载完成后初始化导航
// Initialization moved to `GraphLearn.init()` in `js/core.js` to ensure a single, predictable startup sequence.
// Navigation.init() will be called from there.
