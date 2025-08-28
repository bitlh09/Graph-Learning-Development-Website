// 用户认证模块
const Auth = {
    // 认证状态
    state: {
        currentUser: null,
        isLoggedIn: false
    },

    // 初始化认证
    init() {
        this.loadUserFromStorage();
        this.bindEvents();
        this.renderAuthState();
    },

    // 绑定事件
    bindEvents() {
        // 登录/注册按钮事件
        document.addEventListener('click', (e) => {
            if (e.target.matches('[onclick*="toggleLogin"]')) {
                e.preventDefault();
                this.toggleLogin();
            }
        });

        // 登录表单提交
        document.addEventListener('click', (e) => {
            if (e.target.matches('[onclick*="authLogin"]')) {
                e.preventDefault();
                this.login();
            }
        });

        // 注册表单提交
        document.addEventListener('click', (e) => {
            if (e.target.matches('[onclick*="authRegister"]')) {
                e.preventDefault();
                this.register();
            }
        });
    },

    // 切换登录弹窗
    toggleLogin() {
        const existing = document.getElementById('auth-modal');
        if (existing) {
            existing.remove();
            return;
        }

        const modal = document.createElement('div');
        modal.id = 'auth-modal';
        modal.className = 'fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="bg-white rounded-lg w-80 p-6 shadow-xl">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-lg font-semibold text-gray-900">登录 / 注册</h3>
                    <button onclick="Auth.toggleLogin()" class="text-gray-400 hover:text-gray-600">
                        <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                        </svg>
                    </button>
                </div>
                
                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">用户名</label>
                        <input id="auth-username" class="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="请输入用户名"/>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">密码</label>
                        <input id="auth-password" type="password" class="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="请输入密码"/>
                    </div>
                    
                    <div class="flex space-x-3">
                        <button onclick="Auth.login()" class="flex-1 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors">
                            登录
                        </button>
                        <button onclick="Auth.register()" class="flex-1 bg-gray-600 text-white px-4 py-2 rounded-md hover:bg-gray-700 transition-colors">
                            注册
                        </button>
                    </div>
                </div>
                
                <div class="mt-4 text-center">
                    <button onclick="Auth.toggleLogin()" class="text-gray-500 hover:text-gray-700">
                        关闭
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // 点击模态框外部关闭
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.toggleLogin();
            }
        });

        // 回车键提交
        const usernameInput = document.getElementById('auth-username');
        const passwordInput = document.getElementById('auth-password');
        
        if (usernameInput && passwordInput) {
            const handleEnter = (e) => {
                if (e.key === 'Enter') {
                    this.login();
                }
            };
            
            usernameInput.addEventListener('keypress', handleEnter);
            passwordInput.addEventListener('keypress', handleEnter);
        }
    },

    // 用户登录
    login() {
        const username = document.getElementById('auth-username')?.value || '';
        const password = document.getElementById('auth-password')?.value || '';

        if (!username || !password) {
            this.showMessage('请输入用户名和密码', 'error');
            return;
        }

        // 从本地存储获取用户数据
        const users = this.getUsers();
        
        if (users[username] && users[username] === password) {
            this.state.currentUser = username;
            this.state.isLoggedIn = true;
            this.saveUserToStorage();
            this.renderAuthState();
            this.showMessage(`登录成功：${username}`, 'success');
            this.toggleLogin();
        } else {
            this.showMessage('用户名或密码错误', 'error');
        }
    },

    // 用户注册
    register() {
        const username = document.getElementById('auth-username')?.value || '';
        const password = document.getElementById('auth-password')?.value || '';

        if (!username || !password) {
            this.showMessage('请输入用户名和密码', 'error');
            return;
        }

        if (username.length < 3) {
            this.showMessage('用户名至少需要3个字符', 'error');
            return;
        }

        if (password.length < 6) {
            this.showMessage('密码至少需要6个字符', 'error');
            return;
        }

        // 检查用户是否已存在
        const users = this.getUsers();
        
        if (users[username]) {
            this.showMessage('该用户名已存在', 'error');
            return;
        }

        // 保存新用户
        users[username] = password;
        this.saveUsers(users);
        
        this.showMessage('注册成功，请登录', 'success');
        
        // 清空密码字段
        const passwordInput = document.getElementById('auth-password');
        if (passwordInput) {
            passwordInput.value = '';
        }
    },

    // 用户登出
    logout() {
        this.state.currentUser = null;
        this.state.isLoggedIn = false;
        this.saveUserToStorage();
        this.renderAuthState();
        this.showMessage('已退出登录', 'info');
    },

    // 渲染认证状态
    renderAuthState() {
        const authButtons = document.querySelectorAll('[onclick*="toggleLogin"]');
        
        authButtons.forEach(button => {
            if (this.state.isLoggedIn) {
                button.textContent = `已登录：${this.state.currentUser}（退出）`;
                button.onclick = () => this.logout();
                button.className = button.className.replace('bg-white', 'bg-green-100').replace('text-blue-600', 'text-green-700');
            } else {
                button.textContent = '登录/注册';
                button.onclick = () => this.toggleLogin();
                button.className = button.className.replace('bg-green-100', 'bg-white').replace('text-green-700', 'text-blue-600');
            }
        });
    },

    // 显示消息
    showMessage(message, type = 'info') {
        // 移除现有消息
        const existingMessage = document.querySelector('.auth-message');
        if (existingMessage) {
            existingMessage.remove();
        }

        const messageEl = document.createElement('div');
        messageEl.className = `auth-message fixed top-4 right-4 z-50 px-4 py-2 rounded-md shadow-lg transition-all duration-300 transform translate-x-full`;
        
        const colors = {
            success: 'bg-green-500 text-white',
            error: 'bg-red-500 text-white',
            info: 'bg-blue-500 text-white',
            warning: 'bg-yellow-500 text-white'
        };
        
        messageEl.className += ` ${colors[type] || colors.info}`;
        messageEl.textContent = message;

        document.body.appendChild(messageEl);

        // 动画显示
        setTimeout(() => {
            messageEl.classList.remove('translate-x-full');
        }, 100);

        // 自动隐藏
        setTimeout(() => {
            messageEl.classList.add('translate-x-full');
            setTimeout(() => {
                if (messageEl.parentNode) {
                    messageEl.parentNode.removeChild(messageEl);
                }
            }, 300);
        }, 3000);
    },

    // 获取用户数据
    getUsers() {
        try {
            const users = localStorage.getItem('gl_users');
            return users ? JSON.parse(users) : {};
        } catch (e) {
            console.error('读取用户数据失败:', e);
            return {};
        }
    },

    // 保存用户数据
    saveUsers(users) {
        try {
            localStorage.setItem('gl_users', JSON.stringify(users));
        } catch (e) {
            console.error('保存用户数据失败:', e);
            this.showMessage('保存失败：存储空间不足', 'error');
        }
    },

    // 从存储加载用户状态
    loadUserFromStorage() {
        try {
            const currentUser = localStorage.getItem('gl_current_user');
            if (currentUser) {
                this.state.currentUser = currentUser;
                this.state.isLoggedIn = true;
            }
        } catch (e) {
            console.error('加载用户状态失败:', e);
        }
    },

    // 保存用户状态到存储
    saveUserToStorage() {
        try {
            if (this.state.isLoggedIn && this.state.currentUser) {
                localStorage.setItem('gl_current_user', this.state.currentUser);
            } else {
                localStorage.removeItem('gl_current_user');
            }
        } catch (e) {
            console.error('保存用户状态失败:', e);
        }
    },

    // 检查用户权限
    hasPermission(permission) {
        if (!this.state.isLoggedIn) return false;
        
        // 简单的权限检查逻辑
        const permissions = {
            'save_code': true,
            'load_code': true,
            'share_project': true,
            'access_advanced': this.state.currentUser === 'admin'
        };
        
        return permissions[permission] || false;
    },

    // 获取用户信息
    getUserInfo() {
        if (!this.state.isLoggedIn) return null;
        
        return {
            username: this.state.currentUser,
            isLoggedIn: this.state.isLoggedIn,
            joinDate: this.getUserJoinDate(),
            lastLogin: this.getUserLastLogin()
        };
    },

    // 获取用户加入日期
    getUserJoinDate() {
        try {
            const joinDate = localStorage.getItem(`gl_user_${this.state.currentUser}_join_date`);
            return joinDate || new Date().toISOString();
        } catch (e) {
            return new Date().toISOString();
        }
    },

    // 获取用户最后登录时间
    getUserLastLogin() {
        try {
            const lastLogin = localStorage.getItem(`gl_user_${this.state.currentUser}_last_login`);
            return lastLogin || new Date().toISOString();
        } catch (e) {
            return new Date().toISOString();
        }
    },

    // 更新最后登录时间
    updateLastLogin() {
        if (this.state.isLoggedIn) {
            try {
                localStorage.setItem(`gl_user_${this.state.currentUser}_last_login`, new Date().toISOString());
            } catch (e) {
                console.error('更新最后登录时间失败:', e);
            }
        }
    },

    // 清除所有用户数据（用于测试）
    clearAllUserData() {
        try {
            localStorage.removeItem('gl_users');
            localStorage.removeItem('gl_current_user');
            this.state.currentUser = null;
            this.state.isLoggedIn = false;
            this.renderAuthState();
            this.showMessage('所有用户数据已清除', 'info');
        } catch (e) {
            console.error('清除用户数据失败:', e);
        }
    }
};

// 全局认证函数
window.toggleLogin = () => Auth.toggleLogin();
window.authLogin = () => Auth.login();
window.authRegister = () => Auth.register();
window.logout = () => Auth.logout();
window.renderAuthState = () => Auth.renderAuthState();

// 页面加载完成后初始化认证
// Initialization moved to `GraphLearn.init()` in `js/core.js` to ensure a single, predictable startup sequence.
// Auth.init() will be called from there.
