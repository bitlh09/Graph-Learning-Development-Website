// 用户认证模块

// 显示登录弹窗
function toggleLogin() {
    var existing = document.getElementById('auth-modal');
    if (existing) { 
        existing.remove(); 
        return; 
    }
    
    var modal = document.createElement('div');
    modal.id = 'auth-modal';
    modal.className = 'fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50';
    modal.innerHTML = `
        <div class="bg-white rounded-lg w-96 p-6 shadow-xl">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-xl font-semibold text-gray-900">登录</h3>
                <button onclick="toggleLogin()" class="text-gray-400 hover:text-gray-500">
                    <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>
            <form onsubmit="authLogin(event)" class="space-y-4">
                <div>
                    <label for="auth-username" class="block text-sm font-medium text-gray-700 mb-1">用户名</label>
                    <input id="auth-username" class="w-full border rounded-md px-3 py-2 focus:ring-indigo-500 focus:border-indigo-500" placeholder="请输入用户名" required/>
                </div>
                <div>
                    <label for="auth-password" class="block text-sm font-medium text-gray-700 mb-1">密码</label>
                    <input id="auth-password" type="password" class="w-full border rounded-md px-3 py-2 focus:ring-indigo-500 focus:border-indigo-500" placeholder="请输入密码" required/>
                </div>
                <div class="flex items-center justify-between">
                    <label class="flex items-center">
                        <input type="checkbox" id="remember-me" class="form-checkbox h-4 w-4 text-indigo-600 transition duration-150 ease-in-out">
                        <span class="ml-2 text-sm text-gray-600">记住我</span>
                    </label>
                    <a href="#" class="text-sm text-indigo-600 hover:text-indigo-500">忘记密码？</a>
                </div>
                <button type="submit" class="w-full bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2">登录</button>
            </form>
            <div class="mt-4 text-center text-sm">
                <span class="text-gray-600">还没有账号？</span>
                <a href="register.html" class="text-indigo-600 hover:text-indigo-500 font-medium">立即注册</a>
            </div>
        </div>`;
    document.body.appendChild(modal);
    
    // 如果有存储的用户名和密码，自动填充
    if (localStorage.getItem('remember_me') === 'true') {
        document.getElementById('auth-username').value = localStorage.getItem('saved_username') || '';
        document.getElementById('auth-password').value = localStorage.getItem('saved_password') || '';
        document.getElementById('remember-me').checked = true;
    }
}

// 用户登录处理
function authLogin(event) {
    if (event) event.preventDefault();
    
    var u = document.getElementById('auth-username').value.trim();
    var p = document.getElementById('auth-password').value;
    var rememberMe = document.getElementById('remember-me').checked;
    
    // 表单验证
    if (!u) {
        alert('请输入用户名');
        return;
    }
    if (!p) {
        alert('请输入密码');
        return;
    }

    var store = JSON.parse(localStorage.getItem('gl_users') || '{}');
    if (store[u] && store[u] === p) {
        localStorage.setItem('gl_current_user', u);
        
        // 处理"记住我"功能
        if (rememberMe) {
            localStorage.setItem('remember_me', 'true');
            localStorage.setItem('saved_username', u);
            localStorage.setItem('saved_password', p);
        } else {
            localStorage.removeItem('remember_me');
            localStorage.removeItem('saved_username');
            localStorage.removeItem('saved_password');
        }

        toggleLogin();
        renderAuthState();
        
        // 显示成功通知
        showNotification('登录成功：' + u, 'success');
    } else {
        showNotification('用户名或密码错误', 'error');
    }
}

// 用户注册处理
function authRegister() {
    var u = (document.getElementById('auth-username') || {}).value || '';
    var p = (document.getElementById('auth-password') || {}).value || '';
    if (!u || !p) { 
        alert('请输入用户名与密码'); 
        return; 
    }
    var store = JSON.parse(localStorage.getItem('gl_users') || '{}');
    if (store[u]) { 
        alert('该用户名已存在'); 
        return; 
    }
    store[u] = p;
    localStorage.setItem('gl_users', JSON.stringify(store));
    alert('注册成功，请登录');
}

// 渲染登录状态
function renderAuthState() {
    var u = localStorage.getItem('gl_current_user');
    var btns = document.querySelectorAll('button[onclick="toggleLogin()"]');
    btns.forEach(function(btn) {
        if (u) { 
            btn.textContent = '已登录：' + u + '（退出）'; 
            btn.onclick = logout; 
        } else { 
            btn.textContent = '登录/注册'; 
            btn.onclick = toggleLogin; 
        }
    });
}

// 用户退出登录
function logout() {
    if (confirm('确定要退出登录吗？')) {
        localStorage.removeItem('gl_current_user');
        localStorage.removeItem('remember_me');
        localStorage.removeItem('saved_username');
        localStorage.removeItem('saved_password');
        renderAuthState();
        
        showNotification('已成功退出登录', 'success');
    }
}

// 通知显示功能
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    const bgColor = type === 'success' ? 'bg-green-500' : 
                   type === 'error' ? 'bg-red-500' : 'bg-blue-500';
    notification.className = `fixed top-4 right-4 ${bgColor} text-white px-6 py-3 rounded-lg shadow-lg transform transition-transform duration-300 ease-in-out z-50`;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.transform = 'translateX(150%)';
        setTimeout(() => notification.remove(), 300);
    }, 2000);
}

// 导出函数到全局作用域
window.toggleLogin = toggleLogin;
window.authLogin = authLogin;
window.authRegister = authRegister;
window.renderAuthState = renderAuthState;
window.logout = logout;
window.showNotification = showNotification;