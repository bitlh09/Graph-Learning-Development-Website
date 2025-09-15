// 社区功能模块
(function() {
    'use strict';

    // 社区数据结构
    const STORAGE_KEYS = {
        POSTS: 'gl_community_posts',
        COMMENTS: 'gl_community_comments',
        CURRENT_USER: 'gl_current_user'
    };

    // 板块配置
    const SECTIONS = {
        discussions: { name: '学习讨论区', color: 'blue', icon: '📝' },
        projects: { name: '项目展示区', color: 'green', icon: '🚀' },
        pitfalls: { name: '算法吐槽区', color: 'red', icon: '⚠️' },
        challenges: { name: '每周挑战', color: 'purple', icon: '🏆' },
        resources: { name: '资源分享', color: 'yellow', icon: '📚' }
    };

    // 获取当前登录用户
    function getCurrentUser() {
        return localStorage.getItem(STORAGE_KEYS.CURRENT_USER);
    }

    // 检查用户是否已登录
    function isUserLoggedIn() {
        return !!getCurrentUser();
    }

    // 生成唯一ID
    function generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    // 格式化时间
    function formatTime(timestamp) {
        const now = new Date();
        const time = new Date(timestamp);
        const diff = now - time;
        
        const minute = 60 * 1000;
        const hour = minute * 60;
        const day = hour * 24;
        const week = day * 7;
        
        if (diff < minute) {
            return '刚刚';
        } else if (diff < hour) {
            return Math.floor(diff / minute) + '分钟前';
        } else if (diff < day) {
            return Math.floor(diff / hour) + '小时前';
        } else if (diff < week) {
            return Math.floor(diff / day) + '天前';
        } else {
            return time.toLocaleDateString();
        }
    }

    // 获取帖子数据
    function getPosts(section = null) {
        const posts = JSON.parse(localStorage.getItem(STORAGE_KEYS.POSTS) || '[]');
        if (section) {
            return posts.filter(post => post.section === section);
        }
        return posts;
    }

    // 保存帖子
    function savePost(post) {
        const posts = getPosts();
        posts.unshift(post);
        localStorage.setItem(STORAGE_KEYS.POSTS, JSON.stringify(posts));
        updateCommunityStats();
    }

    // 获取评论数据
    function getComments(postId) {
        const comments = JSON.parse(localStorage.getItem(STORAGE_KEYS.COMMENTS) || '[]');
        return comments.filter(comment => comment.postId === postId);
    }

    // 保存评论
    function saveComment(comment) {
        const comments = JSON.parse(localStorage.getItem(STORAGE_KEYS.COMMENTS) || '[]');
        comments.push(comment);
        localStorage.setItem(STORAGE_KEYS.COMMENTS, JSON.stringify(comments));
    }

    // 更新社区统计数据
    function updateCommunityStats() {
        const posts = getPosts();
        const comments = JSON.parse(localStorage.getItem(STORAGE_KEYS.COMMENTS) || '[]');
        
        // 更新讨论话题数
        const discussionCount = posts.filter(p => p.section === 'discussions').length;
        const discussionEl = document.querySelector('.text-green-600');
        if (discussionEl) {
            discussionEl.textContent = discussionCount;
        }
        
        // 更新项目作品数
        const projectCount = posts.filter(p => p.section === 'projects').length;
        const projectEl = document.querySelector('.text-purple-600');
        if (projectEl) {
            projectEl.textContent = projectCount;
        }
    }

    // 创建帖子发布表单
    function createPostForm(section) {
        const sectionConfig = SECTIONS[section];
        
        return `
            <div id="post-form-${section}" class="bg-white rounded-lg shadow p-6 mb-6">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-lg font-semibold text-gray-900">发布新内容 - ${sectionConfig.name}</h3>
                    <button onclick="hidePostForm('${section}')" class="text-gray-500 hover:text-gray-700">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </button>
                </div>
                
                <form onsubmit="submitPost(event, '${section}')">
                    <div class="mb-4">
                        <label for="post-title-${section}" class="block text-sm font-medium text-gray-700 mb-2">标题</label>
                        <input type="text" id="post-title-${section}" required
                               class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-${sectionConfig.color}-500"
                               placeholder="请输入标题...">
                    </div>
                    
                    <div class="mb-4">
                        <label for="post-content-${section}" class="block text-sm font-medium text-gray-700 mb-2">内容</label>
                        <textarea id="post-content-${section}" required rows="4"
                                  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-${sectionConfig.color}-500"
                                  placeholder="分享你的想法..."></textarea>
                    </div>
                    
                    <div class="mb-4">
                        <label for="post-tags-${section}" class="block text-sm font-medium text-gray-700 mb-2">标签 (可选)</label>
                        <input type="text" id="post-tags-${section}"
                               class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-${sectionConfig.color}-500"
                               placeholder="用逗号分隔多个标签，如：GCN,节点分类,PyTorch">
                    </div>
                    
                    <div class="flex justify-end space-x-3">
                        <button type="button" onclick="hidePostForm('${section}')"
                                class="px-4 py-2 text-gray-600 bg-gray-200 rounded-md hover:bg-gray-300 transition-colors">
                            取消
                        </button>
                        <button type="submit"
                                class="px-4 py-2 text-white bg-${sectionConfig.color}-600 rounded-md hover:bg-${sectionConfig.color}-700 transition-colors">
                            发布
                        </button>
                    </div>
                </form>
            </div>
        `;
    }

    // 创建帖子卡片
    function createPostCard(post) {
        const comments = getComments(post.id);
        const sectionConfig = SECTIONS[post.section];
        const tags = post.tags ? post.tags.split(',').map(tag => tag.trim()).filter(tag => tag) : [];
        
        return `
            <div class="border-l-4 border-${sectionConfig.color}-500 bg-${sectionConfig.color}-50 p-4 rounded-r-lg mb-4">
                <div class="flex justify-between items-start">
                    <div class="flex-1">
                        <h3 class="font-semibold text-gray-900 mb-2">${escapeHtml(post.title)}</h3>
                        <p class="text-gray-600 text-sm mb-3">${escapeHtml(post.content)}</p>
                        
                        ${tags.length > 0 ? `
                        <div class="flex flex-wrap gap-1 mb-2">
                            ${tags.map(tag => `
                                <span class="inline-block bg-${sectionConfig.color}-100 text-${sectionConfig.color}-800 text-xs px-2 py-1 rounded-full">
                                    ${escapeHtml(tag)}
                                </span>
                            `).join('')}
                        </div>
                        ` : ''}
                        
                        <div class="flex items-center space-x-4 text-xs text-gray-500 mb-2">
                            <span>👤 ${escapeHtml(post.author)}</span>
                            <span>💬 ${comments.length} 条评论</span>
                            <span>🕒 ${formatTime(post.timestamp)}</span>
                        </div>
                        
                        <div class="flex items-center space-x-2">
                            <button onclick="toggleComments('${post.id}')" 
                                    class="text-${sectionConfig.color}-600 hover:text-${sectionConfig.color}-800 text-sm">
                                ${comments.length > 0 ? '查看评论' : '添加评论'}
                            </button>
                        </div>
                    </div>
                </div>
                
                <!-- 评论区 -->
                <div id="comments-${post.id}" class="mt-4 hidden">
                    <div class="border-t pt-3">
                        <!-- 评论列表 -->
                        <div id="comments-list-${post.id}" class="space-y-2 mb-3">
                            ${comments.map(comment => createCommentCard(comment)).join('')}
                        </div>
                        
                        <!-- 评论表单 -->
                        ${isUserLoggedIn() ? `
                        <div class="bg-white rounded p-3 border">
                            <form onsubmit="submitComment(event, '${post.id}')">
                                <textarea id="comment-content-${post.id}" required rows="2"
                                          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-${sectionConfig.color}-500 text-sm"
                                          placeholder="写下你的评论..."></textarea>
                                <div class="flex justify-end mt-2">
                                    <button type="submit"
                                            class="px-3 py-1 text-white bg-${sectionConfig.color}-600 rounded text-sm hover:bg-${sectionConfig.color}-700 transition-colors">
                                        发表评论
                                    </button>
                                </div>
                            </form>
                        </div>
                        ` : `
                        <div class="text-center py-3 text-gray-500 text-sm">
                            <a href="#" onclick="toggleLogin()" class="text-${sectionConfig.color}-600 hover:text-${sectionConfig.color}-800">登录</a> 后参与评论
                        </div>
                        `}
                    </div>
                </div>
            </div>
        `;
    }

    // 创建评论卡片
    function createCommentCard(comment) {
        return `
            <div class="bg-gray-50 rounded p-2 text-sm">
                <div class="flex justify-between items-start mb-1">
                    <span class="font-medium text-gray-900">${escapeHtml(comment.author)}</span>
                    <span class="text-xs text-gray-500">${formatTime(comment.timestamp)}</span>
                </div>
                <p class="text-gray-700">${escapeHtml(comment.content)}</p>
            </div>
        `;
    }

    // HTML转义
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // 显示发帖表单
    window.showPostForm = function(section) {
        if (!isUserLoggedIn()) {
            alert('请先登录后再发帖');
            return;
        }
        
        const container = document.getElementById(section);
        if (container) {
            const existingForm = document.getElementById(`post-form-${section}`);
            if (existingForm) {
                existingForm.remove();
            }
            
            // 隐藏发帖按钮
            const postButton = container.querySelector('button[onclick*="showPostForm"]');
            if (postButton) {
                postButton.style.display = 'none';
            }
            
            container.insertAdjacentHTML('afterbegin', createPostForm(section));
        }
    };

    // 隐藏发帖表单
    window.hidePostForm = function(section) {
        const form = document.getElementById(`post-form-${section}`);
        if (form) {
            form.remove();
        }
        
        // 重新显示发帖按钮
        const container = document.getElementById(section);
        if (container) {
            const postButton = container.querySelector('button[onclick*="showPostForm"]');
            if (postButton) {
                postButton.style.display = '';
            }
        }
    };

    // 提交帖子
    window.submitPost = function(event, section) {
        event.preventDefault();
        
        const title = document.getElementById(`post-title-${section}`).value.trim();
        const content = document.getElementById(`post-content-${section}`).value.trim();
        const tags = document.getElementById(`post-tags-${section}`).value.trim();
        
        if (!title || !content) {
            alert('请填写标题和内容');
            return;
        }
        
        const post = {
            id: generateId(),
            title: title,
            content: content,
            tags: tags,
            author: getCurrentUser(),
            section: section,
            timestamp: Date.now()
        };
        
        savePost(post);
        hidePostForm(section);
        renderSectionPosts(section);
        
        // 确保发帖按钮重新显示
        const container = document.getElementById(section);
        if (container) {
            const postButton = container.querySelector('button[onclick*="showPostForm"]');
            if (postButton) {
                postButton.style.display = '';
            }
        }
        
        alert('发布成功！');
    };

    // 切换评论显示
    window.toggleComments = function(postId) {
        const commentsDiv = document.getElementById(`comments-${postId}`);
        if (commentsDiv) {
            commentsDiv.classList.toggle('hidden');
        }
    };

    // 提交评论
    window.submitComment = function(event, postId) {
        event.preventDefault();
        
        const content = document.getElementById(`comment-content-${postId}`).value.trim();
        
        if (!content) {
            alert('请输入评论内容');
            return;
        }
        
        const comment = {
            id: generateId(),
            postId: postId,
            content: content,
            author: getCurrentUser(),
            timestamp: Date.now()
        };
        
        saveComment(comment);
        
        // 清空表单
        document.getElementById(`comment-content-${postId}`).value = '';
        
        // 重新渲染评论列表
        const commentsList = document.getElementById(`comments-list-${postId}`);
        if (commentsList) {
            const comments = getComments(postId);
            commentsList.innerHTML = comments.map(comment => createCommentCard(comment)).join('');
        }
        
        // 更新评论计数 - 需要重新渲染整个帖子区域
        const posts = getPosts().filter(post => post.id === postId);
        if (posts.length > 0) {
            renderSectionPosts(posts[0].section);
        }
    };

    // 渲染指定板块的帖子
    function renderSectionPosts(section) {
        const container = document.getElementById(section);
        if (!container) return;
        
        const posts = getPosts(section);
        const sectionConfig = SECTIONS[section];
        
        // 查找帖子列表容器，如果不存在则创建
        let postsContainer = container.querySelector('.posts-container');
        if (!postsContainer) {
            const headerDiv = container.querySelector('.bg-white.rounded-lg.shadow');
            if (headerDiv) {
                postsContainer = document.createElement('div');
                postsContainer.className = 'posts-container space-y-4';
                headerDiv.appendChild(postsContainer);
            }
        }
        
        if (postsContainer) {
            if (posts.length === 0) {
                postsContainer.innerHTML = `
                    <div class="text-center py-8 text-gray-500">
                        <div class="text-4xl mb-2">${sectionConfig.icon}</div>
                        <p>还没有内容，来发布第一个帖子吧！</p>
                    </div>
                `;
            } else {
                postsContainer.innerHTML = posts.map(post => createPostCard(post)).join('');
            }
        }
    }

    // 初始化社区板块
    function initCommunitySection(section) {
        renderSectionPosts(section);
        updateCommunityStats();
    }

    // 重写showCommunitySection函数以支持新功能
    const originalShowCommunitySection = window.showCommunitySection;
    window.showCommunitySection = function(id) {
        // 调用原始函数
        if (originalShowCommunitySection) {
            originalShowCommunitySection(id);
        }
        
        // 初始化选中的板块
        setTimeout(() => {
            initCommunitySection(id);
        }, 100);
    };

    // 页面加载时的初始化
    document.addEventListener('DOMContentLoaded', function() {
        updateCommunityStats();
        
        // 检查是否有当前显示的板块
        const activeSections = ['discussions', 'projects', 'pitfalls', 'challenges', 'resources'];
        const visibleSection = activeSections.find(section => {
            const el = document.getElementById(section);
            return el && el.style.display !== 'none';
        });
        
        if (visibleSection) {
            initCommunitySection(visibleSection);
        }
    });

})();