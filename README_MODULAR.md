# 图学习网站 - 模块化版本

## 📁 项目结构

```
Graph-Learning-Development-Website/
├── index.html                 # 主HTML文件
├── css/
│   └── styles.css            # 所有CSS样式
├── js/
│   ├── core.js               # 核心功能模块
│   ├── navigation.js         # 导航功能模块
│   ├── playground.js         # 在线实践环境模块
│   ├── visualization.js      # 可视化模块
│   └── auth.js               # 用户认证模块
├── pages/
│   ├── home.html             # 首页内容
│   ├── tutorials.html        # 教程页面内容
│   ├── playground.html       # 在线实践页面内容
│   ├── community.html        # 社区页面内容
│   └── resources.html        # 资源页面内容
└── README_MODULAR.md         # 本文件
```

## 🚀 快速开始

### 1. 直接打开
```bash
# 双击 index.html 文件
# 或在浏览器中打开 index.html
```

### 2. 使用本地服务器（推荐）
```bash
# 使用Python
python -m http.server 8000

# 使用Node.js
npx http-server

# 使用VS Code Live Server扩展
# 右键 index.html -> Open with Live Server
```

## 📋 模块说明

### 核心模块 (core.js)
- **功能**: 页面加载、状态管理、工具函数
- **主要特性**:
  - 动态页面内容加载
  - 全局状态管理
  - 工具函数（防抖、节流、深拷贝等）
  - 本地存储管理
  - 网络请求封装

### 导航模块 (navigation.js)
- **功能**: 页面导航、路由管理
- **主要特性**:
  - 页面区域切换
  - 教程内容导航
  - 社区分区切换
  - 学习进度跟踪
  - 面包屑导航
  - 浏览器前进后退支持

### 实践环境模块 (playground.js)
- **功能**: 在线编程环境
- **主要特性**:
  - CodeMirror代码编辑器
  - Pyodide Python执行环境
  - 多种算法环境（GCN、GAT、GraphSAGE）
  - 参数调节面板
  - 代码保存/加载
  - 实时可视化

### 可视化模块 (visualization.js)
- **功能**: 图表和动画效果
- **主要特性**:
  - D3.js图结构可视化
  - Plotly.js训练曲线
  - 注意力热力图
  - 节点嵌入可视化
  - 动画效果
  - 粒子效果

### 认证模块 (auth.js)
- **功能**: 用户认证和权限管理
- **主要特性**:
  - 用户注册/登录
  - 本地存储用户数据
  - 权限检查
  - 用户状态管理
  - 消息提示系统

## 🎯 使用指南

### 开发新功能
1. **添加新页面**: 在 `pages/` 目录创建HTML文件
2. **添加新模块**: 在 `js/` 目录创建JavaScript文件
3. **添加样式**: 在 `css/styles.css` 中添加CSS规则
4. **更新导航**: 在 `js/navigation.js` 中添加路由

### 自定义样式
```css
/* 在 css/styles.css 中添加自定义样式 */
.my-custom-class {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 8px;
    padding: 16px;
}
```

### 添加新的实践环境
```javascript
// 在 js/playground.js 中添加新环境
loadNewEnvironment() {
    // 加载新环境的代码
    const newCode = `your code here`;
    if (this.state.cm) {
        this.state.cm.setValue(newCode);
    }
    this.updateEnvironmentTitle('新环境名称');
}
```

## 🔧 配置选项

### 修改默认参数
```javascript
// 在 js/playground.js 中修改默认参数
state: {
    learningRate: 0.01,    // 默认学习率
    hiddenDim: 16,         // 默认隐藏层维度
    epochs: 200,           // 默认训练轮数
    execMode: 'simulate'   // 默认执行模式
}
```

### 修改主题颜色
```css
/* 在 css/styles.css 中修改主题色 */
:root {
    --primary-color: #3b82f6;
    --secondary-color: #10b981;
    --accent-color: #8b5cf6;
}
```

## 📦 依赖管理

### 外部依赖
- **Tailwind CSS**: 样式框架
- **CodeMirror**: 代码编辑器
- **D3.js**: 数据可视化
- **Plotly.js**: 图表库
- **Pyodide**: Python运行时

### 加载顺序
```html
<!-- 在 index.html 中按顺序加载 -->
<script src="js/core.js"></script>
<script src="js/navigation.js"></script>
<script src="js/playground.js"></script>
<script src="js/visualization.js"></script>
<script src="js/auth.js"></script>
```

## 🐛 故障排除

### 常见问题

1. **页面无法加载**
   - 检查文件路径是否正确
   - 使用本地服务器而不是直接打开文件
   - 检查浏览器控制台错误

2. **Pyodide加载失败**
   - 检查网络连接
   - 切换到前端模拟模式
   - 清除浏览器缓存

3. **样式显示异常**
   - 检查Tailwind CSS是否正确加载
   - 验证CSS文件路径
   - 检查浏览器兼容性

### 调试技巧
```javascript
// 在浏览器控制台中调试
console.log('当前状态:', GraphLearn.state);
console.log('用户信息:', Auth.getUserInfo());
console.log('导航状态:', Navigation.state);
```

## 🔄 版本更新

### 从单文件版本迁移
1. 备份原始 `graph_learning_website (1).html` 文件
2. 使用新的模块化结构
3. 测试所有功能是否正常
4. 更新相关文档

### 添加新功能
1. 在相应模块中添加功能
2. 更新相关文档
3. 测试新功能
4. 提交代码

## 📞 支持

如有问题或建议，请：
1. 检查本文档
2. 查看浏览器控制台错误
3. 尝试清除浏览器缓存
4. 使用不同的浏览器测试

## 🎉 特性对比

| 特性 | 单文件版本 | 模块化版本 |
|------|------------|------------|
| 文件大小 | 181KB | 分散到多个小文件 |
| 维护性 | 困难 | 容易 |
| 加载速度 | 慢 | 快（按需加载） |
| 开发效率 | 低 | 高 |
| 代码复用 | 困难 | 容易 |
| 团队协作 | 困难 | 容易 |

模块化版本提供了更好的可维护性、更快的加载速度和更高的开发效率！
