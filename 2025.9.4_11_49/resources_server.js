const express = require('express');
const path = require('path');
const fs = require('fs').promises;

const app = express();
const PORT = 3000;

// 以当前目录（resources/）为根
const ROOT = __dirname;
// 仅暴露 downloads 子目录为静态资源路径 /resources
const DOWNLOADS_DIR = path.join(ROOT, 'downloads');

// 首页（可用你自己的 index.html）
app.get('/', (req, res) => {
  res.sendFile(path.join(ROOT, 'index.html'));
});

// 提供所有静态文件
app.use(express.static(ROOT, {
  index: false,
  dotfiles: 'ignore',
}));

// 只开放 downloads 子目录，避免暴露 server.js / package.json
app.use('/resources', express.static(DOWNLOADS_DIR, {
  index: false,
  dotfiles: 'ignore',
}));

// 需要显示大小的文件名（与前端 href 保持一致）
const FILE_LIST = [
  'PDF教程合集.zip',
  '代码示例.zip',
];

// 字节数转人类可读
function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const num = bytes / Math.pow(k, i);
  return `${num % 1 === 0 ? num : num.toFixed(2)} ${sizes[i]}`;
}

// 返回真实大小
app.get('/api/resource-sizes', async (req, res) => {
  try {
    const results = await Promise.all(FILE_LIST.map(async (name) => {
      const full = path.join(DOWNLOADS_DIR, name);
      try {
        const stat = await fs.stat(full);
        return { name, bytes: stat.size, human: formatBytes(stat.size) };
      } catch {
        return { name, error: 'not_found' };
      }
    }));
    res.json({ files: results });
  } catch {
    res.status(500).json({ error: 'internal_error' });
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
