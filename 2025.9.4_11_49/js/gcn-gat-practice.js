(function(){
  function toFixed(val, n){return Number(val).toFixed(n||4)}
  function rand(a=0,b=1){return a + Math.random()*(b-a)}
  function decayingSeries(start, end, n){
    const out=[]; for(let i=0;i<n;i++){const t=i/(n-1); out.push(start*(1-t)+end*t + rand(-0.03,0.03));}
    return out.map(v=>Math.max(0, v))
  }
  function risingSeries(start, end, n){
    const out=[]; for(let i=0;i<n;i++){const t=i/(n-1); out.push(start*(1-t)+end*t + rand(-0.02,0.02));}
    return out.map(v=>Math.max(0, Math.min(1,v)))
  }
  function simulateTraining(algorithm, epochs, lr, heads){
    const base = algorithm==='GAT' ? 0.64 : 0.60;
    const headBoost = algorithm==='GAT' ? (Math.log2(heads||1)/12) : 0;
    const lrEffect = Math.max(0, 0.02 - Math.abs(lr-0.01)*0.6);
    const finalAcc = Math.min(0.92, base + headBoost + lrEffect + rand(0.01,0.03));
    const finalLoss = Math.max(0.15, 0.9 - (finalAcc-0.5));
    const losses = decayingSeries(1.2, finalLoss, epochs);
    const accs = risingSeries(0.3, finalAcc, epochs);
    const correctFlags = Array(7).fill(0).map(()=>Math.random()>0.25);
    return {losses, accuracies: accs, finalAcc, correctFlags};
  }
  function randomAttention(n){
    const A=[]; for(let i=0;i<n;i++){const row=[]; let sum=0; for(let j=0;j<n;j++){const v = i===j?0:rand(0.0,1); row.push(v); sum+=v;}
      A.push(row.map(v=> sum>0? v/sum : 0));
    }
    return A;
  }
  function renderChartsTo(containerLossId, containerAccId, losses, accs){
    // 暂借 visualization.js 的渲染函数，通过动态切换ID实现多实例渲染
    const lossDiv = document.getElementById(containerLossId);
    const accDiv = document.getElementById(containerAccId);
    if(!lossDiv || !accDiv) return;
    // 暂存原ID
    const backupLossId = lossDiv.id; const backupAccId = accDiv.id;
    // 设为库要求的ID
    lossDiv.id = 'loss-chart';
    accDiv.id = 'accuracy-chart';
    try { if(window.renderCharts){ window.renderCharts(losses, accs); } }
    finally { lossDiv.id = backupLossId; accDiv.id = backupAccId; }
  }
  function renderGraphTo(containerId, params){
    const wrap = document.getElementById(containerId);
    if(!wrap) return;
    const oldId = wrap.id; wrap.id = 'graph-visualization';
    try { if(window.renderGraph){ window.renderGraph(params.finalAcc, params.correctFlags, params.neighborsInfo||null, params.attentionWeights||null); } }
    finally { wrap.id = oldId; }
  }

  function updateAll(){
    const lr = parseFloat(document.getElementById('lrInput').value);
    const epochs = parseInt(document.getElementById('epochInput').value,10);
    const heads = parseInt(document.getElementById('headInput').value,10);
    document.getElementById('lrVal').textContent = toFixed(lr);
    document.getElementById('epochVal').textContent = String(epochs);
    document.getElementById('headVal').textContent = String(heads);

    // 模拟训练
    const gcn = simulateTraining('GCN', epochs, lr, 1);
    const gat = simulateTraining('GAT', epochs, lr, heads);
    gat.attentionWeights = randomAttention(7);

    // 图可视化（两次调用，分别占位）
    renderGraphTo('gcn-graph', gcn);
    renderGraphTo('gat-graph', gat);

    // 曲线渲染
    renderChartsTo('gcn-loss-chart', 'gcn-accuracy-chart', gcn.losses, gcn.accuracies);
    renderChartsTo('gat-loss-chart', 'gat-accuracy-chart', gat.losses, gat.accuracies);
  }

  document.addEventListener('DOMContentLoaded', function(){
    const quickBtn = document.getElementById('quickCompareBtn');
    const runBtn = document.getElementById('runBtn');
    const resetBtn = document.getElementById('resetBtn');
    const lrInput = document.getElementById('lrInput');
    const epochInput = document.getElementById('epochInput');
    const headInput = document.getElementById('headInput');

    function reset(){
      lrInput.value = 0.01; epochInput.value = 120; headInput.value = 4;
      updateAll();
    }

    [quickBtn, runBtn].forEach(btn=> btn && btn.addEventListener('click', updateAll));
    resetBtn && resetBtn.addEventListener('click', reset);
    [lrInput, epochInput, headInput].forEach(inp=> inp && inp.addEventListener('input', updateAll));

    // 首次渲染
    updateAll();
  });
})();