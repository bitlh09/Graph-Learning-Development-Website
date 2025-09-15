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
    // 显示训练进度
    showTrainingProgress();
    
    const base = algorithm==='GAT' ? 0.64 : 0.60;
    const headBoost = algorithm==='GAT' ? (Math.log2(heads||1)/12) : 0;
    const lrEffect = Math.max(0, 0.02 - Math.abs(lr-0.01)*0.6);
    const finalAcc = Math.min(0.92, base + headBoost + lrEffect + rand(0.01,0.03));
    const finalLoss = Math.max(0.15, 0.9 - (finalAcc-0.5));
    
    // 增强的训练曲线生成，添加更真实的噪声
    const losses = decayingSeries(1.2, finalLoss, epochs).map(v => Math.max(0.01, v + (Math.random()-0.5)*0.05));
    const accs = risingSeries(0.3, finalAcc, epochs).map(v => Math.max(0.1, Math.min(0.98, v + (Math.random()-0.5)*0.03)));
    const correctFlags = Array(7).fill(0).map(()=>Math.random()>0.25);
    
    // 隐藏训练进度并显示完成提示
    setTimeout(() => {
      hideTrainingProgress();
      showTrainingComplete();
    }, 500);
    
    return {losses, accuracies: accs, finalAcc, correctFlags};
  }
  
  // 显示训练进度
  function showTrainingProgress() {
    const progressHTML = `
      <div id="training-progress" class="fixed top-4 right-4 bg-white rounded-lg shadow-lg p-4 z-50 border-l-4 border-blue-500" style="position: fixed; top: 16px; right: 16px; background: white; border-radius: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); padding: 16px; z-index: 50; border-left: 4px solid #3b82f6;">
        <div style="display: flex; align-items: center; gap: 12px;">
          <div style="width: 24px; height: 24px; border: 2px solid #3b82f6; border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
          <span style="font-size: 14px; font-weight: 600; color: #374151;">🚀 训练进行中...</span>
        </div>
      </div>
      <style>
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
      </style>
    `;
    document.body.insertAdjacentHTML('beforeend', progressHTML);
  }
  
  // 隐藏训练进度
  function hideTrainingProgress() {
    const progress = document.getElementById('training-progress');
    if (progress) {
      progress.remove();
    }
  }
  
  // 显示训练完成提示
  function showTrainingComplete() {
    const completeHTML = `
      <div id="training-complete" style="position: fixed; top: 16px; right: 16px; background: #f0fdf4; border-radius: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); padding: 16px; z-index: 50; border-left: 4px solid #10b981;">
        <div style="display: flex; align-items: center; gap: 12px;">

         
        </div>
      </div>
    `;
    document.body.insertAdjacentHTML('beforeend', completeHTML);
    
    setTimeout(() => {
      const complete = document.getElementById('training-complete');
      if (complete) {
        complete.style.transition = 'all 0.3s ease';
        complete.style.opacity = '0';
        complete.style.transform = 'translateX(100%)';
        setTimeout(() => complete.remove(), 300);
      }
    }, 2000);
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

    // 增强的参数变化监听 - 添加实时反馈
    function addParameterFeedback(element, paramName) {
      element.addEventListener('input', function() {
        // 添加参数变化动画
        element.style.transform = 'scale(1.05)';
        element.style.transition = 'transform 0.2s ease';
        
        // 显示参数变化提示
        showParameterChange(paramName, element.value);
        
        setTimeout(() => {
          element.style.transform = 'scale(1)';
        }, 200);
        
        // 延迟更新以避免频繁计算
        clearTimeout(element.updateTimeout);
        element.updateTimeout = setTimeout(updateAll, 300);
      });
    }

    function reset(){
      // 添加重置动画
      if(resetBtn) {
        resetBtn.style.transform = 'scale(0.95)';
        setTimeout(() => {
          resetBtn.style.transform = 'scale(1)';
        }, 150);
      }
      
      // 显示重置提示
      showResetFeedback();
      
      lrInput.value = 0.01; epochInput.value = 120; headInput.value = 4;
      
      // 更新显示值
      document.getElementById('lrVal').textContent = '0.0100';
      document.getElementById('epochVal').textContent = '120';
      document.getElementById('headVal').textContent = '4';
      
      updateAll();
    }

    // 增强的按钮事件
    if(quickBtn) {
      quickBtn.addEventListener('click', function() {
        this.style.transform = 'scale(0.95)';
        setTimeout(() => {
          this.style.transform = 'scale(1)';
        }, 150);
        updateAll();
      });
    }
    
    if(runBtn) {
      runBtn.addEventListener('click', function() {
        this.style.transform = 'scale(0.95)';
        setTimeout(() => {
          this.style.transform = 'scale(1)';
        }, 150);
        updateAll();
      });
    }
    
    resetBtn && resetBtn.addEventListener('click', reset);
    
    // 添加参数变化的视觉反馈
    if(lrInput) addParameterFeedback(lrInput, '学习率');
    if(epochInput) addParameterFeedback(epochInput, '训练轮数');
    if(headInput) addParameterFeedback(headInput, '注意力头数');

    // 首次渲染
    updateAll();
  });
  
  // 显示参数变化反馈
  function showParameterChange(paramName, value) {
    const existingFeedback = document.getElementById('param-feedback');
    if (existingFeedback) {
      existingFeedback.remove();
    }
    
    const feedbackHTML = `
      <div id="param-feedback" style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0,0,0,0.8); color: white; padding: 12px 20px; border-radius: 8px; z-index: 100; font-size: 14px; font-weight: 600;">
        ${paramName}: ${value}
      </div>
    `;
    document.body.insertAdjacentHTML('beforeend', feedbackHTML);
    
    setTimeout(() => {
      const feedback = document.getElementById('param-feedback');
      if (feedback) {
        feedback.style.transition = 'opacity 0.3s ease';
        feedback.style.opacity = '0';
        setTimeout(() => feedback.remove(), 300);
      }
    }, 1000);
  }
  
  // 显示重置反馈
  function showResetFeedback() {
    const resetHTML = `
      <div id="reset-feedback" style="position: fixed; top: 4px; left: 50%; transform: translateX(-50%); background: #fef3c7; color: #92400e; padding: 12px 20px; border-radius: 8px; z-index: 100; font-size: 14px; font-weight: 600; border: 1px solid #fbbf24;">
        🔄 参数已重置为默认值
      </div>
    `;
    document.body.insertAdjacentHTML('beforeend', resetHTML);
    
    setTimeout(() => {
      const feedback = document.getElementById('reset-feedback');
      if (feedback) {
        feedback.style.transition = 'all 0.3s ease';
        feedback.style.opacity = '0';
        feedback.style.transform = 'translateX(-50%) translateY(-20px)';
        setTimeout(() => feedback.remove(), 300);
      }
    }, 2000);
  }
})();