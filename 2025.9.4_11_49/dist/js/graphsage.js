// GraphSAGE 演示脚本：K-邻居采样 + L 层聚合 + 训练曲线模拟
(function(){
  // 工具函数
  function rand(min, max){ return Math.random()*(max-min)+min; }
  function choice(arr){ return arr[Math.floor(Math.random()*arr.length)]; }
  function shuffle(arr){ return arr.map(v=>[Math.random(),v]).sort((a,b)=>a[0]-b[0]).map(x=>x[1]); }

  // 固定的 10 节点示例图（需与 renderGraph 内 GraphSAGE 模式保持一致）
  const N = 10;
  const EDGES = [
    [0,1],[0,3],[1,2],[1,4],[2,5],[3,4],[3,6],[4,5],[4,7],[5,8],[6,7],[6,9],[7,8],[8,9]
  ];
  const ADJ = Array.from({length:N}, ()=>new Set());
  EDGES.forEach(([u,v])=>{ ADJ[u].add(v); ADJ[v].add(u); });

  // 根据 K / L 为每个节点进行分层采样，返回 neighborsInfo: {nodeId: [sampledIds...]}
  function sampleNeighborsAll(K, L){
    const info = {};
    for(let src=0; src<N; src++){
      let frontier = new Set([src]);
      let sampled = new Set();
      for(let layer=0; layer<L; layer++){
        const next = new Set();
        frontier.forEach(f=>{
          const neigh = Array.from(ADJ[f]);
          if(neigh.length===0) return;
          const picks = shuffle(neigh).slice(0, Math.min(K, neigh.length));
          picks.forEach(n=>{ if(n!==src){ sampled.add(n); next.add(n); } });
        });
        frontier = next;
        if(frontier.size===0) break;
      }
      info[src] = Array.from(sampled);
    }
    return info;
  }

  // 训练曲线模拟：K/L 越大，通常收敛更快（更低的 loss、更快上升的 acc）
  function simulateTraining(K, L, epochs=30){
    const losses = []; const accs = [];
    let loss = 1.0; let acc = 0.30;
    const lrLoss = Math.max(0.75, 0.93 - 0.03*(K-1) - 0.04*(L-1));
    const alphaAcc = Math.min(0.35, 0.08 + 0.02*(K-1) + 0.03*(L-1));
    for(let t=0;t<epochs;t++){
      loss = Math.max(0.02, loss*lrLoss + rand(-0.01, 0.015));
      acc = Math.min(0.99, acc + (1-acc)*alphaAcc + rand(-0.01,0.01));
      losses.push(Number(loss.toFixed(3)));
      accs.push(Number(Math.max(0,Math.min(1,acc)).toFixed(3)));
    }
    return { losses, accs };
  }

  // 根据最终准确率生成正确标记（演示用途）
  function makeCorrectFlags(finalAcc){
    const cnt = Math.round(finalAcc * N);
    const order = shuffle([...Array(N).keys()]);
    const flags = Array(N).fill(false);
    for(let i=0;i<cnt;i++) flags[order[i]] = true;
    return flags;
  }

  // 一次完整渲染：采样 -> 曲线 -> 图
  function renderOnce(K, L){
    const neighborsInfo = sampleNeighborsAll(K, L);
    const {losses, accs} = simulateTraining(K, L);
    if(window.renderCharts) window.renderCharts(losses, accs);
    const finalAcc = accs[accs.length-1] || 0.8;
    const correctFlags = makeCorrectFlags(finalAcc);
    if(window.renderGraph) window.renderGraph(finalAcc, correctFlags, neighborsInfo, null);
  }

  // UI 绑定
  function bindUI(){
    const kSlider = document.getElementById('kSlider');
    const lSlider = document.getElementById('lSlider');
    const kValue = document.getElementById('kValue');
    const lValue = document.getElementById('lValue');
    const runBtn = document.getElementById('runBtn');
    const resetBtn = document.getElementById('resetBtn');

    // 非 GraphSAGE 页面防御：所需元素不存在则跳过初始化
    if(!kSlider || !lSlider || !kValue || !lValue || !runBtn || !resetBtn){
      return;
    }

    function getK(){ return parseInt(kSlider.value,10); }
    function getL(){ return parseInt(lSlider.value,10); }

    kSlider.addEventListener('input', ()=>{ kValue.textContent = getK(); });
    lSlider.addEventListener('input', ()=>{ lValue.textContent = getL(); });

    runBtn.addEventListener('click', ()=>{
      renderOnce(getK(), getL());
    });

    resetBtn.addEventListener('click', ()=>{
      kSlider.value = 2; lSlider.value = 2; kValue.textContent = '2'; lValue.textContent = '2';
      // 清空并渲染初始状态
      const gv = document.getElementById('graph-visualization');
      if(gv) gv.innerHTML='';
      const lc = document.getElementById('loss-chart');
      const ac = document.getElementById('accuracy-chart');
      if(lc) lc.innerHTML=''; if(ac) ac.innerHTML='';
    });

    // 首屏给一个默认演示
    renderOnce(getK(), getL());
  }

  document.addEventListener('DOMContentLoaded', bindUI);
})();