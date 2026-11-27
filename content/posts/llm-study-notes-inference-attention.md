---
title: "4.3 多头自注意力：QKV、分数、遮罩、加权求和、拼接"
date: 2026-11-27
categories: ["llm-study-notes"]
tags: ["LLM", "大模型", "学习笔记", "推理", "注意力", "QKV", "Softmax"]
description: "大模型推理第四步 4.3：多头自注意力，每一步都有逐格动画。"
---

这是 [第四步：Transformer 块，逐个拆](/posts/llm-study-notes-inference-transformer-block/) 里的 4.3，隶属于 [LLM 学习笔记](/posts/llm-study-notes/)。上一页是 [4.1 RMSNorm](/posts/llm-study-notes-inference-rmsnorm/)。

玩具模型还是那个：d_model=4，2 个头，每个头 2 维，“猫 吃 鱼”3 个 token。输入 X 是 3 行 × 4 列：

|  | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 猫 | 1 | 0 | 1 | 0 |
| 吃 | 0 | 1 | 0 | 1 |
| 鱼 | 1 | 1 | 0 | 0 |

下面五步，每一步都有动画：播放 / 暂停、下一格、全部算完、重置，点结果格子也能直接跳到那一格。

## 第一步：QKV，三个线性层

Q、K、V 都是 X 乘一个权重矩阵得到的，都是 Linear（第三步讲过的）：$Q = X W_Q^T$，$K = X W_K^T$，$V = X W_V^T$。

- **Q（查询）**：每个词问“我要找什么”
- **K（键）**：每个词说“我这里有什么”
- **V（值）**：每个词真正要传出去的内容

### 动画 1：Q = X × W_Q^T

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="qkv-q-anim" style="overflow-x:auto;"></div>
<p id="qkv-q-explain">点“播放”开始。</p>
<button id="qkv-q-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="qkv-q-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="qkv-q-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="qkv-q-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>

### 动画 2：K = X × W_K^T

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="qkv-k-anim" style="overflow-x:auto;"></div>
<p id="qkv-k-explain">点“播放”开始。</p>
<button id="qkv-k-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="qkv-k-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="qkv-k-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="qkv-k-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>

### 动画 3：V = X × W_V^T

这里 W_V 是单位矩阵，所以 V = X，直接看结果：

|  | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 猫 | 1 | 0 | 1 | 0 |
| 吃 | 0 | 1 | 0 | 1 |
| 鱼 | 1 | 1 | 0 | 0 |

算出来的 Q 和 K 是：

Q（3 行 × 4 列）：

|  | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 猫 | 2 | 0 | 1 | 1 |
| 吃 | 0 | 2 | 1 | 1 |
| 鱼 | 1 | 1 | 1 | 1 |

K（3 行 × 4 列）：

|  | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 猫 | 1 | 1 | 1 | 1 |
| 吃 | 1 | 1 | 1 | 1 |
| 鱼 | 2 | 1 | 0 | 1 |

<script>
function matAnim(p, A, BT, C, names){
  // A (m x n) x BT (n x p) = C (m x p); BT is already transposed
  var m=A.length, n=A[0].length, pc=BT[0].length;
  var steps=[];
  for(var i=0;i<m;i++) for(var j=0;j<pc;j++){
    for(var k=0;k<n;k++) steps.push({type:'mul', i:i, j:j, k:k});
    steps.push({type:'sum', i:i, j:j});
  }
  var idx=-1, timer=null, playing=false;
  function cell(v, hl){
    var bg = hl==='cur' ? '#facc15' : hl==='done' ? '#2563eb' : '#1f2937';
    var col = hl==='cur' ? '#000' : '#fff';
    return '<td style="border:1px solid #888;padding:8px 12px;background:'+bg+';color:'+col+';">'+v+'</td>';
  }
  function mat(M, hi, hj, doneSet){
    var h='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;">';
    for(var r=0;r<M.length;r++){
      h+='<tr><td style="padding:4px;">'+(names?names[r]:'')+'</td>';
      for(var c=0;c<M[r].length;c++){
        var hl='';
        if(hi!==null && hi.r===r && hi.c===c) hl='cur';
        else if(doneSet && doneSet[r+','+c]) hl='done';
        h+=cell(M[r][c], hl);
      }
      h+='</tr>';
    }
    return h+'</table>';
  }
  function render(){
    var s = idx>=0 ? steps[idx] : null;
    var aHi=null, bHi=null, cCur=null, cDone={};
    if(s){
      aHi={r:s.i, c:s.k}; bHi={r:s.k, c:s.j}; cCur={r:s.i, c:s.j};
      for(var t=0;t<=idx;t++){ var st=steps[t]; if(st.type==='sum') cDone[st.i+','+st.j]=1; }
    }
    var html=mat(A, aHi, null, null);
    html+='<span style="margin:0 8px;">×</span>';
    html+=mat(BT, bHi, null, null);
    html+='<span style="margin:0 8px;">=</span>';
    html+=mat(C, cCur, null, cDone);
    document.getElementById(p+'-anim').innerHTML=html;
    var ex=document.getElementById(p+'-explain');
    if(!s){ ex.textContent='点“播放”开始。'; return; }
    if(s.type==='mul'){
      ex.textContent='算第 '+(s.i+1)+' 行第 '+(s.j+1)+' 列：'+A[s.i][s.k]+'×'+BT[s.k][s.j]+' = '+A[s.i][s.k]*BT[s.k][s.j]+'。黄色是正在乘的两个数。';
    } else {
      var parts=[]; for(var k2=0;k2<n;k2++) parts.push(A[s.i][k2]+'×'+BT[k2][s.j]);
      ex.textContent='第 '+(s.i+1)+' 行第 '+(s.j+1)+' 列 = '+parts.join(' + ')+' = '+C[s.i][s.j]+'。';
    }
    // make result cells clickable to jump
    var tbls=document.getElementById(p+'-anim').querySelectorAll('table');
    var ct=tbls[tbls.length-1];
    ct.querySelectorAll('tr').forEach(function(tr, r){
      tr.querySelectorAll('td').forEach(function(td, c){
        if(c===0) return;
        var cc=c-1;
        td.style.cursor='pointer';
        td.onclick=function(){
          stop();
          for(var t2=0;t2<steps.length;t2++){ if(steps[t2].type==='sum'&&steps[t2].i===r&&steps[t2].j===cc){ idx=t2; break; } }
          render();
        };
      });
    });
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById(p+'-play').textContent='▶ 播放'; }
  function step(){ if(idx>=steps.length-1){ stop(); return; } idx++; render(); if(idx>=steps.length-1) stop(); }
  document.getElementById(p+'-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=steps.length-1){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 900);
  });
  document.getElementById(p+'-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById(p+'-all').addEventListener('click', function(){ stop(); idx=steps.length-1; render(); });
  document.getElementById(p+'-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
}
(function(){
  var X=[[1,0,1,0],[0,1,0,1],[1,1,0,0]];
  var names=['猫','吃','鱼'];
  var WqT=[[1,0,1,0],[0,1,0,1],[1,0,0,1],[0,1,1,0]];
  var WkT=[[1,0,0,1],[1,1,0,0],[0,1,1,0],[0,0,1,1]];
  var Q=[[2,0,1,1],[0,2,1,1],[1,1,1,1]];
  var K=[[1,1,1,1],[1,1,1,1],[2,1,0,1]];
  matAnim('qkv-q', X, WqT, Q, names);
  matAnim('qkv-k', X, WkT, K, names);
})();
</script>

## 第二步：注意力分数，谁在看谁

每个头单独算。头 1 用 Q、K 的第 1–2 列：$S_1 = Q_1 K_1^T \div \sqrt{d_h}$，$d_h = 2$，$\sqrt{2} \approx 1.41$。

形状：(3 行 × 2 列) × (2 行 × 3 列) → 3 行 × 3 列。行是“谁在看”，列是“被看的是谁”。分数越大，第 i 个词越“关注”第 j 个词。

除以 $\sqrt{d_h}$ 是为了防止点积太大，让后面的 Softmax 不至于太尖锐。

### 动画 4：头 1 的分数 S₁ = Q₁ × K₁^T ÷ √2

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="score-anim" style="overflow-x:auto;"></div>
<p id="score-explain">点“播放”开始。</p>
<button id="score-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="score-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="score-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="score-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var Q1=[[2,0],[0,2],[1,1]], K1T=[[1,1,2],[1,1,1]];
  var S1=[[1.41,1.41,2.83],[1.41,1.41,1.41],[1.41,1.41,2.12]];
  var names=['猫','吃','鱼'];
  var steps=[];
  for(var i=0;i<3;i++) for(var j=0;j<3;j++){
    for(var k=0;k<2;k++) steps.push({type:'mul', i:i, j:j, k:k});
    steps.push({type:'sum', i:i, j:j});
  }
  var idx=-1, timer=null, playing=false;
  function cell(v, hl){
    var bg = hl==='cur' ? '#facc15' : hl==='done' ? '#2563eb' : '#1f2937';
    var col = hl==='cur' ? '#000' : '#fff';
    return '<td style="border:1px solid #888;padding:8px 12px;background:'+bg+';color:'+col+';">'+v+'</td>';
  }
  function mat(M, hi, doneSet){
    var h='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;">';
    for(var r=0;r<M.length;r++){
      h+='<tr><td style="padding:4px;">'+names[r]+'</td>';
      for(var c=0;c<M[r].length;c++){
        var hl='';
        if(hi && hi.r===r && hi.c===c) hl='cur';
        else if(doneSet && doneSet[r+','+c]) hl='done';
        h+=cell(M[r][c], hl);
      }
      h+='</tr>';
    }
    return h+'</table>';
  }
  function render(){
    var s = idx>=0 ? steps[idx] : null;
    var aHi=null, bHi=null, cDone={};
    if(s){ aHi={r:s.i, c:s.k}; bHi={r:s.k, c:s.j}; for(var t=0;t<=idx;t++){ var st=steps[t]; if(st.type==='sum') cDone[st.i+','+st.j]=1; } }
    var html=mat(Q1, aHi, null)+'<span style="margin:0 8px;">×</span>'+mat(K1T, bHi, null)+'<span style="margin:0 8px;">÷ √2 =</span>'+mat(S1, s?{r:s.i,c:s.j}:null, cDone);
    document.getElementById('score-anim').innerHTML=html;
    var ex=document.getElementById('score-explain');
    if(!s){ ex.textContent='点“播放”开始。'; return; }
    if(s.type==='mul') ex.textContent='算 S['+s.i+']['+s.j+']：'+Q1[s.i][s.k]+'×'+K1T[s.k][s.j]+' = '+Q1[s.i][s.k]*K1T[s.k][s.j]+'。';
    else {
      var raw=0; for(var k2=0;k2<2;k2++) raw+=Q1[s.i][k2]*K1T[k2][s.j];
      ex.textContent='S['+s.i+']['+s.j+'] = '+raw+' ÷ 1.41 = '+S1[s.i][s.j]+'。第 '+(s.i+1)+' 个词对第 '+(s.j+1)+' 个词的关注分数。';
    }
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById('score-play').textContent='▶ 播放'; }
  function step(){ if(idx>=steps.length-1){ stop(); return; } idx++; render(); if(idx>=steps.length-1) stop(); }
  document.getElementById('score-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=steps.length-1){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 900);
  });
  document.getElementById('score-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById('score-all').addEventListener('click', function(){ stop(); idx=steps.length-1; render(); });
  document.getElementById('score-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
})();
</script>

头 1 的分数（除以 √2 之后）：

|  | 猫 | 吃 | 鱼 |
| --- | --- | --- | --- |
| 猫 | 1.41 | 1.41 | 2.83 |
| 吃 | 1.41 | 1.41 | 1.41 |
| 鱼 | 1.41 | 1.41 | 2.12 |

“猫”最关注“鱼”（2.83），“鱼”也最关注“鱼”（2.12）。头 2 同理，算法一模一样，只是用的列不同。

## 第三步：遮罩 + Softmax，变成权重

解码器是自回归的：生成第 i 个词时，只能看前面的词，不能看后面的。所以分数矩阵的上三角要遮掉，变成 -∞，Softmax 之后就是 0。

### 动画 5：遮罩 + Softmax，一行一行算

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="mask-anim" style="overflow-x:auto;"></div>
<p id="mask-explain">点“播放”开始。</p>
<button id="mask-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="mask-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="mask-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="mask-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var S=[[1.41,1.41,2.83],[1.41,1.41,1.41],[1.41,1.41,2.12]];
  var names=['猫','吃','鱼'];
  var W=[[1,0,0],[0.5,0.5,0],[0.25,0.25,0.5]];
  // steps: for each row: mask step, then softmax step
  var steps=[];
  for(var i=0;i<3;i++){ steps.push({type:'mask', i:i}); steps.push({type:'soft', i:i}); }
  var idx=-1, timer=null, playing=false;
  function cell(v, hl){
    var bg = hl==='cur' ? '#facc15' : hl==='done' ? '#2563eb' : hl==='mask' ? '#7f1d1d' : '#1f2937';
    var col = hl==='cur' ? '#000' : '#fff';
    return '<td style="border:1px solid #888;padding:8px 12px;background:'+bg+';color:'+col+';">'+v+'</td>';
  }
  function render(){
    var s = idx>=0 ? steps[idx] : null;
    function mat(M, rowHi, doneRows){
      var h='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;">';
      for(var r=0;r<3;r++){
        h+='<tr><td style="padding:4px;">'+names[r]+'</td>';
        for(var c=0;c<3;c++){
          var hl='';
          if(rowHi===r) hl='cur';
          else if(doneRows && doneRows.indexOf(r)>=0) hl='done';
          h+=cell(M[r][c], hl);
        }
        h+='</tr>';
      }
      return h+'</table>';
    }
    var masked=S.map(function(row,i){ return row.map(function(v,j){ return j<=i ? v : '-∞'; }); });
    var html=mat(S, s&&s.type==='mask'?s.i:null, null);
    html+='<span style="margin:0 8px;">→ 遮罩 →</span>'+mat(masked, s&&s.type==='mask'?s.i:null, null);
    html+='<span style="margin:0 8px;">→ Softmax →</span>'+mat(W, s&&s.type==='soft'?s.i:null, null);
    document.getElementById('mask-anim').innerHTML=html;
    var ex=document.getElementById('mask-explain');
    if(!s){ ex.textContent='点“播放”开始。'; return; }
    if(s.type==='mask'){
      var kept=[]; for(var j=0;j<=s.i;j++) kept.push(S[s.i][j]);
      ex.textContent='第 '+(s.i+1)+' 行遮罩：只能看前 '+(s.i+1)+' 个词，后面的变成 -∞（红色）。保留 '+kept.join(', ')+'。';
    } else {
      var row=W[s.i].filter(function(v){return v>0;});
      ex.textContent='第 '+(s.i+1)+' 行 Softmax：e 的指数归一化，得到权重 ['+W[s.i].join(', ')+']，一行加起来是 1。';
    }
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById('mask-play').textContent='▶ 播放'; }
  function step(){ if(idx>=steps.length-1){ stop(); return; } idx++; render(); if(idx>=steps.length-1) stop(); }
  document.getElementById('mask-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=steps.length-1){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 1000);
  });
  document.getElementById('mask-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById('mask-all').addEventListener('click', function(){ stop(); idx=steps.length-1; render(); });
  document.getElementById('mask-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
})();
</script>

遮罩后的权重（头 1）：

|  | 猫 | 吃 | 鱼 |
| --- | --- | --- | --- |
| 猫 | 1 | 0 | 0 |
| 吃 | 0.5 | 0.5 | 0 |
| 鱼 | 0.25 | 0.25 | 0.5 |

每一行加起来都是 1。“猫”只能看自己，“吃”看猫和自己各一半，“鱼”看猫、吃各 0.25，看自己 0.5。

## 第四步：加权求和，拿权重去乘 V

输出 = 权重 × V。每个词的新表示，就是它关注的那些词的 V 按权重加权平均。

### 动画 6：头 1 输出 = 权重 × V₁

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="ws-anim" style="overflow-x:auto;"></div>
<p id="ws-explain">点“播放”开始。</p>
<button id="ws-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="ws-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="ws-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="ws-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var W=[[1,0,0],[0.5,0.5,0],[0.25,0.25,0.5]];
  var V1=[[1,0],[0,1],[1,1]];
  var O1=[[1,0],[0.5,0.5],[0.75,0.75]];
  var names=['猫','吃','鱼'];
  var steps=[];
  for(var i=0;i<3;i++) for(var j=0;j<2;j++){
    for(var k=0;k<3;k++) steps.push({type:'mul', i:i, j:j, k:k});
    steps.push({type:'sum', i:i, j:j});
  }
  var idx=-1, timer=null, playing=false;
  function cell(v, hl){
    var bg = hl==='cur' ? '#facc15' : hl==='done' ? '#2563eb' : '#1f2937';
    var col = hl==='cur' ? '#000' : '#fff';
    return '<td style="border:1px solid #888;padding:8px 12px;background:'+bg+';color:'+col+';">'+v+'</td>';
  }
  function mat(M, hi, doneSet){
    var h='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;">';
    for(var r=0;r<M.length;r++){
      h+='<tr><td style="padding:4px;">'+names[r]+'</td>';
      for(var c=0;c<M[r].length;c++){
        var hl='';
        if(hi && hi.r===r && hi.c===c) hl='cur';
        else if(doneSet && doneSet[r+','+c]) hl='done';
        h+=cell(M[r][c], hl);
      }
      h+='</tr>';
    }
    return h+'</table>';
  }
  function render(){
    var s = idx>=0 ? steps[idx] : null;
    var aHi=null, bHi=null, cDone={};
    if(s){ aHi={r:s.i, c:s.k}; bHi={r:s.k, c:s.j}; for(var t=0;t<=idx;t++){ var st=steps[t]; if(st.type==='sum') cDone[st.i+','+st.j]=1; } }
    var html=mat(W, aHi, null)+'<span style="margin:0 8px;">×</span>'+mat(V1, bHi, null)+'<span style="margin:0 8px;">=</span>'+mat(O1, s?{r:s.i,c:s.j}:null, cDone);
    document.getElementById('ws-anim').innerHTML=html;
    var ex=document.getElementById('ws-explain');
    if(!s){ ex.textContent='点“播放”开始。'; return; }
    if(s.type==='mul') ex.textContent='权重 '+W[s.i][s.k]+' × V['+s.k+']['+s.j+'] = '+W[s.i][s.k]*V1[s.k][s.j]+'。';
    else {
      var parts=[]; for(var k2=0;k2<3;k2++) parts.push(W[s.i][k2]+'×'+V1[k2][s.j]);
      ex.textContent='输出['+s.i+']['+s.j+'] = '+parts.join(' + ')+' = '+O1[s.i][s.j]+'。';
    }
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById('ws-play').textContent='▶ 播放'; }
  function step(){ if(idx>=steps.length-1){ stop(); return; } idx++; render(); if(idx>=steps.length-1) stop(); }
  document.getElementById('ws-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=steps.length-1){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 900);
  });
  document.getElementById('ws-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById('ws-all').addEventListener('click', function(){ stop(); idx=steps.length-1; render(); });
  document.getElementById('ws-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
})();
</script>

头 1 输出（3 行 × 2 列）：

|  | 1 | 2 |
| --- | --- | --- |
| 猫 | 1 | 0 |
| 吃 | 0.5 | 0.5 |
| 鱼 | 0.75 | 0.75 |

头 2 同理，输出是 [[1, 0], [0.5, 0.5], [0.4, 0.4]]。

## 第五步：拼接 + 投影，拼回 4 维

两个头的输出拼起来，3 行 × 2 列 + 3 行 × 2 列 = 3 行 × 4 列，再乘一个投影矩阵 $W_O$（这里是单位矩阵，所以结果不变）：

|  | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 猫 | 1 | 0 | 1 | 0 |
| 吃 | 0.5 | 0.5 | 0.5 | 0.5 |
| 鱼 | 0.75 | 0.75 | 0.4 | 0.4 |

这就是多头注意力的输出，形状和输入一样，都是 3 行 × 4 列，可以直接加残差进下一层。

### 动画 7：拼接，一格一格拼

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="cat-anim" style="overflow-x:auto;"></div>
<p id="cat-explain">点“播放”开始。</p>
<button id="cat-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="cat-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="cat-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="cat-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var O1=[[1,0],[0.5,0.5],[0.75,0.75]], O2=[[1,0],[0.5,0.5],[0.4,0.4]];
  var O=[[1,0,1,0],[0.5,0.5,0.5,0.5],[0.75,0.75,0.4,0.4]];
  var names=['猫','吃','鱼'];
  var steps=[];
  for(var i=0;i<3;i++) for(var j=0;j<4;j++) steps.push({i:i, j:j});
  var idx=-1, timer=null, playing=false;
  function cell(v, hl){
    var bg = hl==='cur' ? '#facc15' : hl==='done' ? '#2563eb' : '#1f2937';
    var col = hl==='cur' ? '#000' : '#fff';
    return '<td style="border:1px solid #888;padding:8px 12px;background:'+bg+';color:'+col+';">'+v+'</td>';
  }
  function mat(M, cur, doneSet){
    var h='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;">';
    for(var r=0;r<M.length;r++){
      h+='<tr><td style="padding:4px;">'+names[r]+'</td>';
      for(var c=0;c<M[r].length;c++){
        var hl='';
        if(cur && cur.r===r && cur.c===c) hl='cur';
        else if(doneSet && doneSet[r+','+c]) hl='done';
        h+=cell(M[r][c], hl);
      }
      h+='</tr>';
    }
    return h+'</table>';
  }
  function render(){
    var s = idx>=0 ? steps[idx] : null;
    var done={};
    for(var t=0;t<=idx;t++) done[steps[t].i+','+steps[t].j]=1;
    var html=mat(O1, null, null)+'<span style="margin:0 8px;">+</span>'+mat(O2, null, null)+'<span style="margin:0 8px;">=</span>'+mat(O, s, done);
    document.getElementById('cat-anim').innerHTML=html;
    var ex=document.getElementById('cat-explain');
    if(!s){ ex.textContent='点“播放”开始。'; return; }
    var src = s.j<2 ? '头 1' : '头 2';
    ex.textContent='输出['+s.i+']['+s.j+'] = '+O[s.i][s.j]+'，来自'+src+'的第 '+(s.j<2?s.j+1:s.j-1)+' 列。两个头拼起来，2 列 + 2 列 = 4 列。';
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById('cat-play').textContent='▶ 播放'; }
  function step(){ if(idx>=steps.length-1){ stop(); return; } idx++; render(); if(idx>=steps.length-1) stop(); }
  document.getElementById('cat-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=steps.length-1){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 600);
  });
  document.getElementById('cat-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById('cat-all').addEventListener('click', function(){ stop(); idx=steps.length-1; render(); });
  document.getElementById('cat-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
})();
</script>

## 自测：注意力过关了吗

### 第 1 题

Q、K、V 分别是干什么的？都是怎么算出来的？

<details>
<summary>点开看解析和答案</summary>

Q 是查询，K 是键，V 是值。都是 X 乘一个权重矩阵得到的 Linear：$Q = X W_Q^T$，$K = X W_K^T$，$V = X W_V^T$。形状都是 3 行 × 4 列。

</details>

### 第 2 题

注意力分数 S = QKᵀ ÷ √d 是怎么算的？行和列分别是什么意思？

<details>
<summary>点开看解析和答案</summary>

每个头单独算，头 1 用 Q、K 的第 1–2 列，(3×2) × (2×3) → 3×3。行是“谁在看”，列是“被看的是谁”，分数越大越关注。除以 √2 是防止点积太大，Softmax 太尖锐。

</details>

### 第 3 题

遮罩是干什么的？为什么“猫”那一行只有第一个数不是 0？

<details>
<summary>点开看解析和答案</summary>

解码器自回归，生成第 i 个词只能看前面的词，不能看后面的。遮罩把上三角变成 -∞，Softmax 后就是 0。“猫”是第 1 个词，前面没有词，只能看自己，所以权重是 [1, 0, 0]。

</details>

### 第 4 题

加权求和是怎么算的？“鱼”那一行的输出 0.75 是怎么来的？

<details>
<summary>点开看解析和答案</summary>

输出 = 权重 × V。鱼的权重是 [0.25, 0.25, 0.5]，V₁ 的第一列是 [1, 0, 1]，所以 0.25×1 + 0.25×0 + 0.5×1 = 0.75。

</details>

### 第 5 题

为什么要多头？两个头拼起来之后形状是多少？

<details>
<summary>点开看解析和答案</summary>

每个头关注的东西不一样，一个头看语法，一个头看语义，合起来更全面。两个头各输出 3 行 × 2 列，拼起来是 3 行 × 4 列，再乘投影矩阵 $W_O$，形状不变。

</details>

全答得出来，4.3 就过了。

---

下一步：[4.4 FFN（SwiGLU）](/posts/llm-study-notes-inference-transformer-block/)（还没写，先回第四步目录）

回目录：[第四步：Transformer 块，逐个拆](/posts/llm-study-notes-inference-transformer-block/)
