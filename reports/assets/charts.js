// Portfolio v7.6 Report Charts
(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var danger = style.getPropertyValue('--danger').trim();
  var warn = style.getPropertyValue('--warn').trim();
  var surface = style.getPropertyValue('--surface').trim();

  // Common chart config
  var commonGrid = {
    left: '8%',
    right: '8%',
    top: '12%',
    bottom: '15%'
  };

  var commonAxis = {
    axisLine: { lineStyle: { color: rule } },
    axisLabel: { color: muted, fontSize: 11 },
    splitLine: { lineStyle: { color: rule, opacity: 0.3 } }
  };

  var commonTooltip = {
    backgroundColor: surface,
    borderColor: rule,
    textStyle: { color: ink, fontSize: 13 }
  };

  // ===== Chart 1: ETF vs Stock Split (Doughnut) =====
  var chart1 = echarts.init(document.getElementById('chart-asset-split'));
  chart1.setOption({
    tooltip: Object.assign(commonTooltip, { trigger: 'item', formatter: '{b}: {c}元 ({d}%)' }),
    legend: {
      bottom: 10,
      textStyle: { color: muted, fontSize: 12 },
      data: ['ETF', '股票']
    },
    series: [{
      type: 'pie',
      radius: ['45%', '70%'],
      center: ['50%', '42%'],
      avoidLabelOverlap: false,
      itemStyle: { borderRadius: 6, borderColor: surface, borderWidth: 2 },
      label: {
        show: true,
        position: 'center',
        formatter: function() {
          return '{a|总投入}\n{b|87,985元}';
        },
        rich: {
          a: { color: muted, fontSize: 12, lineHeight: 20 },
          b: { color: ink, fontSize: 20, fontWeight: 'bold', fontFamily: 'IBMPlexMono, monospace' }
        }
      },
      emphasis: {
        label: { show: true, fontSize: 14, fontWeight: 'bold' }
      },
      data: [
        { value: 52077, name: 'ETF', itemStyle: { color: accent } },
        { value: 35908, name: '股票', itemStyle: { color: accent2 } }
      ]
    }]
  });

  // ===== Chart 2: Theme Split (Horizontal Bar) =====
  var chart2 = echarts.init(document.getElementById('chart-theme-split'));
  chart2.setOption({
    tooltip: Object.assign(commonTooltip, { trigger: 'item', formatter: '{b}: {c}%' }),
    grid: { left: '20%', right: '10%', top: '10%', bottom: '10%' },
    xAxis: {
      type: 'value',
      max: 35,
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11, formatter: '{value}%' },
      splitLine: { lineStyle: { color: rule, opacity: 0.3 } }
    },
    yAxis: {
      type: 'category',
      data: ['金融地产', '消费白酒', '其他', '新能源', '消费医药'],
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: ink, fontSize: 12 }
    },
    series: [{
      type: 'bar',
      barWidth: '60%',
      data: [
        { value: 8.6, itemStyle: { color: danger } },
        { value: 14.3, itemStyle: { color: '#a371f7' } },
        { value: 21.0, itemStyle: { color: warn } },
        { value: 26.5, itemStyle: { color: accent2 } },
        { value: 29.6, itemStyle: { color: accent } }
      ],
      label: {
        show: true,
        position: 'right',
        color: ink,
        fontSize: 12,
        fontFamily: 'IBMPlexMono, monospace',
        formatter: '{c}%'
      }
    }]
  });

  // ===== Chart 3: Risk-Return Scatter =====
  var chart3 = echarts.init(document.getElementById('chart-risk-return'));
  chart3.setOption({
    tooltip: Object.assign(commonTooltip, {
      trigger: 'item',
      formatter: function(p) {
        return p.data[3] + '<br/>盈利概率: ' + (p.data[0] * 100).toFixed(1) + '%<br/>预期收益: ' + (p.data[1] * 100).toFixed(2) + '%<br/>持仓金额: ' + p.data[2].toLocaleString() + '元';
      }
    }),
    grid: commonGrid,
    xAxis: {
      type: 'value',
      name: '盈利概率',
      nameLocation: 'middle',
      nameGap: 30,
      nameTextStyle: { color: muted, fontSize: 12 },
      min: 0.3,
      max: 0.65,
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11, formatter: function(v) { return (v * 100).toFixed(0) + '%'; } },
      splitLine: { lineStyle: { color: rule, opacity: 0.3 } }
    },
    yAxis: {
      type: 'value',
      name: '预期收益率',
      nameLocation: 'middle',
      nameGap: 40,
      nameTextStyle: { color: muted, fontSize: 12 },
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11, formatter: function(v) { return (v * 100).toFixed(1) + '%'; } },
      splitLine: { lineStyle: { color: rule, opacity: 0.3 } }
    },
    series: [{
      type: 'scatter',
      symbolSize: function(data) { return Math.sqrt(data[2] / 50) * 2.5 + 15; },
      data: [
        [0.579, 0.009, 12625, '山西汾酒'],
        [0.534, 0.0018, 9335, '比亚迪'],
        [0.523, 0.003, 13006, '消费ETF'],
        [0.493, 0.0042, 13019, '食品饮料ETF'],
        [0.475, 0.0024, 7551, '地产ETF'],
        [0.472, 0.0035, 12824, '食品饮料ETF华安'],
        [0.43, -0.0052, 13948, '隆基绿能'],
        [0.364, -0.0047, 5676, '汽车ETF']
      ],
      itemStyle: {
        color: function(p) { return p.data[1] >= 0 ? accent2 : danger; },
        opacity: 0.7,
        shadowBlur: 10,
        shadowColor: 'rgba(88,166,255,0.3)'
      },
      label: {
        show: true,
        position: 'top',
        color: ink,
        fontSize: 11,
        formatter: function(p) { return p.data[3]; }
      }
    }],
    visualMap: {
      show: false,
      pieces: [
        { gte: 0, color: accent2 },
        { lt: 0, color: danger }
      ],
      dimension: 1
    }
  });

  // ===== Chart 4: Radar Chart =====
  var chart4 = echarts.init(document.getElementById('chart-radar'));
  chart4.setOption({
    tooltip: commonTooltip,
    legend: {
      bottom: 5,
      textStyle: { color: muted, fontSize: 11 },
      data: ['山西汾酒', '比亚迪', '隆基绿能', '消费ETF']
    },
    radar: {
      indicator: [
        { name: '趋势评分', max: 100 },
        { name: '风险评分', max: 100 },
        { name: '估值评分', max: 100 },
        { name: 'RSI', max: 100 },
        { name: 'ADX', max: 50 },
        { name: '5日涨幅', max: 10 },
        { name: '20日涨幅', max: 20 }
      ],
      center: ['50%', '48%'],
      radius: '62%',
      axisName: { color: muted, fontSize: 11 },
      splitArea: { areaStyle: { color: ['rgba(88,166,255,0.03)', 'rgba(88,166,255,0.06)'] } },
      splitLine: { lineStyle: { color: rule } },
      axisLine: { lineStyle: { color: rule } }
    },
    series: [{
      type: 'radar',
      areaStyle: { opacity: 0.1 },
      lineStyle: { width: 2 },
      data: [
        { value: [97.8, 7.6, 26.3, 62.5, 32, 4.26, 15.43], name: '山西汾酒', itemStyle: { color: accent2 } },
        { value: [45, 15, 40.2, 57.4, 27.1, -1.01, 17.13], name: '比亚迪', itemStyle: { color: accent } },
        { value: [65, 8.9, 50.9, 52, 14.3, 6.29, -3.35], name: '隆基绿能', itemStyle: { color: '#a371f7' } },
        { value: [90.6, 46.9, 27.9, 60.9, 28, 1.87, 9.36], name: '消费ETF', itemStyle: { color: warn } }
      ]
    }]
  });

  // ===== Chart 5: Method Comparison =====
  var chart5 = echarts.init(document.getElementById('chart-method-compare'));
  chart5.setOption({
    tooltip: Object.assign(commonTooltip, {
      trigger: 'axis',
      axisPointer: { type: 'shadow' }
    }),
    legend: {
      bottom: 5,
      textStyle: { color: muted, fontSize: 11 },
      data: ['历史模式匹配', '蒙特卡洛模拟', '综合概率']
    },
    grid: { left: '10%', right: '8%', top: '12%', bottom: '18%' },
    xAxis: {
      type: 'category',
      data: ['山西汾酒', '比亚迪', '消费ETF', '食品饮料ETF', '地产ETF', '食品饮料华安', '隆基绿能', '酒ETF', '汽车ETF'],
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 10, rotate: 30 }
    },
    yAxis: {
      type: 'value',
      min: 0.2,
      max: 0.7,
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11, formatter: function(v) { return (v * 100).toFixed(0) + '%'; } },
      splitLine: { lineStyle: { color: rule, opacity: 0.3 } }
    },
    series: [
      {
        name: '历史模式匹配',
        type: 'bar',
        barGap: 0,
        data: [0.625, 0.545, 0.55, 0.50, 0.467, 0.474, 0.407, 0.417, 0.308],
        itemStyle: { color: accent, borderRadius: [4, 4, 0, 0] }
      },
      {
        name: '蒙特卡洛模拟',
        type: 'bar',
        data: [0.51, 0.516, 0.483, 0.481, 0.489, 0.47, 0.464, 0.449, 0.448],
        itemStyle: { color: accent2, borderRadius: [4, 4, 0, 0] }
      },
      {
        name: '综合概率',
        type: 'line',
        data: [0.579, 0.534, 0.523, 0.493, 0.475, 0.472, 0.43, 0.429, 0.364],
        itemStyle: { color: warn },
        lineStyle: { width: 3, type: 'dashed' },
        symbol: 'circle',
        symbolSize: 8
      }
    ]
  });

  // Resize all charts on window resize
  window.addEventListener('resize', function() {
    chart1.resize();
    chart2.resize();
    chart3.resize();
    chart4.resize();
    chart5.resize();
  });
})();
