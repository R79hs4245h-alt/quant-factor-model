(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var green = style.getPropertyValue('--green').trim();
  var red = style.getPropertyValue('--red').trim();
  var yellow = style.getPropertyValue('--yellow').trim();
  var purple = style.getPropertyValue('--purple').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  // ===== Chart 1: Holdings Bar+Line =====
  var chart1 = echarts.init(document.getElementById('chart-holdings'));
  chart1.setOption({
    title: { text: '持仓权重与盈利概率', left: 'center', textStyle: { color: ink, fontSize: 14 } },
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    legend: { data: ['权重(%)', '盈利概率(%)', '预期收益(%)'], bottom: 0, textStyle: { color: muted } },
    grid: { left: '3%', right: '4%', bottom: '12%', top: '15%', containLabel: true },
    xAxis: {
      type: 'category',
      data: ['消费ETF', '食品饮料ETF', '食品饮料华安', '地产ETF', '汽车ETF'],
      axisLabel: { color: muted, fontSize: 11, rotate: 20 },
      axisLine: { lineStyle: { color: rule } }
    },
    yAxis: [
      { type: 'value', name: '权重/概率(%)', axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule, opacity: 0.3 } } },
      { type: 'value', name: '预期收益(%)', axisLabel: { color: muted, formatter: '{value}%' }, splitLine: { show: false } }
    ],
    series: [
      { name: '权重(%)', type: 'bar', data: [25.0, 25.0, 24.6, 14.5, 11.0], itemStyle: { color: accent }, barWidth: '30%' },
      { name: '盈利概率(%)', type: 'bar', data: [52.3, 49.3, 47.2, 47.5, 36.4], itemStyle: { color: accent2 }, barWidth: '30%' },
      { name: '预期收益(%)', type: 'line', yAxisIndex: 1, data: [0.30, 0.42, 0.35, 0.24, -0.47],
        itemStyle: { color: function(p) { return p.value >= 0 ? green : red; } },
        lineStyle: { width: 2 }, symbolSize: 8,
        label: { show: true, position: 'top', color: muted, fontSize: 10, formatter: function(p) { return (p.value > 0 ? '+' : '') + p.value.toFixed(2) + '%'; } }
      }
    ]
  });

  // ===== Chart 2: Radar =====
  var chart2 = echarts.init(document.getElementById('chart-radar'));
  chart2.setOption({
    title: { text: '三重筛选评分雷达图', left: 'center', textStyle: { color: ink, fontSize: 14 } },
    tooltip: { trigger: 'item' },
    legend: { bottom: 0, textStyle: { color: muted, fontSize: 10 } },
    radar: {
      indicator: [
        { name: '风险评分', max: 100 },
        { name: '趋势评分', max: 100 },
        { name: '估值评分', max: 100 },
        { name: '盈利概率', max: 100 }
      ],
      center: ['50%', '50%'],
      radius: '60%',
      axisName: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: rule, opacity: 0.3 } },
      splitArea: { areaStyle: { color: [bg2, 'transparent'] } },
      axisLine: { lineStyle: { color: rule } }
    },
    series: [{
      type: 'radar',
      data: [
        { value: [46.9, 90.6, 27.9, 52.3], name: '消费ETF 510630', itemStyle: { color: accent }, areaStyle: { opacity: 0.15 } },
        { value: [48.8, 86.3, 33.7, 49.3], name: '食品饮料ETF 515170', itemStyle: { color: accent2 }, areaStyle: { opacity: 0.1 } },
        { value: [50.9, 66.2, 34.1, 47.2], name: '食品饮料华安 516900', itemStyle: { color: green }, areaStyle: { opacity: 0.1 } },
        { value: [17.2, 50.2, 43.8, 47.5], name: '地产ETF 159707', itemStyle: { color: yellow }, areaStyle: { opacity: 0.1 } },
        { value: [16.7, 52.5, 47.8, 36.4], name: '汽车ETF 159512', itemStyle: { color: purple }, areaStyle: { opacity: 0.1 } }
      ]
    }]
  });

  // ===== Chart 3: Allocation Pie =====
  var chart3 = echarts.init(document.getElementById('chart-alloc'));
  chart3.setOption({
    title: { text: '资金分配(元)', left: 'center', textStyle: { color: ink, fontSize: 14 } },
    tooltip: { trigger: 'item', formatter: '{b}: {c}元 ({d}%)' },
    legend: { bottom: 0, textStyle: { color: muted, fontSize: 10 } },
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      center: ['50%', '45%'],
      data: [
        { value: 23722, name: '消费ETF 510630', itemStyle: { color: accent } },
        { value: 23735, name: '食品饮料ETF 515170', itemStyle: { color: accent2 } },
        { value: 23303, name: '食品饮料华安 516900', itemStyle: { color: green } },
        { value: 13695, name: '地产ETF 159707', itemStyle: { color: yellow } },
        { value: 10389, name: '汽车ETF 159512', itemStyle: { color: purple } },
        { value: 5156, name: '现金保留', itemStyle: { color: muted } }
      ],
      label: { color: ink, fontSize: 10, formatter: '{b}\n{c}元' }
    }]
  });

  // Resize
  window.addEventListener('resize', function() {
    chart1.resize();
    chart2.resize();
    chart3.resize();
  });
})();
