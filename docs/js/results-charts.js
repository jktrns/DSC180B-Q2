/**
 * results-charts.js
 * Renders Plotly charts on the Results page and builds the per-query table.
 */
(function () {
  'use strict';

  const METHODS = ['Wide DP-SGD', 'Per-table DP-SGD', 'MST', 'Private Evolution'];
  const COLORS  = ['#6366f1', '#2563eb', '#16a34a', '#d97706'];

  document.addEventListener('DOMContentLoaded', async () => {
    const page = document.body.getAttribute('data-page');
    if (page !== 'results') return;

    let data = [];
    try {
      const res = await fetch('data/results.json');
      data = await res.json();
    } catch (e) {
      console.error('Failed to load results.json', e);
      return;
    }

    buildQueryTable(data);
    renderPassRateChart(data);
    renderByTypeChart(data);
  });

  function buildQueryTable(data) {
    const container = document.getElementById('queryTable');
    if (!container) return;

    let html = '<table><thead><tr>';
    html += '<th>Query</th><th>Type</th>';
    METHODS.forEach(m => { html += '<th>' + m + '</th>'; });
    html += '</tr></thead><tbody>';

    data.forEach(q => {
      html += '<tr>';
      html += '<td title="' + (q.desc || '') + '">' + q.name + '</td>';
      html += '<td>' + (q.type || '') + '</td>';
      METHODS.forEach(m => {
        const s = (q.scores || []).find(s => s.method === m);
        if (!s) {
          html += '<td class="cell-na">N/A</td>';
        } else if (s.value === null) {
          html += '<td class="cell-na">N/A</td>';
        } else if (s.note === 'pass') {
          html += '<td class="cell-pass">' + s.value.toFixed(3) + '</td>';
        } else if (s.note === 'partial') {
          html += '<td class="cell-partial">' + s.value.toFixed(3) + '</td>';
        } else {
          html += '<td class="cell-fail">' + s.value.toFixed(3) + '</td>';
        }
      });
      html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;
  }

  function renderPassRateChart(data) {
    const el = document.getElementById('chartPassRate');
    if (!el) return;

    const counts = {};
    const totals = {};
    METHODS.forEach(m => { counts[m] = 0; totals[m] = 0; });

    data.forEach(q => {
      (q.scores || []).forEach(s => {
        if (s.value !== null) {
          totals[s.method] = (totals[s.method] || 0) + 1;
          if (s.note === 'pass') counts[s.method] = (counts[s.method] || 0) + 1;
        }
      });
    });

    const labels = METHODS.map(m => m.replace('Wide DP-SGD', 'Wide DP-VAE'));
    const values = METHODS.map(m => counts[m] || 0);
    const text = METHODS.map(m => (counts[m] || 0) + '/' + (totals[m] || 0));

    Plotly.newPlot(el, [{
      x: labels, y: values, type: 'bar', text: text, textposition: 'outside',
      marker: { color: COLORS }
    }], {
      yaxis: { title: 'Queries Passed', range: [0, Math.max(...values) + 2] },
      margin: { t: 20, r: 20, b: 60, l: 50 },
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent'
    }, { responsive: true, displayModeBar: false });
  }

  function renderByTypeChart(data) {
    const el = document.getElementById('chartByType');
    if (!el) return;

    const types = [...new Set(data.map(q => q.type).filter(Boolean))];
    const traces = METHODS.map((m, i) => {
      const yVals = types.map(t => {
        const qs = data.filter(q => q.type === t);
        const scores = qs.map(q => {
          const s = (q.scores || []).find(s => s.method === m);
          return s && s.value !== null ? s.value : null;
        }).filter(v => v !== null);
        return scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
      });
      return {
        x: types, y: yVals, name: m.replace('Wide DP-SGD', 'Wide DP-VAE'),
        type: 'bar', marker: { color: COLORS[i] }
      };
    });

    Plotly.newPlot(el, traces, {
      barmode: 'group',
      yaxis: { title: 'Avg Score (0-1)', range: [0, 1] },
      legend: { orientation: 'h', y: -0.25 },
      margin: { t: 20, r: 20, b: 80, l: 50 },
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent'
    }, { responsive: true, displayModeBar: false });
  }
})();
