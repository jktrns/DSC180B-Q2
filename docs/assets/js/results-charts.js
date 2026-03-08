/**
 * results-charts.js
 * Renders Plotly charts and builds the per-query table from results.json.
 */
(function () {
  'use strict';

  var METHODS = ['Wide DP-SGD', 'Per-table DP-SGD', 'MST', 'PE Vanilla', 'PE Conditional'];
  var COLORS  = ['#8b6e5a', '#5a7e8b', '#7e8b5a', '#8b5a7e', '#5a8b6e'];

  function buildQueryTable(data) {
    var container = document.getElementById('queryTable');
    if (!container) return;

    var html = '<table class="fixed-cols"><colgroup>';
    html += '<col style="width:15%"><col style="width:10%">';
    METHODS.forEach(function() { html += '<col style="width:15%">'; });
    html += '</colgroup><thead><tr>';
    html += '<th>Query</th><th>Type</th>';
    METHODS.forEach(function(m) { html += '<th>' + m.replace('Wide DP-SGD', 'Wide DP-VAE').replace('PE Vanilla', 'PE (Vanilla)').replace('PE Conditional', 'PE (Conditional)') + '</th>'; });
    html += '</tr></thead><tbody>';

    data.forEach(function(q) {
      html += '<tr>';
      html += '<td title="' + (q.desc || '') + '">' + q.name + '</td>';
      html += '<td>' + (q.type || '') + '</td>';
      METHODS.forEach(function(m) {
        var s = (q.scores || []).find(function(s) { return s.method === m; });
        if (!s || s.value === null) {
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
    var el = document.getElementById('chartPassRate');
    if (!el || typeof Plotly === 'undefined') return;

    var counts = {};
    var totals = {};
    METHODS.forEach(function(m) { counts[m] = 0; totals[m] = 0; });

    data.forEach(function(q) {
      (q.scores || []).forEach(function(s) {
        if (s.value !== null) {
          totals[s.method] = (totals[s.method] || 0) + 1;
          if (s.note === 'pass') counts[s.method] = (counts[s.method] || 0) + 1;
        }
      });
    });

    var labels = METHODS.map(function(m) { return m.replace('Wide DP-SGD', 'Wide DP-VAE').replace('PE Vanilla', 'PE (Vanilla)').replace('PE Conditional', 'PE (Cond.)'); });
    var values = METHODS.map(function(m) { return counts[m] || 0; });
    var text = METHODS.map(function(m) { return (counts[m] || 0) + '/' + (totals[m] || 0); });

    Plotly.newPlot(el, [{
      x: labels, y: values, type: 'bar', text: text, textposition: 'outside',
      marker: { color: COLORS }
    }], {
      yaxis: { title: 'Queries passed', range: [0, Math.max.apply(null, values) + 2], gridcolor: '#2e2e2e', color: '#a8a49a' },
      xaxis: { color: '#a8a49a' },
      margin: { t: 20, r: 20, b: 60, l: 50 },
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: { family: "'Open Sans', 'Helvetica Neue', Arial, sans-serif", color: '#e0ddd5' }
    }, { responsive: true, displayModeBar: false });
  }

  function renderByTypeChart(data) {
    var el = document.getElementById('chartByType');
    if (!el || typeof Plotly === 'undefined') return;

    var typeSet = {};
    data.forEach(function(q) { if (q.type) typeSet[q.type] = true; });
    var types = Object.keys(typeSet);

    var traces = METHODS.map(function(m, i) {
      var yVals = types.map(function(t) {
        var qs = data.filter(function(q) { return q.type === t; });
        var scores = qs.map(function(q) {
          var s = (q.scores || []).find(function(s) { return s.method === m; });
          return s && s.value !== null ? s.value : null;
        }).filter(function(v) { return v !== null; });
        return scores.length ? scores.reduce(function(a, b) { return a + b; }, 0) / scores.length : null;
      });
      return {
        x: types, y: yVals, name: m.replace('Wide DP-SGD', 'Wide DP-VAE').replace('PE Vanilla', 'PE (Vanilla)').replace('PE Conditional', 'PE (Cond.)'),
        type: 'bar', marker: { color: COLORS[i] }
      };
    });

    Plotly.newPlot(el, traces, {
      barmode: 'group',
      yaxis: { title: 'Avg score (0\u20131)', range: [0, 1], gridcolor: '#2e2e2e', color: '#a8a49a' },
      xaxis: { color: '#a8a49a' },
      legend: { orientation: 'h', y: -0.25, font: { color: '#a8a49a' } },
      margin: { t: 20, r: 20, b: 80, l: 50 },
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: { family: "'Open Sans', 'Helvetica Neue', Arial, sans-serif", color: '#e0ddd5' }
    }, { responsive: true, displayModeBar: false });
  }

  // Initialize when DOM ready
  document.addEventListener('DOMContentLoaded', function() {
    fetch('assets/data/results.json')
      .then(function(r) { return r.json(); })
      .then(function(data) {
        buildQueryTable(data);
        renderPassRateChart(data);
        renderByTypeChart(data);
      })
      .catch(function(e) { console.error('Failed to load results.json', e); });
  });
})();
