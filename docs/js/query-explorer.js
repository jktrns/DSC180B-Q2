(function () {
  'use strict';

  const METHODS = [
    { key: 'widetable', label: 'Wide-table DP-VAE' },
    { key: 'pertable', label: 'Per-table DP-SGD' },
    { key: 'mst', label: 'MST' },
    { key: 'pe', label: 'Private Evolution' },
  ];

  const FILES = {
    widetable: { summary: 'evaluation_widetable.csv', detail: 'evaluation_widetable_detail.csv' },
    pertable: { summary: 'evaluation_pertable.csv', detail: 'evaluation_pertable_detail.csv' },
    mst: { summary: 'evaluation_mst.csv', detail: 'evaluation_mst_detail.csv' },
    pe: { summary: 'evaluation_pe.csv', detail: 'evaluation_pe_detail.csv' },
  };

  const DATA_BASE_CANDIDATES = [
    '../data/results',   // when serving repo root and opening /docs/query-explorer.html
    'data/results',      // fallback for alternate static hosting layouts
  ];

  async function fetchText(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`Failed to load ${path}`);
    return res.text();
  }

  async function fetchTextWithBase(fileName) {
    let lastError = null;
    for (const base of DATA_BASE_CANDIDATES) {
      const path = `${base}/${fileName}`;
      try {
        return await fetchText(path);
      } catch (err) {
        lastError = err;
      }
    }
    throw lastError || new Error(`Could not load ${fileName}`);
  }

  function parseCSV(text) {
    const rows = [];
    let row = [];
    let cell = '';
    let i = 0;
    let inQuotes = false;

    while (i < text.length) {
      const ch = text[i];
      const next = text[i + 1];

      if (inQuotes) {
        if (ch === '"' && next === '"') {
          cell += '"';
          i += 2;
          continue;
        }
        if (ch === '"') {
          inQuotes = false;
          i += 1;
          continue;
        }
        cell += ch;
        i += 1;
        continue;
      }

      if (ch === '"') {
        inQuotes = true;
        i += 1;
        continue;
      }
      if (ch === ',') {
        row.push(cell);
        cell = '';
        i += 1;
        continue;
      }
      if (ch === '\n') {
        row.push(cell);
        rows.push(row);
        row = [];
        cell = '';
        i += 1;
        continue;
      }
      if (ch === '\r') {
        i += 1;
        continue;
      }

      cell += ch;
      i += 1;
    }

    if (cell.length || row.length) {
      row.push(cell);
      rows.push(row);
    }

    if (!rows.length) return [];
    const headers = rows[0];
    return rows.slice(1).filter((r) => r.length && r.some((c) => c !== '')).map((r) => {
      const out = {};
      headers.forEach((h, idx) => {
        out[h] = r[idx] ?? '';
      });
      return out;
    });
  }

  function toNumber(value) {
    if (value === '' || value == null) return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function formatScore(value) {
    if (value == null) return '—';
    return value.toFixed(3);
  }

  function escapeHTML(s) {
    return String(s)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  function chartCell(scores) {
    const vals = scores.filter((x) => x != null);
    if (!vals.length) return '<span class="small">No scores available</span>';
    const max = Math.max(...vals, 1);
    const bars = METHODS.map((m) => {
      const v = scoresByMethod(scores, m.key);
      const width = v == null ? 0 : Math.max(2, (v / max) * 100);
      return `<div class="qe-bar-row"><span>${m.label}</span><div class="qe-bar-track"><div class="qe-bar" style="width:${width}%"></div></div><strong>${formatScore(v)}</strong></div>`;
    }).join('');
    return `<div class="qe-bars">${bars}</div>`;
  }

  function scoresByMethod(scoreObj, key) {
    return scoreObj[key] ?? null;
  }

  async function initQueryResultsExplorer() {
    const body = document.getElementById('qeBody');
    const search = document.getElementById('qeSearch');
    const typeFilter = document.getElementById('qeTypeFilter');
    const sort = document.getElementById('qeSort');
    const summary = document.getElementById('qeSummary');

    if (!body || !search || !typeFilter || !sort || !summary) return;

    try {
      const loaded = {};
      for (const m of METHODS) {
        const [summaryCsv, detailCsv] = await Promise.all([
          fetchTextWithBase(FILES[m.key].summary),
          fetchTextWithBase(FILES[m.key].detail),
        ]);
        loaded[m.key] = {
          summary: parseCSV(summaryCsv),
          detail: parseCSV(detailCsv),
        };
      }

      const byQuery = new Map();

      for (const m of METHODS) {
        for (const row of loaded[m.key].summary) {
          const query = row.query;
          if (!query) continue;
          if (!byQuery.has(query)) {
            byQuery.set(query, {
              query,
              type: row.type || 'unknown',
              scores: {},
              passed: {},
              errors: {},
              details: {},
            });
          }
          const entry = byQuery.get(query);
          entry.type = entry.type || row.type || 'unknown';
          entry.scores[m.key] = toNumber(row.score);
          entry.passed[m.key] = row.passed;
          entry.errors[m.key] = row.error || '';
        }

        for (const d of loaded[m.key].detail) {
          const query = d.query;
          if (!query) continue;
          if (!byQuery.has(query)) {
            byQuery.set(query, {
              query,
              type: d.query_type || 'unknown',
              scores: {},
              passed: {},
              errors: {},
              details: {},
            });
          }
          const entry = byQuery.get(query);
          if (!entry.details[m.key]) entry.details[m.key] = [];
          entry.details[m.key].push(d);
        }
      }

      let rows = Array.from(byQuery.values());
      rows.forEach((r) => {
        const vals = METHODS.map((m) => r.scores[m.key]).filter((v) => v != null);
        r.avgScore = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
      });

      const types = Array.from(new Set(rows.map((r) => r.type).filter(Boolean))).sort();
      types.forEach((t) => {
        const opt = document.createElement('option');
        opt.value = t;
        opt.textContent = t;
        typeFilter.appendChild(opt);
      });

      function applyFilters() {
        const q = search.value.trim().toLowerCase();
        const t = typeFilter.value;
        const sortKey = sort.value;

        let filtered = rows.filter((r) => {
          const matchText = !q || r.query.toLowerCase().includes(q);
          const matchType = t === 'all' || r.type === t;
          return matchText && matchType;
        });

        filtered.sort((a, b) => {
          const cmpNumDesc = (x, y) => (y ?? -Infinity) - (x ?? -Infinity);
          const cmpNumAsc = (x, y) => (x ?? Infinity) - (y ?? Infinity);
          switch (sortKey) {
            case 'query_desc':
              return b.query.localeCompare(a.query);
            case 'avg_desc':
              return cmpNumDesc(a.avgScore, b.avgScore);
            case 'avg_asc':
              return cmpNumAsc(a.avgScore, b.avgScore);
            case 'mst_desc':
              return cmpNumDesc(a.scores.mst, b.scores.mst);
            case 'pertable_desc':
              return cmpNumDesc(a.scores.pertable, b.scores.pertable);
            case 'widetable_desc':
              return cmpNumDesc(a.scores.widetable, b.scores.widetable);
            case 'pe_desc':
              return cmpNumDesc(a.scores.pe, b.scores.pe);
            case 'query_asc':
            default:
              return a.query.localeCompare(b.query);
          }
        });

        summary.textContent = `${filtered.length} queries shown (${rows.length} total from evaluation artifacts).`;

        body.innerHTML = '';

        filtered.forEach((r) => {
          const tr = document.createElement('tr');
          tr.className = 'qe-row';
          tr.innerHTML = `
            <td><button class="qe-expand" aria-expanded="false">+ ${escapeHTML(r.query)}</button></td>
            <td>${escapeHTML(r.type || 'unknown')}</td>
            <td>${formatScore(r.scores.widetable)}</td>
            <td>${formatScore(r.scores.pertable)}</td>
            <td>${formatScore(r.scores.mst)}</td>
            <td>${formatScore(r.scores.pe)}</td>
            <td>${formatScore(r.avgScore)}</td>
          `;

          const detailsTr = document.createElement('tr');
          detailsTr.className = 'qe-details-row';
          detailsTr.hidden = true;

          const detailParts = METHODS.map((m) => {
            const methodDetails = r.details[m.key] || [];
            const error = r.errors[m.key];
            const top = methodDetails.slice(0, 5).map((d) => {
              return `<li><code>${escapeHTML(d.column || '')}</code> · ${escapeHTML(d.metric_type || '')} = <strong>${escapeHTML(d.value || '')}</strong> (${escapeHTML(d.passed || '')})</li>`;
            }).join('');
            return `
              <div class="qe-detail-card">
                <h4>${m.label}</h4>
                <p class="small">Score: <strong>${formatScore(r.scores[m.key])}</strong>${error ? ` · Error: ${escapeHTML(error)}` : ''}</p>
                <ul class="qe-list">${top || '<li>No metric-level detail rows available.</li>'}</ul>
              </div>
            `;
          }).join('');

          detailsTr.innerHTML = `<td colspan="7"><div class="qe-details-grid">${detailParts}</div><div class="qe-chart">${chartCell(r.scores)}</div></td>`;

          tr.querySelector('.qe-expand').addEventListener('click', (e) => {
            const btn = e.currentTarget;
            const expanded = btn.getAttribute('aria-expanded') === 'true';
            btn.setAttribute('aria-expanded', String(!expanded));
            btn.textContent = `${expanded ? '+' : '−'} ${r.query}`;
            detailsTr.hidden = expanded;
          });

          body.appendChild(tr);
          body.appendChild(detailsTr);
        });
      }

      search.addEventListener('input', applyFilters);
      typeFilter.addEventListener('change', applyFilters);
      sort.addEventListener('change', applyFilters);
      applyFilters();
    } catch (err) {
      summary.textContent = `Could not load explorer artifacts: ${err.message}`;
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    if (document.body.getAttribute('data-page') === 'query-explorer') {
      initQueryResultsExplorer();
    }
  });
})();
