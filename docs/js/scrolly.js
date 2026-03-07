/**
 * scrolly.js
 * Scrollytelling module for the explore.html page.
 * Uses IntersectionObserver to detect active step and update sticky panel.
 */

(function () {
  'use strict';

  const steps = document.querySelectorAll('.step');
  const panel = document.getElementById('panelContent');
  const progressText = document.querySelector('.progress-text');
  const progressBar = document.querySelector('.progress-bar');
  let currentStep = null;
  let histogramInitialized = false;

  if (!steps.length || !panel) {
    console.warn('Scrollytelling: .step or #panelContent not found');
    return;
  }

  const totalSteps = steps.length;

  /**
   * Update progress indicator
   */
  function updateProgress(index) {
    if (progressText) {
      progressText.textContent = `Step ${index + 1} of ${totalSteps}`;
    }
    if (progressBar) {
      const percent = ((index + 1) / totalSteps) * 100;
      progressBar.style.height = percent + '%';
    }
  }

  /**
   * Set panel content based on step ID
   */
  function setPanel(stepId) {
    let html = '';

    switch (stepId) {
      case 'step-hero':
        html = `
          <div class="compare-panel">
            <h4 style="margin:0 0 8px;">Project Overview</h4>
            <p class="small" style="margin:0;">This section compares four differentially private synthetic data methods across 21 benchmark SQL queries. Scroll to inspect distributions, per-query comparisons, and final scorecards.</p>
          </div>
        `;
        break;

      case 'step-queries':
        html = `
          <div class="benchmark-slider-panel">
            <div class="benchmark-slider-controls">
              <button type="button" id="benchPrev" class="bench-nav-btn">←</button>
              <div style="width: 100%;">
                <label for="benchRange" class="small"><strong>Query:</strong> <span id="benchLabel">Loading…</span></label>
                <input id="benchRange" type="range" min="0" max="0" value="0" step="1" />
              </div>
              <button type="button" id="benchNext" class="bench-nav-btn">→</button>
            </div>
            <div class="benchmark-detail" id="benchDetail">
              <p class="small">Loading benchmark results…</p>
            </div>
          </div>
        `;
        break;

      case 'step-histogram':
        html = `
          <div class="histogram-panel">
            <div style="margin-bottom: 12px;">
              <label for="histogramDataset" style="display: block; margin-bottom: 6px;"><strong>Dataset:</strong></label>
              <select id="histogramDataset" style="width: 100%; padding: 6px; border: 1px solid var(--border); border-radius: 6px;">
                <option value="popular_browsers">Browser Distribution</option>
                <option value="ram_utilization">RAM Utilization</option>
                <option value="display_vendors">Display Vendors</option>
              </select>
            </div>
            <div style="margin-bottom: 12px;">
              <label for="histogramMethod" style="display: block; margin-bottom: 6px;"><strong>Method:</strong></label>
              <select id="histogramMethod" style="width: 100%; padding: 6px; border: 1px solid var(--border); border-radius: 6px;">
                <option value="MST">MST</option>
                <option value="PerTable_DPSGD">Per-Table DP-SGD</option>
                <option value="Wide_DPVAE">Wide-Table DP-VAE</option>
                <option value="PrivateEvolution">Private Evolution</option>
              </select>
            </div>
            <div style="margin-bottom: 12px;">
              <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
                <input type="checkbox" id="histogramNormalize" />
                <span>Normalize (probability)</span>
              </label>
            </div>
            <div id="histogramChart" style="width: 100%; height: 300px; margin-top: 12px;"></div>
            <p class="small" id="histogramMetric" style="margin-top: 8px;"></p>
          </div>
        `;
        break;

      case 'step-compare':
        html = `
          <div class="compare-panel">
            <h4 style="margin: 0 0 12px;">Per-query Method Comparison</h4>
            <div style="margin-bottom: 10px;">
              <label for="compareQuerySelect" class="small"><strong>Query:</strong></label>
              <select id="compareQuerySelect" style="width: 100%; padding: 8px; border: 1px solid var(--border); border-radius: 8px;"></select>
            </div>
            <div id="compareQueryMeta" class="small" style="margin-bottom: 8px;"></div>
            <div id="compareQueryBars" class="bench-bars"></div>
            <p class="caption" style="margin-top: 8px;">Bars are generated from committed evaluation score artifacts (no placeholder images).</p>
          </div>
        `;
        break;

      case 'step-scoreboard':
        html = `
          <div class="scoreboard-panel">
            <div style="display: grid; grid-template-columns: 1fr; gap: 12px;">
              <div class="score-card" style="text-align: center; padding: 16px; border: 1px solid var(--border); border-radius: 8px; background: var(--card);">
                <h4 style="margin: 0 0 6px;">Per-Table DP-SGD</h4>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0; color: #2563eb;">6/21</p>
                <p class="small" style="margin: 4px 0 0;">Queries passed</p>
              </div>
              <div class="score-card" style="text-align: center; padding: 16px; border: 1px solid var(--border); border-radius: 8px; background: var(--card);">
                <h4 style="margin: 0 0 6px;">MST (Marginal)</h4>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0; color: #2563eb;">6/21</p>
                <p class="small" style="margin: 4px 0 0;">Queries passed</p>
              </div>
              <div class="score-card" style="text-align: center; padding: 16px; border: 1px solid var(--border); border-radius: 8px; background: var(--card);">
                <h4 style="margin: 0 0 6px;">Wide-Table DP-VAE</h4>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0; color: #dc2626;">1/8</p>
                <p class="small" style="margin: 4px 0 0;">Evaluated subset</p>
              </div>
              <div class="score-card" style="text-align: center; padding: 16px; border: 1px solid var(--border); border-radius: 8px; background: var(--card);">
                <h4 style="margin: 0 0 6px;">Private Evolution</h4>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0; color: #dc2626;">2/8</p>
                <p class="small" style="margin: 4px 0 0;">Evaluated subset</p>
              </div>
            </div>
            <p class="small" style="margin-top: 12px; text-align: center;">Simple distributions pass; complex joins remain hard.</p>
          </div>
        `;
        break;

      default:
        html = `<div class="figure-placeholder"><p>Loading...</p></div>`;
    }

    panel.innerHTML = html;

    // Initialize components if not already done
    if (stepId === 'step-queries') {
      setTimeout(() => {
        window.initBenchmarkQueryExplorer && window.initBenchmarkQueryExplorer();
      }, 100);
    }

    if (stepId === 'step-histogram' && !histogramInitialized) {
      histogramInitialized = true;
      setTimeout(() => {
        window.initHistogramPanel && window.initHistogramPanel();
      }, 100);
    }

    if (stepId === 'step-compare') {
      setTimeout(() => {
        window.initComparePanel && window.initComparePanel();
      }, 100);
    }
  }

  /**
   * Determine active step by largest visible ratio in viewport.
   */
  function refreshActiveStep() {
    let bestStep = null;
    let bestRatio = -1;

    steps.forEach((step) => {
      const rect = step.getBoundingClientRect();
      const visible = Math.max(0, Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top, 0));
      const ratio = rect.height > 0 ? visible / rect.height : 0;
      if (ratio > bestRatio) {
        bestRatio = ratio;
        bestStep = step;
      }
    });

    if (!bestStep || bestRatio < 0.15) return;

    const stepId = bestStep.dataset.step;
    if (stepId !== currentStep) {
      currentStep = stepId;
      const stepIndex = Array.from(steps).indexOf(bestStep);
      setPanel(stepId);
      updateProgress(stepIndex);
    }

    steps.forEach((step) => {
      step.classList.toggle('active', step === bestStep);
    });
  }

  window.addEventListener('scroll', refreshActiveStep, { passive: true });
  window.addEventListener('resize', refreshActiveStep);

  // Initialize first step manually
  if (steps.length > 0) {
    const firstStepId = steps[0].dataset.step;
    setPanel(firstStepId);
    updateProgress(0);
    steps[0].classList.add('active');
    requestAnimationFrame(refreshActiveStep);
  }

  window.initBenchmarkQueryExplorer = function () {
    const rangeEl = document.getElementById('benchRange');
    const labelEl = document.getElementById('benchLabel');
    const detailEl = document.getElementById('benchDetail');
    const prevEl = document.getElementById('benchPrev');
    const nextEl = document.getElementById('benchNext');

    if (!rangeEl || !labelEl || !detailEl || !prevEl || !nextEl) return;

    const METHOD_ORDER = [
      ['widetable', 'Wide-table DP-VAE'],
      ['pertable', 'Per-table DP-SGD'],
      ['mst', 'MST (Marginal)'],
      ['pe', 'Private Evolution'],
    ];

    const fmt = (v) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(3) : '—');

    fetch('data/benchmark_slider.json')
      .then((r) => {
        if (!r.ok) throw new Error('data/benchmark_slider.json');
        return r.json();
      })
      .then((records) => {
        if (!Array.isArray(records) || !records.length) {
          detailEl.innerHTML = '<p class="small">No benchmark records found.</p>';
          return;
        }

        rangeEl.min = '0';
        rangeEl.max = String(records.length - 1);
        rangeEl.value = '0';

        const render = (idx) => {
          const i = Math.max(0, Math.min(records.length - 1, idx));
          const rec = records[i];
          labelEl.textContent = `${i + 1} of ${records.length} (${rec.query})`;

          const values = METHOD_ORDER.map(([key]) => rec.scores?.[key]).filter((v) => typeof v === 'number');
          const max = values.length ? Math.max(...values, 1) : 1;

          const best = METHOD_ORDER
            .map(([key, label]) => ({ label, value: rec.scores?.[key] }))
            .filter((x) => typeof x.value === 'number')
            .sort((a, b) => b.value - a.value)[0];
          const bestText = best ? `Best method: ${best.label} (${best.value.toFixed(3)})` : 'Best method: not available';

          const passCount = METHOD_ORDER.filter(([key]) => rec.passed?.[key] === true).length;
          const scoreList = METHOD_ORDER.map(([key, label]) => {
            const err = rec.errors?.[key] ? ` · ${rec.errors[key]}` : '';
            return `<li><strong>${label}:</strong> ${fmt(rec.scores?.[key])}${err}</li>`;
          }).join('');

          const bars = METHOD_ORDER.map(([key, label]) => {
            const v = rec.scores?.[key];
            const width = typeof v === 'number' ? Math.max(2, (v / max) * 100) : 0;
            return `<div class="bench-bar-row"><span>${label}</span><div class="bench-bar-track"><div class="bench-bar" style="width:${width}%"></div></div><strong>${fmt(v)}</strong></div>`;
          }).join('');

          detailEl.innerHTML = `
            <h4 style="margin:0 0 6px;">${rec.query}</h4>
            <p class="small" style="margin:0 0 6px;"><strong>Type:</strong> ${rec.type || 'unknown'}</p>
            <p class="small" style="margin:0 0 10px;">${bestText} · Pass across methods: ${passCount}/4</p>
            <ul class="bench-score-list">${scoreList}</ul>
            <div class="bench-bars">${bars}</div>
          `;
        };

        rangeEl.addEventListener('input', () => render(Number(rangeEl.value)));
        prevEl.addEventListener('click', () => {
          rangeEl.value = String(Math.max(0, Number(rangeEl.value) - 1));
          render(Number(rangeEl.value));
        });
        nextEl.addEventListener('click', () => {
          rangeEl.value = String(Math.min(records.length - 1, Number(rangeEl.value) + 1));
          render(Number(rangeEl.value));
        });

        render(0);
      })
      .catch((e) => {
        detailEl.innerHTML = `<p class="small">Could not load slider data: ${e.message}</p>`;
      });
  };

  window.initComparePanel = function () {
    const selectEl = document.getElementById('compareQuerySelect');
    const metaEl = document.getElementById('compareQueryMeta');
    const barsEl = document.getElementById('compareQueryBars');
    if (!selectEl || !metaEl || !barsEl) return;

    const METHODS = [
      { key: 'widetable', label: 'Wide-table DP-VAE' },
      { key: 'pertable', label: 'Per-table DP-SGD' },
      { key: 'mst', label: 'MST (Marginal)' },
      { key: 'pe', label: 'Private Evolution' },
    ];
    const FILES = {
      widetable: 'evaluation_widetable.csv',
      pertable: 'evaluation_pertable.csv',
      mst: 'evaluation_mst.csv',
      pe: 'evaluation_pe.csv',
    };
    const BASES = ['../data/results', 'data/results', './data/results', '/data/results'];

    const parseCSV = (text) => {
      const rows = [];
      let row = [];
      let cell = '';
      let i = 0;
      let inQuotes = false;
      while (i < text.length) {
        const ch = text[i];
        const next = text[i + 1];
        if (inQuotes) {
          if (ch === '"' && next === '"') { cell += '"'; i += 2; continue; }
          if (ch === '"') { inQuotes = false; i += 1; continue; }
          cell += ch; i += 1; continue;
        }
        if (ch === '"') { inQuotes = true; i += 1; continue; }
        if (ch === ',') { row.push(cell); cell = ''; i += 1; continue; }
        if (ch === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; i += 1; continue; }
        if (ch !== '\r') { cell += ch; }
        i += 1;
      }
      if (cell.length || row.length) { row.push(cell); rows.push(row); }
      if (!rows.length) return [];
      const headers = rows[0];
      return rows.slice(1).filter((r) => r.some((c) => c !== '')).map((r) => {
        const out = {};
        headers.forEach((h, idx) => { out[h] = r[idx] ?? ''; });
        return out;
      });
    };

    const toNum = (v) => {
      if (v === '' || v == null) return null;
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    };

    const fetchCsv = async (name) => {
      let last = null;
      for (const b of BASES) {
        try {
          const res = await fetch(`${b}/${name}`);
          if (!res.ok) throw new Error(`${b}/${name}`);
          return parseCSV(await res.text());
        } catch (e) { last = e; }
      }
      throw last || new Error(name);
    };

    const fetchQuestion = async (queryId) => {
      try {
        const res = await fetch(`queries/${queryId}.json`);
        if (!res.ok) return null;
        const d = await res.json();
        const row = Array.isArray(d) ? d[0] : d;
        return row && row.question ? row.question : null;
      } catch { return null; }
    };

    Promise.all(METHODS.map((m) => fetchCsv(FILES[m.key])))
      .then(async (tables) => {
        const methodRows = {};
        METHODS.forEach((m, i) => { methodRows[m.key] = tables[i]; });
        const sets = METHODS.map((m) => new Set((methodRows[m.key] || []).map((r) => r.query).filter(Boolean)));
        let common = sets[0];
        for (let i = 1; i < sets.length; i++) common = new Set([...common].filter((q) => sets[i].has(q)));

        const records = [];
        for (const query of [...common].sort()) {
          const rec = { query, type: '', scores: {}, question: null };
          METHODS.forEach((m) => {
            const row = methodRows[m.key].find((r) => r.query === query) || {};
            rec.type = rec.type || row.type || '';
            rec.scores[m.key] = toNum(row.score);
          });
          rec.question = await fetchQuestion(query);
          records.push(rec);
        }

        if (!records.length) {
          metaEl.textContent = 'No overlapping query IDs available in all four method CSVs.';
          return;
        }

        records.forEach((r, i) => {
          const opt = document.createElement('option');
          opt.value = String(i);
          opt.textContent = r.query;
          selectEl.appendChild(opt);
        });

        const fmt = (v) => (v == null ? '—' : v.toFixed(3));

        const render = (i) => {
          const r = records[i];
          const desc = r.question ? ` — ${r.question}` : '';
          metaEl.textContent = `Type: ${r.type || 'unknown'}${desc}`;
          const vals = METHODS.map((m) => r.scores[m.key]).filter((v) => v != null);
          const max = vals.length ? Math.max(...vals, 1) : 1;
          barsEl.innerHTML = METHODS.map((m) => {
            const v = r.scores[m.key];
            const w = v == null ? 0 : Math.max(2, (v / max) * 100);
            return `<div class="bench-bar-row"><span>${m.label}</span><div class="bench-bar-track"><div class="bench-bar" style="width:${w}%"></div></div><strong>${fmt(v)}</strong></div>`;
          }).join('');
        };

        selectEl.addEventListener('change', () => render(Number(selectEl.value)));
        render(0);
      })
      .catch((e) => {
        metaEl.textContent = `Could not load comparison data: ${e.message}. Check data/results path availability for this server root.`;
      });
  };

  // Expose histogram initialization to window
  window.initHistogramPanel = function () {
    const datasetSelect = document.getElementById('histogramDataset');
    const methodSelect = document.getElementById('histogramMethod');
    const normalizeCheckbox = document.getElementById('histogramNormalize');

    if (!datasetSelect || !methodSelect) return;

    function renderHistogram() {
      const dataset = datasetSelect.value;
      const method = methodSelect.value;
      const normalize = normalizeCheckbox.checked;

      window.renderHistogram('histogramChart', dataset, method, normalize);

      // Update metric display
      fetch('data/histograms.json')
        .then((r) => r.json())
        .then((data) => {
          const d = data.find((d) => d.id === dataset);
          if (d && d.tv_distances && d.tv_distances[method]) {
            const tv = d.tv_distances[method].toFixed(4);
            const passed = d.metrics.passed[method] ? 'PASS' : 'FAIL';
            document.getElementById(
              'histogramMetric'
            ).textContent = `Total Variation: ${tv} [${passed}] | Threshold: ${d.metrics.threshold}`;
          }
        })
        .catch((e) => console.error('Error loading histogram data:', e));
    }

    datasetSelect.addEventListener('change', renderHistogram);
    methodSelect.addEventListener('change', renderHistogram);
    normalizeCheckbox.addEventListener('change', renderHistogram);

    renderHistogram();
  };
})();
