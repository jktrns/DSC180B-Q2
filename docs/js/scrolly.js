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
  let queryExplorerInitialized = false;
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
          <div class="figure-placeholder">
            <p>Project Overview Diagram</p>
            <small>Workflow comparing training-based (DP-SGD) and training-free (Private Evolution) synthesis.</small>
          </div>
        `;
        break;

      case 'step-queries':
        html = `
          <div class="query-explorer-panel">
            <div style="margin-bottom: 12px;">
              <label for="queryRange" style="display: block; margin-bottom: 8px;"><strong>Query:</strong> <span id="queryLabel">Loading...</span></label>
              <input id="queryRange" type="range" min="0" max="0" value="0" step="1" style="width: 100%;" />
            </div>
            <div class="query-card" id="queryCard">
              <div>
                <p class="meta"><span id="queryType"></span></p>
                <p id="queryDesc"></p>
                <div class="scores" id="queryScores"></div>
              </div>
              <div class="query-figure">
                <img id="queryImg" src="" alt="Query figure" style="width: 100%; border-radius: 8px; background: var(--card);" />
                <p class="caption" id="queryCaption"></p>
              </div>
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
            <h4 style="margin: 0 0 12px;">Browser Histogram: Real vs. Synthetic</h4>
            <div class="img-compare" data-left-label="Real" data-right-label="Synthetic" style="height: 350px;">
              <div style="width: 100%; height: 100%; background: var(--card); display: flex; align-items: center; justify-content: center; color: var(--muted);">
                <p>Image: q_browserhist_real.png</p>
              </div>
              <div class="img-compare__overlay" style="width: 50%;">
                <div style="width: 100%; height: 100%; background: var(--card); display: flex; align-items: center; justify-content: center; color: var(--muted);">
                  <p>Image: q_browserhist_synth.png</p>
                </div>
              </div>
              <div class="img-compare__handle" role="slider" aria-label="Image comparison handle"></div>
            </div>
            <p class="caption" style="margin-top: 8px;">Drag to compare real (left) and synthetic (right) plots.</p>
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
    if (stepId === 'step-queries' && !queryExplorerInitialized) {
      queryExplorerInitialized = true;
      setTimeout(() => {
        window.initQueryExplorer && window.initQueryExplorer();
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
        window.initImageCompare && window.initImageCompare();
      }, 100);
    }
  }

  /**
   * IntersectionObserver callback
   */
  const observerCallback = (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting && entry.intersectionRatio > 0.4) {
        const stepId = entry.target.dataset.step;
        if (stepId !== currentStep) {
          currentStep = stepId;
          const stepIndex = Array.from(steps).indexOf(entry.target);
          setPanel(stepId);
          updateProgress(stepIndex);
          entry.target.classList.add('active');
        }
      } else {
        entry.target.classList.remove('active');
      }
    });
  };

  // Set up IntersectionObserver
  const observer = new IntersectionObserver(observerCallback, {
    threshold: [0.4, 0.6],
  });

  steps.forEach((step) => {
    observer.observe(step);
  });

  // Initialize first step manually
  if (steps.length > 0) {
    const firstStepId = steps[0].dataset.step;
    setPanel(firstStepId);
    updateProgress(0);
    steps[0].classList.add('active');
  }

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
