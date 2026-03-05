/**
 * app.js - Shared logic for all pages.
 * Partial injection, nav highlighting, query explorer, image compare.
 */

// === PARTIAL INJECTION ===
async function injectPartials() {
  try {
    const [headerRes, footerRes] = await Promise.all([
      fetch('partials/header.html'),
      fetch('partials/footer.html')
    ]);
    const headerHtml = await headerRes.text();
    const footerHtml = await footerRes.text();
    const headerEl = document.getElementById('siteHeader');
    const footerEl = document.getElementById('siteFooter');
    if (headerEl) headerEl.innerHTML = headerHtml;
    if (footerEl) footerEl.innerHTML = footerHtml;
    setActiveNav();
  } catch (e) {
    console.error('Error injecting partials:', e);
  }
}

// === NAV HIGHLIGHTING ===
function setActiveNav() {
  const currentPage = document.body.getAttribute('data-page');
  document.querySelectorAll('.nav-link').forEach(link => {
    const linkPage = link.getAttribute('data-page');
    link.classList.toggle('active', linkPage === currentPage);
  });
}

// === QUERY EXPLORER (used in scrollytelling) ===
async function initQueryExplorer() {
  const range = document.getElementById('queryRange');
  const label = document.getElementById('queryLabel');
  const typeEl = document.getElementById('queryType');
  const descEl = document.getElementById('queryDesc');
  const scoresEl = document.getElementById('queryScores');

  if (!range || !label) return;

  let data = [];
  try {
    const res = await fetch('data/results.json');
    data = await res.json();
  } catch (e) {
    label.textContent = 'Could not load results.json';
    return;
  }

  range.min = 0;
  range.max = Math.max(0, data.length - 1);
  range.value = 0;

  function render(i) {
    const q = data[i];
    if (!q) return;
    label.textContent = (i + 1) + ' of ' + data.length + ' \u2014 ' + q.name;
    if (typeEl) typeEl.textContent = q.type ? 'Type: ' + q.type : '';
    if (descEl) descEl.textContent = q.desc || '';

    if (scoresEl) {
      scoresEl.innerHTML = '';
      (q.scores || []).forEach(function(s) {
        var pill = document.createElement('div');
        pill.className = 'score-pill';
        if (s.value === null) {
          pill.classList.add('na');
          pill.innerHTML = '<span><strong>' + s.method + '</strong></span><span>N/A</span>';
        } else if (s.note === 'pass') {
          pill.classList.add('pass');
          pill.innerHTML = '<span><strong>' + s.method + '</strong></span><span>' + s.value.toFixed(3) + '</span><span class="small">\u2713 pass</span>';
        } else if (s.note === 'partial') {
          pill.classList.add('partial');
          pill.innerHTML = '<span><strong>' + s.method + '</strong></span><span>' + s.value.toFixed(3) + '</span><span class="small">\u25CB partial</span>';
        } else {
          pill.classList.add('fail');
          pill.innerHTML = '<span><strong>' + s.method + '</strong></span><span>' + s.value.toFixed(3) + '</span><span class="small">\u2717 fail</span>';
        }
        scoresEl.appendChild(pill);
      });
    }
  }

  range.addEventListener('input', function() { render(Number(range.value)); });
  render(0);
}
// Expose for scrolly.js
window.initQueryExplorer = initQueryExplorer;

// === IMAGE COMPARE SLIDER ===
function initImageCompare() {
  document.querySelectorAll('.img-compare').forEach(function(comp) {
    if (comp.dataset.compareInit) return;
    comp.dataset.compareInit = '1';

    var overlay = comp.querySelector('.img-compare__overlay');
    var handle = comp.querySelector('.img-compare__handle');
    if (!overlay || !handle) return;

    var dragging = false;

    function setPos(clientX) {
      var rect = comp.getBoundingClientRect();
      var x = Math.min(Math.max(clientX - rect.left, 0), rect.width);
      var pct = (x / rect.width) * 100;
      overlay.style.width = pct + '%';
      handle.style.left = 'calc(' + pct + '% - 10px)';
    }

    comp.addEventListener('pointerdown', function(e) {
      dragging = true;
      if (comp.setPointerCapture) comp.setPointerCapture(e.pointerId);
      setPos(e.clientX);
    });
    comp.addEventListener('pointermove', function(e) {
      if (dragging) setPos(e.clientX);
    });
    window.addEventListener('pointerup', function() { dragging = false; });
  });
}
window.initImageCompare = initImageCompare;

// === PAGE INIT ===
document.addEventListener('DOMContentLoaded', async function() {
  await injectPartials();
  initImageCompare();
});
