/**
 * app.js
 * Shared application logic for all pages.
 * - Partial injection (header/footer)
 * - Active nav highlighting
 * - Page-specific initialization
 */

// ============================================================================
// PARTIAL INJECTION
// ============================================================================

async function injectPartials() {
  try {
    // Inject header
    const headerRes = await fetch('partials/header.html');
    const headerHtml = await headerRes.text();
    const headerEl = document.getElementById('siteHeader');
    if (headerEl) {
      headerEl.innerHTML = headerHtml;
    }

    // Inject footer
    const footerRes = await fetch('partials/footer.html');
    const footerHtml = await footerRes.text();
    const footerEl = document.getElementById('siteFooter');
    if (footerEl) {
      footerEl.innerHTML = footerHtml;
    }

    setActiveNav();
  } catch (e) {
    console.error('Error injecting partials:', e);
  }
}

// ============================================================================
// NAVIGATION HIGHLIGHTING
// ============================================================================

function setActiveNav() {
  const currentPage = document.body.getAttribute('data-page');
  const navLinks = document.querySelectorAll('.nav-link');

  navLinks.forEach((link) => {
    const linkPage = link.getAttribute('data-page');
    if (linkPage === currentPage) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });
}

// ============================================================================
// QUERY EXPLORER
// ============================================================================

async function initQueryExplorer() {
  const range = document.getElementById('queryRange');
  const label = document.getElementById('queryLabel');
  const typeEl = document.getElementById('queryType');
  const descEl = document.getElementById('queryDesc');
  const scoresEl = document.getElementById('queryScores');
  const imgEl = document.getElementById('queryImg');
  const capEl = document.getElementById('queryCaption');

  if (!range || !label) return; // elements don't exist on this page

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
    label.textContent = `${i + 1} of ${data.length} (${q.name})`;
    typeEl.textContent = q.type ? `Type: ${q.type}` : '';
    descEl.textContent = q.desc || '';
    if (imgEl) imgEl.src = q.image || '';
    if (capEl) capEl.textContent = q.caption || '';

    if (scoresEl) {
      scoresEl.innerHTML = '';
      (q.scores || []).forEach((s) => {
        const pill = document.createElement('div');
        pill.className = 'score-pill';
        const v = typeof s.value === 'number' ? s.value.toFixed(3) : String(s.value);
        pill.innerHTML = `<span><strong>${s.method}</strong></span><span>${v}</span><span class="small">${s.note || ''}</span>`;
        scoresEl.appendChild(pill);
      });
    }
  }

  range.addEventListener('input', () => render(Number(range.value)));
  render(0);
}

// ============================================================================
// IMAGE COMPARE SLIDER
// ============================================================================

function initImageCompare() {
  const comps = document.querySelectorAll('.img-compare');

  comps.forEach((comp) => {
    const overlay = comp.querySelector('.img-compare__overlay');
    const handle = comp.querySelector('.img-compare__handle');

    if (!overlay || !handle) return;

    let dragging = false;

    function setPos(clientX) {
      const rect = comp.getBoundingClientRect();
      const x = Math.min(Math.max(clientX - rect.left, 0), rect.width);
      const pct = (x / rect.width) * 100;
      overlay.style.width = pct + '%';
      handle.style.left = `calc(${pct}% - 10px)`;
    }

    function onDown(e) {
      dragging = true;
      if (comp.setPointerCapture) {
        comp.setPointerCapture(e.pointerId);
      }
      setPos(e.clientX);
    }

    function onMove(e) {
      if (!dragging) return;
      setPos(e.clientX);
    }

    function onUp() {
      dragging = false;
    }

    comp.addEventListener('pointerdown', onDown);
    comp.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);

    // Set default position to center
    setPos(comp.getBoundingClientRect().left + comp.getBoundingClientRect().width / 2);
  });
}

// ============================================================================
// PAGE INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
  // Always inject partials and set active nav
  await injectPartials();

  // Page-specific initialization
  const page = document.body.getAttribute('data-page');

  switch (page) {
    case 'explore':
      // Scrollytelling and histogram initializations are handled by scrolly.js
      initImageCompare();
      break;
    default:
      // Other pages may have image compares
      initImageCompare();
      break;
  }
});