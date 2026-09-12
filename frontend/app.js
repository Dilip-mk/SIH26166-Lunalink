/**
 * Lunar Image Registration — SIH 2026
 * Frontend Application Logic
 *
 * Connects to FastAPI backend at http://localhost:8000
 * POST /register  →  multipart/form-data  →  field: "file"
 */

const API_BASE = 'http://localhost:8000';
const API_REGISTER = `${API_BASE}/register`;

// ── DOM References ────────────────────────────────────────────────
const sourceInput     = document.getElementById('source-file-input');
const dropZone        = document.getElementById('drop-zone');
const srcPreviewWrap  = document.getElementById('src-preview-wrap');
const srcPreviewImg   = document.getElementById('src-preview-img');
const srcFilename     = document.getElementById('src-filename');
const srcFilesize     = document.getElementById('src-filesize');
const btnRemoveSrc    = document.getElementById('btn-remove-src');
const btnRegister     = document.getElementById('btn-register');
const errorBanner     = document.getElementById('error-banner');
const errorTitleText  = document.getElementById('error-title-text');
const errorDetail     = document.getElementById('error-detail');
const processingPanel = document.getElementById('processing-panel');
const stageItems      = document.querySelectorAll('.stage-item');

// Results
const resultsSections = document.querySelectorAll('.results-section');

// Metric value elements
const valRefKp        = document.getElementById('val-ref-kp');
const valSrcKp        = document.getElementById('val-src-kp');
const valGoodMatches  = document.getElementById('val-good-matches');
const valInliers      = document.getElementById('val-inliers');
const valOutliers     = document.getElementById('val-outliers');
const valInlierRatio  = document.getElementById('val-inlier-ratio');
const valRmse         = document.getElementById('val-rmse');
const valCoverage     = document.getElementById('val-coverage');
const valCoverageCells= document.getElementById('val-coverage-cells');

// Quality summary
const qsInlierRatio   = document.getElementById('qs-inlier-ratio');
const qsRmse          = document.getElementById('qs-rmse');
const qsCoverage      = document.getElementById('qs-coverage');

// Image elements
const imgOriginal     = document.getElementById('img-original');
const imgRegistered   = document.getElementById('img-registered');
const imgOverlay      = document.getElementById('img-overlay');
const imgRansac       = document.getElementById('img-ransac');
const imgCorrespondence = document.getElementById('img-correspondence');

// Placeholders
const origPlaceholder = document.getElementById('orig-placeholder');
const regPlaceholder  = document.getElementById('reg-placeholder');
const ovlPlaceholder  = document.getElementById('ovl-placeholder');
const rscPlaceholder  = document.getElementById('rsc-placeholder');
const corrPlaceholder = document.getElementById('corr-placeholder');

// Tabs
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');

// Methodology
const methodologyToggle  = document.getElementById('methodology-toggle');
const methodologyContent = document.getElementById('methodology-content');

// ── State ─────────────────────────────────────────────────────────
let selectedFile   = null;
let stageTimerIds  = [];
let objectUrlToRevoke = null;

// Processing stage definitions
const STAGES = [
  { id: 'stage-0', label: 'Detecting keypoints',       icon: '⬤' },
  { id: 'stage-1', label: 'Computing descriptors',      icon: '⬤' },
  { id: 'stage-2', label: 'Matching features',          icon: '⬤' },
  { id: 'stage-3', label: 'Running RANSAC',             icon: '⬤' },
  { id: 'stage-4', label: 'Estimating transformation',  icon: '⬤' },
  { id: 'stage-5', label: 'Generating registered image',icon: '⬤' },
  { id: 'stage-6', label: 'Calculating metrics',        icon: '⬤' },
];

// Approximate per-stage delay in ms (visual only — does NOT fake progress)
const STAGE_DELAY = 520;

// ── File Selection Handling ───────────────────────────────────────

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(2)} MB`;
}

function applyFile(file) {
  if (!file) return;

  // Validate extension
  const ext = file.name.split('.').pop().toLowerCase();
  const allowed = ['jpg', 'jpeg', 'png', 'tif', 'tiff'];
  if (!allowed.includes(ext)) {
    showError('Unsupported File Type', `"${file.name}" is not a supported image type. Allowed: JPG, PNG, TIF.`);
    return;
  }

  selectedFile = file;
  hideError();

  // Revoke previous object URL
  if (objectUrlToRevoke) {
    URL.revokeObjectURL(objectUrlToRevoke);
    objectUrlToRevoke = null;
  }

  const objectUrl = URL.createObjectURL(file);
  objectUrlToRevoke = objectUrl;

  // Show preview
  srcPreviewImg.src = objectUrl;
  srcPreviewImg.alt = `Preview of ${file.name}`;
  srcFilename.textContent = file.name;
  srcFilesize.textContent = formatBytes(file.size);

  dropZone.style.display = 'none';
  srcPreviewWrap.classList.add('visible');

  // Show original in tab
  imgOriginal.src = objectUrl;
  imgOriginal.style.display = 'block';
  origPlaceholder.style.display = 'none';

  // Enable register button
  btnRegister.disabled = false;
}

function clearFile() {
  selectedFile = null;
  sourceInput.value = '';

  if (objectUrlToRevoke) {
    URL.revokeObjectURL(objectUrlToRevoke);
    objectUrlToRevoke = null;
  }

  srcPreviewImg.src = '';
  srcPreviewWrap.classList.remove('visible');
  dropZone.style.display = '';

  // Reset original tab
  imgOriginal.src = '';
  imgOriginal.style.display = 'none';
  origPlaceholder.style.display = '';

  btnRegister.disabled = true;
}

// File input change
sourceInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    applyFile(e.target.files[0]);
  }
});

// Remove button
btnRemoveSrc.addEventListener('click', (e) => {
  e.stopPropagation();
  clearFile();
  hideError();
  hideResults();
  resetProcessingUI();
});

// ── Drag and Drop ─────────────────────────────────────────────────

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer?.files?.[0];
  if (file) applyFile(file);
});

// ── Tabs ──────────────────────────────────────────────────────────

tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const targetPane = btn.getAttribute('aria-controls');

    tabBtns.forEach(b => {
      b.classList.remove('active');
      b.setAttribute('aria-selected', 'false');
    });
    tabPanes.forEach(p => p.classList.remove('active'));

    btn.classList.add('active');
    btn.setAttribute('aria-selected', 'true');
    document.getElementById(targetPane)?.classList.add('active');
  });
});

// ── Methodology Toggle ────────────────────────────────────────────

methodologyToggle.addEventListener('click', () => {
  const isOpen = methodologyContent.classList.toggle('open');
  methodologyToggle.classList.toggle('open', isOpen);
  methodologyToggle.setAttribute('aria-expanded', String(isOpen));
  methodologyToggle.querySelector('span:first-child').textContent =
    isOpen ? 'Collapse Technical Details' : 'Expand Technical Details';
});

// ── Processing UI Helpers ─────────────────────────────────────────

function resetProcessingUI() {
  // Clear all timers
  stageTimerIds.forEach(id => clearTimeout(id));
  stageTimerIds = [];

  stageItems.forEach(el => {
    el.classList.remove('active', 'done');
    el.querySelector('.stage-icon').textContent = el.dataset.stage !== undefined
      ? String(Number(el.dataset.stage) + 1)
      : '•';
  });

  processingPanel.classList.remove('visible');
}

function startProcessingAnimation() {
  resetProcessingUI();
  processingPanel.classList.add('visible');

  STAGES.forEach((_, idx) => {
    // Activate stage
    const activateId = setTimeout(() => {
      stageItems[idx]?.classList.add('active');
    }, idx * STAGE_DELAY);
    stageTimerIds.push(activateId);

    // Mark previous stage done when next activates
    if (idx > 0) {
      const doneId = setTimeout(() => {
        stageItems[idx - 1]?.classList.remove('active');
        stageItems[idx - 1]?.classList.add('done');
        stageItems[idx - 1].querySelector('.stage-icon').textContent = '✓';
      }, idx * STAGE_DELAY);
      stageTimerIds.push(doneId);
    }
  });
}

function finishProcessingAnimation() {
  // Clear pending timers
  stageTimerIds.forEach(id => clearTimeout(id));
  stageTimerIds = [];

  // Mark all done
  stageItems.forEach(el => {
    el.classList.remove('active');
    el.classList.add('done');
    el.querySelector('.stage-icon').textContent = '✓';
  });

  // Hide processing panel after brief pause
  setTimeout(() => {
    processingPanel.classList.remove('visible');
  }, 800);
}

// ── Error Helpers ─────────────────────────────────────────────────

function showError(title, detail) {
  errorTitleText.textContent = title;
  errorDetail.textContent = detail;
  errorBanner.classList.add('visible');
  errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
  errorBanner.classList.remove('visible');
}

// ── Results Helpers ───────────────────────────────────────────────

function showResults() {
  resultsSections.forEach(s => {
    s.classList.add('visible');
    s.classList.add('fade-in');
  });
}

function hideResults() {
  resultsSections.forEach(s => {
    s.classList.remove('visible', 'fade-in');
  });
  // Reset metric values
  [valRefKp, valSrcKp, valGoodMatches, valInliers, valOutliers,
   valInlierRatio, valRmse, valCoverage].forEach(el => { el.textContent = '—'; });
  valCoverageCells.textContent = '— of 16 grid cells';
  qsInlierRatio.textContent = '—';
  qsRmse.textContent = '—';
  qsCoverage.textContent = '—';

  // Reset images
  [imgRegistered, imgOverlay, imgRansac, imgCorrespondence].forEach(img => {
    img.src = '';
    img.style.display = 'none';
  });
  [regPlaceholder, ovlPlaceholder, rscPlaceholder, corrPlaceholder].forEach(el => {
    el.style.display = '';
  });
}

// ── Number animation ──────────────────────────────────────────────

function animateNumber(el, target, decimals = 0, suffix = '') {
  const duration = 600;
  const start = performance.now();
  const from = 0;

  function update(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // ease out
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = from + (target - from) * eased;
    el.textContent = current.toFixed(decimals) + suffix;
    if (progress < 1) requestAnimationFrame(update);
  }

  requestAnimationFrame(update);
}

// ── Populate Results ──────────────────────────────────────────────

function populateResults(data) {
  const m = data.metrics;
  const o = data.outputs;

  // Animate numeric metrics
  animateNumber(valRefKp, m.reference_keypoints, 0);
  animateNumber(valSrcKp, m.source_keypoints, 0);
  animateNumber(valGoodMatches, m.good_matches, 0);
  animateNumber(valInliers, m.ransac_inliers, 0);
  animateNumber(valOutliers, m.ransac_outliers, 0);
  animateNumber(valInlierRatio, m.inlier_ratio, 2, '%');
  animateNumber(valRmse, m.rmse, 2, ' px');
  animateNumber(valCoverage, m.spatial_coverage, 1, '%');
  valCoverageCells.textContent = `${m.occupied_grid_cells} of ${m.total_grid_cells} grid cells`;

  // Quality summary
  animateNumber(qsInlierRatio, m.inlier_ratio, 2, '%');
  animateNumber(qsRmse, m.rmse, 2, ' px');
  animateNumber(qsCoverage, m.spatial_coverage, 1, '%');

  // Output images — URLs are /outputs/<filename> relative to API_BASE
  function setImage(imgEl, placeholder, relUrl) {
    if (!relUrl) return;
    const fullUrl = `${API_BASE}${relUrl}`;
    imgEl.src = fullUrl;
    imgEl.style.display = 'block';
    placeholder.style.display = 'none';
  }

  setImage(imgRegistered,    regPlaceholder,  o.registered_image);
  setImage(imgOverlay,       ovlPlaceholder,  o.overlay_image);
  setImage(imgRansac,        rscPlaceholder,  o.ransac_visualization);
  setImage(imgCorrespondence, corrPlaceholder, o.ransac_visualization);
}

// ── Register Button ───────────────────────────────────────────────

btnRegister.addEventListener('click', async () => {
  if (!selectedFile) return;

  // Reset previous state
  hideError();
  hideResults();
  resetProcessingUI();

  // Disable button during processing
  btnRegister.disabled = true;
  btnRegister.innerHTML = '<span class="btn-icon" aria-hidden="true">⏳</span> Processing…';

  // Start visual stage animation
  startProcessingAnimation();

  // Prepare form data
  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch(API_REGISTER, {
      method: 'POST',
      body: formData,
    });

    // Complete animation
    finishProcessingAnimation();

    if (!response.ok) {
      let errorMsg = `HTTP ${response.status} ${response.statusText}`;
      try {
        const errData = await response.json();
        if (errData.detail) errorMsg = errData.detail;
      } catch (_) {}
      showError('Registration Failed', errorMsg);
      return;
    }

    const data = await response.json();
    populateResults(data);
    showResults();

    // Scroll to results
    setTimeout(() => {
      document.getElementById('section-results')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);

  } catch (err) {
    finishProcessingAnimation();

    if (err instanceof TypeError && err.message.toLowerCase().includes('fetch')) {
      showError(
        'Cannot Connect to Backend',
        `The registration service at ${API_BASE} is not responding. ` +
        `Please start the FastAPI server:\n\n  cd backend\n  uvicorn main:app --reload`
      );
    } else {
      showError('Unexpected Error', err.message || String(err));
    }
  } finally {
    // Re-enable button
    btnRegister.disabled = false;
    btnRegister.innerHTML = '<span class="btn-icon" aria-hidden="true">⚙️</span> Register Images';
  }
});

// ── Nav link smooth scroll ────────────────────────────────────────

document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', (e) => {
    const target = document.querySelector(link.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});
