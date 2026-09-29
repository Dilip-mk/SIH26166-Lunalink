/**
 * Lunar Image Registration — SIH 2026
 * Frontend Application Logic
 *
 * Connects to FastAPI backend at http://localhost:8000
 * POST /register  →  multipart/form-data  →  fields: "reference_file", "source_file"
 */

const API_BASE = 'https://sih26166-lunalink.onrender.com';
const API_REGISTER = `${API_BASE}/register`;

// ── DOM References ────────────────────────────────────────────────

// Reference image elements
const refInput        = document.getElementById('ref-file-input');
const refDropZone     = document.getElementById('ref-drop-zone');
const refPreviewWrap  = document.getElementById('ref-preview-wrap');
const refPreviewImg   = document.getElementById('ref-preview-img');
const refFilename     = document.getElementById('ref-filename');
const refFilesize     = document.getElementById('ref-filesize');
const btnRemoveRef    = document.getElementById('btn-remove-ref');
const refUploadTag    = document.getElementById('ref-upload-tag');

// Source image elements
const sourceInput     = document.getElementById('source-file-input');
const srcDropZone     = document.getElementById('src-drop-zone');
const srcPreviewWrap  = document.getElementById('src-preview-wrap');
const srcPreviewImg   = document.getElementById('src-preview-img');
const srcFilename     = document.getElementById('src-filename');
const srcFilesize     = document.getElementById('src-filesize');
const btnRemoveSrc    = document.getElementById('btn-remove-src');
const srcUploadTag    = document.getElementById('src-upload-tag');

// Shared UI
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
const imgReference    = document.getElementById('img-reference');
const imgOriginal     = document.getElementById('img-original');
const imgRegistered   = document.getElementById('img-registered');
const imgOverlay      = document.getElementById('img-overlay');
const imgRansac       = document.getElementById('img-ransac');
const imgCorrespondence = document.getElementById('img-correspondence');
const imgSpatialGrid  = document.getElementById('img-spatial-grid');
const imgSpatialMatches = document.getElementById('img-spatial-matches');

// Placeholders
const refImgPlaceholder = document.getElementById('ref-img-placeholder');
const origPlaceholder = document.getElementById('orig-placeholder');
const regPlaceholder  = document.getElementById('reg-placeholder');
const ovlPlaceholder  = document.getElementById('ovl-placeholder');
const rscPlaceholder  = document.getElementById('rsc-placeholder');
const corrPlaceholder = document.getElementById('corr-placeholder');
const sgPlaceholder   = document.getElementById('sg-placeholder');
const smPlaceholder   = document.getElementById('sm-placeholder');

// Spatial Grid Metric elements
const sgCoverage      = document.getElementById('sg-coverage');
const sgCells         = document.getElementById('sg-cells');
const sgStrategy      = document.getElementById('sg-strategy');

// Tabs
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');

// Methodology
const methodologyToggle  = document.getElementById('methodology-toggle');
const methodologyContent = document.getElementById('methodology-content');

// ── State ─────────────────────────────────────────────────────────
let selectedRefFile    = null;
let selectedSrcFile    = null;
let stageTimerIds      = [];
let refObjectUrl       = null;
let srcObjectUrl       = null;

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

// ── Utility ───────────────────────────────────────────────────────

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(2)} MB`;
}

const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'tif', 'tiff'];

function validateExtension(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  return ALLOWED_EXTENSIONS.includes(ext);
}

function updateRegisterButton() {
  btnRegister.disabled = !(selectedRefFile && selectedSrcFile);
}

// ── Reference File Handling ───────────────────────────────────────

function applyRefFile(file) {
  if (!file) return;
  if (!validateExtension(file)) {
    showError('Unsupported File Type', `"${file.name}" is not a supported image type. Allowed: JPG, PNG, TIF.`);
    return;
  }

  selectedRefFile = file;
  hideError();

  if (refObjectUrl) URL.revokeObjectURL(refObjectUrl);
  refObjectUrl = URL.createObjectURL(file);

  refPreviewImg.src = refObjectUrl;
  refPreviewImg.alt = `Preview of ${file.name}`;
  refFilename.textContent = file.name;
  refFilesize.textContent = formatBytes(file.size);

  refDropZone.style.display = 'none';
  refPreviewWrap.classList.add('visible');
  refUploadTag.textContent = 'Selected';
  refUploadTag.classList.remove('active');
  refUploadTag.classList.add('fixed');

  // Show reference in tab
  if (imgReference) {
    imgReference.src = refObjectUrl;
    imgReference.style.display = 'block';
  }
  if (refImgPlaceholder) refImgPlaceholder.style.display = 'none';

  updateRegisterButton();
}

function clearRefFile() {
  selectedRefFile = null;
  refInput.value = '';

  if (refObjectUrl) {
    URL.revokeObjectURL(refObjectUrl);
    refObjectUrl = null;
  }

  refPreviewImg.src = '';
  refPreviewWrap.classList.remove('visible');
  refDropZone.style.display = '';
  refUploadTag.textContent = 'Upload Required';
  refUploadTag.classList.add('active');
  refUploadTag.classList.remove('fixed');

  // Reset reference tab
  if (imgReference) {
    imgReference.src = '';
    imgReference.style.display = 'none';
  }
  if (refImgPlaceholder) refImgPlaceholder.style.display = '';

  updateRegisterButton();
}

// Reference file input change
refInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) applyRefFile(e.target.files[0]);
});

// Reference remove button
btnRemoveRef.addEventListener('click', (e) => {
  e.stopPropagation();
  clearRefFile();
  hideError();
  hideResults();
  resetProcessingUI();
});

// Reference drag and drop
refDropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  refDropZone.classList.add('dragover');
});

refDropZone.addEventListener('dragleave', (e) => {
  e.preventDefault();
  refDropZone.classList.remove('dragover');
});

refDropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  refDropZone.classList.remove('dragover');
  const file = e.dataTransfer?.files?.[0];
  if (file) applyRefFile(file);
});

// ── Source File Handling ──────────────────────────────────────────

function applySrcFile(file) {
  if (!file) return;
  if (!validateExtension(file)) {
    showError('Unsupported File Type', `"${file.name}" is not a supported image type. Allowed: JPG, PNG, TIF.`);
    return;
  }

  selectedSrcFile = file;
  hideError();

  if (srcObjectUrl) URL.revokeObjectURL(srcObjectUrl);
  srcObjectUrl = URL.createObjectURL(file);

  srcPreviewImg.src = srcObjectUrl;
  srcPreviewImg.alt = `Preview of ${file.name}`;
  srcFilename.textContent = file.name;
  srcFilesize.textContent = formatBytes(file.size);

  srcDropZone.style.display = 'none';
  srcPreviewWrap.classList.add('visible');
  srcUploadTag.textContent = 'Selected';
  srcUploadTag.classList.remove('active');
  srcUploadTag.classList.add('fixed');

  // Show original in tab
  imgOriginal.src = srcObjectUrl;
  imgOriginal.style.display = 'block';
  origPlaceholder.style.display = 'none';

  updateRegisterButton();
}

function clearSrcFile() {
  selectedSrcFile = null;
  sourceInput.value = '';

  if (srcObjectUrl) {
    URL.revokeObjectURL(srcObjectUrl);
    srcObjectUrl = null;
  }

  srcPreviewImg.src = '';
  srcPreviewWrap.classList.remove('visible');
  srcDropZone.style.display = '';
  srcUploadTag.textContent = 'Upload Required';
  srcUploadTag.classList.add('active');
  srcUploadTag.classList.remove('fixed');

  // Reset original tab
  imgOriginal.src = '';
  imgOriginal.style.display = 'none';
  origPlaceholder.style.display = '';

  updateRegisterButton();
}

// Source file input change
sourceInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) applySrcFile(e.target.files[0]);
});

// Source remove button
btnRemoveSrc.addEventListener('click', (e) => {
  e.stopPropagation();
  clearSrcFile();
  hideError();
  hideResults();
  resetProcessingUI();
});

// Source drag and drop
srcDropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  srcDropZone.classList.add('dragover');
});

srcDropZone.addEventListener('dragleave', (e) => {
  e.preventDefault();
  srcDropZone.classList.remove('dragover');
});

srcDropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  srcDropZone.classList.remove('dragover');
  const file = e.dataTransfer?.files?.[0];
  if (file) applySrcFile(file);
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
   valInlierRatio, valRmse, valCoverage].forEach(el => { if (el) el.textContent = '—'; });
  if (valCoverageCells) valCoverageCells.textContent = '— of 16 grid cells';
  if (qsInlierRatio) qsInlierRatio.textContent = '—';
  if (qsRmse) qsRmse.textContent = '—';
  if (qsCoverage) qsCoverage.textContent = '—';

  if (sgCoverage) sgCoverage.textContent = '—%';
  if (sgCells) sgCells.textContent = '— / 16';
  if (sgStrategy) sgStrategy.textContent = '—';

  // Reset images
  [imgRegistered, imgOverlay, imgRansac, imgCorrespondence, imgSpatialGrid, imgSpatialMatches].forEach(img => {
    if (img) {
      img.src = '';
      img.style.display = 'none';
    }
  });
  [regPlaceholder, ovlPlaceholder, rscPlaceholder, corrPlaceholder, sgPlaceholder, smPlaceholder].forEach(el => {
    if (el) el.style.display = '';
  });
}

// ── Number animation ──────────────────────────────────────────────

function animateNumber(el, target, decimals = 0, suffix = '') {
  if (!el) return;
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
  if (valRefKp) animateNumber(valRefKp, m.reference_keypoints, 0);
  if (valSrcKp) animateNumber(valSrcKp, m.source_keypoints, 0);
  if (valGoodMatches) animateNumber(valGoodMatches, m.good_matches, 0);
  if (valInliers) animateNumber(valInliers, m.ransac_inliers, 0);
  if (valOutliers) animateNumber(valOutliers, m.ransac_outliers, 0);
  if (valInlierRatio) animateNumber(valInlierRatio, m.inlier_ratio, 2, '%');
  if (valRmse) animateNumber(valRmse, m.rmse, 2, ' px');
  if (valCoverage) animateNumber(valCoverage, m.spatial_coverage, 1, '%');
  if (valCoverageCells) valCoverageCells.textContent = `${m.occupied_grid_cells} of ${m.total_grid_cells} grid cells`;

  // Spatial Grid bar metrics
  if (sgCoverage) animateNumber(sgCoverage, m.spatial_coverage, 1, '%');
  if (sgCells) sgCells.textContent = `${m.occupied_grid_cells} / ${m.total_grid_cells}`;
  if (sgStrategy) sgStrategy.textContent = m.spatial_strategy || 'adaptive_grid';

  // Quality summary
  if (qsInlierRatio) animateNumber(qsInlierRatio, m.inlier_ratio, 2, '%');
  if (qsRmse) animateNumber(qsRmse, m.rmse, 2, ' px');
  if (qsCoverage) animateNumber(qsCoverage, m.spatial_coverage, 1, '%');

  // Output images — URLs are /outputs/<filename> relative to API_BASE or blob/http
  function setImage(imgEl, placeholder, relOrFullUrl) {
    if (!imgEl) return;
    if (!relOrFullUrl) {
      if (placeholder) placeholder.style.display = '';
      imgEl.style.display = 'none';
      return;
    }
    let fullUrl = (relOrFullUrl.startsWith('blob:') || relOrFullUrl.startsWith('http'))
      ? relOrFullUrl
      : `${API_BASE}${relOrFullUrl}`;

    // Append timestamp cache-buster for HTTP/HTTPS outputs
    if (!fullUrl.startsWith('blob:')) {
      const sep = fullUrl.includes('?') ? '&' : '?';
      fullUrl = `${fullUrl}${sep}t=${Date.now()}`;
    }

    imgEl.src = fullUrl;
    imgEl.style.display = 'block';
    if (placeholder) placeholder.style.display = 'none';
  }

  // Reference image tab
  if (imgReference) {
    setImage(imgReference, refImgPlaceholder, o.reference_image || refObjectUrl);
  }

  setImage(imgRegistered,    regPlaceholder,  o.registered_image);
  setImage(imgOverlay,       ovlPlaceholder,  o.overlay_image);
  setImage(imgRansac,        rscPlaceholder,  o.ransac_visualization);
  setImage(imgCorrespondence, corrPlaceholder, o.ransac_visualization);
  setImage(imgSpatialGrid,   sgPlaceholder,   o.spatial_grid_image || o.spatial_visualization);
  setImage(imgSpatialMatches, smPlaceholder,  o.spatial_matches_image || o.spatial_selection_visualization);
}

// ── Register Button ───────────────────────────────────────────────

btnRegister.addEventListener('click', async () => {
  if (!selectedRefFile || !selectedSrcFile) return;

  // Reset previous state — hide error banner, hide previous results, reset processing UI
  hideError();
  hideResults();
  resetProcessingUI();

  // Disable button during processing
  btnRegister.disabled = true;
  btnRegister.innerHTML = '<span class="btn-icon" aria-hidden="true">⏳</span> Processing…';

  // Start visual stage animation
  startProcessingAnimation();

  // Prepare form data — send both reference and source files
  console.log("REGISTER DEBUG", {
    referenceFile: selectedRefFile,
    sourceFile: selectedSrcFile,
    sourceName: selectedSrcFile?.name,
    sourceSize: selectedSrcFile?.size,
    sourceType: selectedSrcFile?.type
  });

  const formData = new FormData();
  formData.append('reference_file', selectedRefFile);
  formData.append('source_file', selectedSrcFile);

  let response;
  try {
    response = await fetch(API_REGISTER, {
      method: 'POST',
      body: formData,
    });
    console.log("REGISTER RESPONSE", response.status, response.ok);
  } catch (err) {
    console.error("REGISTER FETCH ERROR", err);
    finishProcessingAnimation();
    updateRegisterButton();
    btnRegister.innerHTML = '<span class="btn-icon" aria-hidden="true">⚙️</span> Register Images';

    // Show "Cannot Connect to Backend" ONLY on genuine network failure
    showError(
      'Cannot Connect to Backend',
      `The registration service at ${API_BASE} is not responding (${err.message || err}). ` +
      `Please start the FastAPI server:\n\n  python run.py\n  OR: python -m uvicorn main:app --port 8000`
    );
    return;
  }

  // Complete animation
  finishProcessingAnimation();

  if (!response.ok) {
    let errorTitle = 'Registration Failed';
    let errorMsg = `Server returned HTTP ${response.status} ${response.statusText}`;

    if (response.status === 422) {
      errorTitle = 'Validation Error (422)';
    } else if (response.status === 400) {
      errorTitle = 'Registration Failed (400)';
    } else if (response.status === 500) {
      errorTitle = 'Registration Failed (500)';
    }

    try {
      const errData = await response.json();
      if (typeof errData.detail === 'string') {
        errorMsg = errData.detail;
      } else if (Array.isArray(errData.detail)) {
        errorMsg = errData.detail
          .map(e => (typeof e === 'string' ? e : ((e.loc ? `${e.loc.join('.')}: ` : '') + (e.msg || JSON.stringify(e)))))
          .join('; ');
      } else if (errData.detail && typeof errData.detail === 'object') {
        errorMsg = errData.detail.msg || errData.detail.detail || JSON.stringify(errData.detail);
      } else if (typeof errData.error === 'string') {
        errorMsg = errData.error;
      } else if (typeof errData.message === 'string') {
        errorMsg = errData.message;
      } else if (errData) {
        errorMsg = JSON.stringify(errData);
      }
    } catch (_) {
      errorMsg = `Server returned status ${response.status}, but response body could not be parsed.`;
    }

    if (typeof errorMsg !== 'string' || errorMsg.includes('[object Object]')) {
      errorMsg = `Server error (Status ${response.status})`;
    }

    showError(errorTitle, errorMsg);
    updateRegisterButton();
    btnRegister.innerHTML = '<span class="btn-icon" aria-hidden="true">⚙️</span> Register Images';
    return;
  }

  let data;
  try {
    data = await response.json();
  } catch (parseErr) {
    showError('Response Parsing Error', 'The server completed registration, but returned invalid JSON data.');
    updateRegisterButton();
    btnRegister.innerHTML = '<span class="btn-icon" aria-hidden="true">⚙️</span> Register Images';
    return;
  }

  if (data && data.success === false) {
    showError('Registration Failed', data.error || 'Registration service returned an error for the uploaded images.');
    updateRegisterButton();
    btnRegister.innerHTML = '<span class="btn-icon" aria-hidden="true">⚙️</span> Register Images';
    return;
  }

  populateResults(data);
  showResults();

  // Re-enable button so user can click Register again or upload another image
  updateRegisterButton();
  btnRegister.innerHTML = '<span class="btn-icon" aria-hidden="true">⚙️</span> Register Images';

  // Scroll to results
  setTimeout(() => {
    document.getElementById('section-results')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 200);
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

// ── 21st.dev Particle System ──────────────────────────────────────
//
// Ported from the 21st.dev ParticlesComponent (React TSX → vanilla JS).
// particles.js is loaded synchronously before this script via:
//   <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js">
//
// Our app is always dark (deep-space theme), so we always use dark-mode colors.
// The exact particle config, interactivity modes, opacity/size animation,
// grab + push modes, retina_detect, and all other settings are preserved
// verbatim from the original component.

(function initParticleSystem() {
  // Guard: particles.js must be loaded by now (script tag precedes app.js)
  if (typeof window.particlesJS !== 'function') {
    console.warn('[SIH26166] particles.js not yet available — will retry on load');
    window.addEventListener('load', initParticleSystem);
    return;
  }

  // Remove any existing canvas to avoid duplicates (e.g. on hot reload)
  const oldCanvas = document.querySelector('#particles-js canvas');
  if (oldCanvas) oldCanvas.remove();

  if (window.pJSDom && window.pJSDom.length > 0) {
    window.pJSDom.forEach(function (p) {
      p.pJS.fn.vendors.destroypJS();
    });
    window.pJSDom = [];
  }

  // Dark-mode colors — our app is always in dark (space) mode
  const colors = {
    particles : '#00f5ff',   // bright cyan star color
    lines     : '#00d9ff',   // slightly cooler cyan connection lines
    accent    : '#0096c7',   // deep-blue stroke accent
  };

  // ── EXACT 21st.dev structure — refined for deep-space star field ─────────
  // Visual changes from original:
  //   • Color: cyan #00f5ff → white/silver/pale-blue palette
  //   • Opacity: 0.7 → 0.35 (much more subtle)
  //   • Count: 140 → 115 (lighter density)
  //   • Line opacity: 0.4 → 0.10 (barely visible threads)
  //   • Line width: 1.2 → 0.6 (hairline)
  //   • Link distance: 160 → 120 (fewer connections)
  //   • Speed: 2 → 1.2 (slow drift)
  // Interactivity (grab/push) preserved from original component.
  window.particlesJS('particles-js', {
    particles: {
      number: {
        value: 120,
        density: { enable: true, value_area: 800 },
      },
      color: {
        // Star palette as requested: #FFFFFF, #DCEBFF, #38BDF8
        value: ['#FFFFFF', '#FFFFFF', '#DCEBFF', '#38BDF8'],
      },
      shape: {
        type: 'circle',
        stroke: { width: 0, color: 'transparent' },
      },
      opacity: {
        value: 0.65,              // 0.45 - 0.75 range
        random: true,
        anim: {
          enable: true,
          speed: 0.6,
          opacity_min: 0.30,
          sync: false,
        },
      },
      size: {
        value: 2.4,              // 1 - 3px range
        random: true,
        anim: {
          enable: true,
          speed: 1.0,
          size_min: 1.0,
          sync: false,
        },
      },
      line_linked: {
        enable   : true,
        distance : 120,
        color    : '#DCEBFF',
        opacity  : 0.14,          // 0.10 - 0.18 range
        width    : 0.6,           // 0.5 - 0.8px range
      },
      move: {
        enable    : true,
        speed     : 0.8,          // slow and subtle drift
        random    : true,
        straight  : false,
        out_mode  : 'bounce',
        attract   : { enable: false },
      },
    },
    interactivity: {
      detect_on: 'canvas',
      events: {
        onhover : { enable: true, mode: 'grab' },
        onclick  : { enable: true, mode: 'push' },
        resize   : true,
      },
      modes: {
        grab: {
          distance    : 200,
          line_linked : { opacity: 0.35 },
        },
        push: {
          particles_nb: 3,
        },
        repulse: {
          distance : 160,
          duration : 0.4,
        },
      },
    },
    retina_detect: true,
  });
})();

// ── ScrollSpy & Smooth Nav Links ─────────────────────────
(function initDynamicNavbar() {
  const nav = document.querySelector('.nav');
  const navLinks = document.querySelectorAll('.nav-links a');
  const sections = document.querySelectorAll('section[id]');

  if (!nav) return;

  // Active nav link highlighting based on viewport scroll position (ScrollSpy)
  function handleActiveNav() {
    let currentSectionId = '';
    const scrollPos = window.scrollY + 140;

    sections.forEach(section => {
      const top = section.offsetTop;
      const height = section.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        currentSectionId = section.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      const href = link.getAttribute('href');
      if (href && href.startsWith('#')) {
        const targetId = href.substring(1);
        if (targetId === currentSectionId) {
          link.classList.add('active');
        } else {
          link.classList.remove('active');
        }
      }
    });
  }

  // Bind scroll event for ScrollSpy
  window.addEventListener('scroll', handleActiveNav, { passive: true });
  handleActiveNav();

  // Smooth scrolling on nav link click
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      const href = link.getAttribute('href');
      if (href && href.startsWith('#')) {
        const targetSection = document.querySelector(href);
        if (targetSection) {
          e.preventDefault();
          targetSection.scrollIntoView({ behavior: 'smooth' });
        }
      }
    });
  });
})();

