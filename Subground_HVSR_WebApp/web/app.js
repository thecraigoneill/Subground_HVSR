// app.js — HVSR Inversion Viewer

// ══════════════════════════════════════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════════════════════════════════════
let fileLoaded   = false;
let hvsrReady    = false;
let currentChan  = 'Z';

// Waveform plot mapping for click → data coordinates
let waveAxesNorm = null;
let waveDataBnds = null;

// Mask drawing on the waveform
let waveMode       = 'view';     // 'view' | 'mask'
let maskClickStart = null;       // first click time in mask mode
let activeMasks    = [];

// Inversion model (4 layers by default)
let layers = [
    { Vs: 400,  h: 10,   Ds: 0.10, Dp: 0.10 },
    { Vs: 1000, h: 15,   Ds: 0.05, Dp: 0.05 },
    { Vs: 1500, h: 30,   Ds: 0.01, Dp: 0.01 },
    { Vs: 2000, h: 1000, Ds: 0.01, Dp: 0.01 },
];

// Latest auto-axis bounds returned by Python for the velocity model
let modelAutoBounds = null;

const $ = id => document.getElementById(id);

function setStatus(msg, color = '#60a5fa') {
    $('status-bar').textContent = msg;
    $('status-bar').style.color = color;
}


// ══════════════════════════════════════════════════════════════════════════════
// FILE
// ══════════════════════════════════════════════════════════════════════════════
function openFile() {
    setStatus('Opening file dialog…');
    eel.open_tromino_file()(function(result) {
        if (!result || !result.success) {
            setStatus('⚠️  ' + (result ? result.error : 'Unknown error'), '#f87171');
            return;
        }
        fileLoaded = true;
        hvsrReady  = false;
        activeMasks = [];

        $('file-label').textContent = result.filename;
        $('compute-btn').disabled = false;
        $('save-hvsr-btn').disabled = true;
        $('save-hvsr-svg-btn').disabled = true;
        $('invert-btn').disabled = true;
        $('save-inv-btn').disabled = true;
        $('save-model-svg-btn').disabled = true;
        $('mcmc-btn').disabled     = true;
        $('save-mcmc-btn').disabled = true;
        $('mc-stats-wrap').style.display = 'none';
        $('mc-progress').textContent = '';

        // Push auto-detected values to UI
        if (typeof result.header_lines === 'number') {
            $('header-lines').value = result.header_lines;
        }
        if (typeof result.sampling === 'number') {
            $('sampling').value = result.sampling;
        }

        $('wave-info').textContent =
            result.n_samples.toLocaleString() + ' samples · ' +
            result.duration_s + ' s · fs=' + result.sampling + ' Hz';

        const det = [];
        if (result.auto_header) det.push('header=' + result.header_lines);
        if (result.auto_fs)     det.push('fs=' + result.sampling + ' Hz');
        const detMsg = det.length ? ' (auto-detected ' + det.join(', ') + ')' : '';
        setStatus('Loaded ' + result.filename + detMsg + ' — click "Compute HVSR" to process.', '#34d399');

        renderMaskChips();
        $('hvsr-img').style.display = 'none';
        $('hvsr-placeholder').style.display = 'block';
        $('model-img').style.display = 'none';
        $('model-placeholder').style.display = 'block';

        refreshWaveform();
    });
}

function applyFileParams() {
    const s   = parseInt($('sampling').value)     || 512;
    const hl  = parseInt($('header-lines').value) || 33;
    const nc  = parseInt($('n-col').value);
    const ec  = parseInt($('e-col').value);
    const zc  = parseInt($('z-col').value);
    eel.update_file_params(s, hl, nc, ec, zc)(function() {
        if (!fileLoaded) { setStatus('Parameters saved.', '#fbbf24'); return; }
        setStatus('Reloading with new parameters…');
        eel.reload_file()(function(result) {
            if (!result || !result.success) {
                setStatus('⚠️  Reload failed: ' + (result ? result.error : '?'), '#f87171');
                return;
            }
            hvsrReady = false;
            activeMasks = [];
            renderMaskChips();
            $('hvsr-img').style.display = 'none';
            $('hvsr-placeholder').style.display = 'block';
            $('invert-btn').disabled = true;
            $('save-hvsr-btn').disabled = true;
            $('save-hvsr-svg-btn').disabled = true;
            $('mcmc-btn').disabled = true;
            $('save-mcmc-btn').disabled = true;
            $('save-inv-btn').disabled = true;
            $('save-model-svg-btn').disabled = true;
            $('mc-stats-wrap').style.display = 'none';
            $('mc-progress').textContent = '';
            $('wave-info').textContent =
                result.n_samples.toLocaleString() + ' samples · ' +
                result.duration_s + ' s · fs=' + result.sampling + ' Hz';
            setStatus('Reloaded — recompute HVSR.', '#34d399');
            refreshWaveform();
        });
    });
}


// ══════════════════════════════════════════════════════════════════════════════
// WAVEFORM (with mask drawing)
// ══════════════════════════════════════════════════════════════════════════════
function setChannel(ch) {
    currentChan = ch;
    refreshWaveform();
}

function setWaveMode(mode) {
    waveMode = mode;
    maskClickStart = null;
    $('waveform-container').classList.toggle('mask-mode', mode === 'mask');
    setStatus(mode === 'mask'
        ? 'Mask mode ON — click twice on the waveform to mark a time range to exclude.'
        : 'View mode — click anywhere to read time.',
        mode === 'mask' ? '#ef4444' : '#60a5fa');
}

function refreshWaveform() {
    if (!fileLoaded) return;
    eel.plot_waveform(currentChan)(function(result) {
        if (!result || !result.success) {
            setStatus('⚠️  ' + (result ? result.error : 'Plot failed'), '#f87171');
            return;
        }
        $('wave-placeholder').style.display = 'none';
        const img = $('waveform-img');
        img.src = result.image;
        img.style.display = 'block';
        waveAxesNorm = result.axes_norm;
        waveDataBnds = result.data_bounds;
    });
}

function handleWaveClick(event) {
    const img = $('waveform-img');
    if (!img || img.style.display === 'none' || !waveAxesNorm || !waveDataBnds) return;

    const rect    = img.getBoundingClientRect();
    const figFracX = (event.clientX - rect.left) / rect.width;
    const figFracY = (event.clientY - rect.top)  / rect.height;
    const an = waveAxesNorm;
    const axFracX = (figFracX - an.x0) / (an.x1 - an.x0);
    const axFracY = (figFracY - (1 - an.y1)) / (an.y1 - an.y0);
    if (axFracX < 0 || axFracX > 1 || axFracY < 0 || axFracY > 1) return;

    const db    = waveDataBnds;
    const dataX = db.xmin + axFracX * (db.xmax - db.xmin);   // time in seconds

    $('wave-click-info').innerHTML =
        (waveMode === 'mask'
            ? '<strong style="color:#ef4444">MASK</strong> · '
            : 'Mouse · ') +
        't = ' + dataX.toFixed(3) + ' s';

    if (waveMode === 'mask') {
        if (maskClickStart === null) {
            maskClickStart = dataX;
            setStatus('First mask edge at t=' + dataX.toFixed(2) +
                      ' s — click again for the other edge.', '#fbbf24');
        } else {
            const ta = Math.min(maskClickStart, dataX);
            const tb = Math.max(maskClickStart, dataX);
            maskClickStart = null;
            setStatus('Adding mask: [' + ta.toFixed(2) + ', ' + tb.toFixed(2) +
                      '] s. Recompute HVSR to apply.', '#34d399');
            eel.add_mask(ta, tb)(function(masks) {
                activeMasks = masks;
                renderMaskChips();
                refreshWaveform();
            });
        }
    }
}

function clearMasks() {
    eel.clear_masks()(function() {
        activeMasks = [];
        renderMaskChips();
        refreshWaveform();
        setStatus('Masks cleared. Recompute HVSR to apply.', '#fbbf24');
    });
}

function removeMask(idx) {
    eel.remove_mask(idx)(function(masks) {
        activeMasks = masks;
        renderMaskChips();
        refreshWaveform();
    });
}

function renderMaskChips() {
    const wrap = $('mask-list-wrap');
    const list = $('mask-list');
    if (!activeMasks.length) {
        wrap.style.display = 'none';
        list.innerHTML = '';
        return;
    }
    wrap.style.display = 'block';
    list.innerHTML = activeMasks.map(function(m, i) {
        const a = parseFloat(m[0]).toFixed(2);
        const b = parseFloat(m[1]).toFixed(2);
        return '<div class="mask-chip">[' + a + ', ' + b + '] s' +
               '<button onclick="removeMask(' + i + ')" title="Remove">×</button></div>';
    }).join('');
}


// ══════════════════════════════════════════════════════════════════════════════
// HVSR
// ══════════════════════════════════════════════════════════════════════════════
function computeHvsr() {
    if (!fileLoaded) return;
    const win = parseFloat($('win-sec').value) || 10;
    setStatus('Computing HVSR…');
    $('compute-btn').disabled = true;
    eel.compute_hvsr(win)(function(result) {
        $('compute-btn').disabled = false;
        if (!result || !result.success) {
            setStatus('⚠️  ' + (result ? result.error : 'Failed'), '#f87171');
            return;
        }
        hvsrReady = true;
        const keptPct = (result.kept_frac * 100).toFixed(0);
        $('hvsr-info').textContent =
            result.n_points + ' pts · f=[' +
            result.f_min.toFixed(2) + ', ' + result.f_max.toFixed(1) + '] Hz · ' +
            'baseline=' + result.norm_factor.toFixed(3) + ' · ' +
            keptPct + '% data kept';
        updatePkCount(result.n_peaks || 0);
        $('invert-btn').disabled = false;
        $('save-hvsr-btn').disabled = false;
        $('save-hvsr-svg-btn').disabled = false;
        setStatus('HVSR computed (' + keptPct + '% of data after masking) · ' +
                  (result.n_peaks || 0) + ' peaks detected.', '#34d399');
        refreshHvsr();

        // Inversion overlays cleared by backend; hide model plot
        $('model-img').style.display = 'none';
        $('model-placeholder').style.display = 'block';
        $('save-inv-btn').disabled = true;
        $('save-model-svg-btn').disabled = true;
        $('mcmc-btn').disabled = true;
        $('save-mcmc-btn').disabled = true;
        $('mc-stats-wrap').style.display = 'none';
        $('mc-progress').textContent = '';
    });
}

function updatePkCount(n) {
    const el = $('pk-count');
    if (el) el.textContent = n + ' peak' + (n === 1 ? '' : 's') + ' detected';
}

function redetectAndReplot() {
    if (!hvsrReady) return;
    const promLo  = parseFloat($('pk-prom-lo').value) || 0.2;
    const promHi  = parseFloat($('pk-prom-hi').value) || 0.5;
    const distHz  = parseFloat($('pk-dist-hz').value) || 2.0;
    eel.redetect_peaks(promLo, promHi, distHz)(function(result) {
        if (!result || !result.success) {
            setStatus('⚠️  ' + (result ? result.error : 'Peak detection failed'), '#f87171');
            return;
        }
        updatePkCount(result.n_peaks);
        refreshHvsr();   // replot with updated triangles
    });
}

function refreshHvsr() {
    if (!hvsrReady) return;
    const fmin = parseFloat($('fmin').value) || 0.5;
    const fmax = parseFloat($('fmax').value) || 50;
    const showSamples = $('show-samples').checked;
    eel.plot_hvsr(fmin, fmax, showSamples)(function(result) {
        if (!result || !result.success) {
            setStatus('⚠️  ' + (result ? result.error : 'Plot failed'), '#f87171');
            return;
        }
        $('hvsr-placeholder').style.display = 'none';
        const img = $('hvsr-img');
        img.src = result.image;
        img.style.display = 'block';
    });
}


// ══════════════════════════════════════════════════════════════════════════════
// LAYER TABLE
// ══════════════════════════════════════════════════════════════════════════════
function renderLayers() {
    $('layer-tbody').innerHTML = layers.map(function(L, i) {
        const isLast = (i === layers.length - 1);
        const label  = (i + 1) + (isLast ? '*' : '');
        return '<tr>' +
            '<td style="color:#9ca3af;">' + label + '</td>' +
            '<td><input type="number" value="' + L.Vs + '" step="50"   min="100"  oninput="updLayer(' + i + ',\'Vs\',this.value)"></td>' +
            '<td><input type="number" value="' + L.h  + '" step="1"    min="0.1"  oninput="updLayer(' + i + ',\'h\', this.value)"></td>' +
            '<td><input type="number" value="' + L.Ds + '" step="0.01" min="0.001" max="0.5" oninput="updLayer(' + i + ',\'Ds\',this.value)"></td>' +
            '<td><input type="number" value="' + L.Dp + '" step="0.01" min="0.001" max="0.5" oninput="updLayer(' + i + ',\'Dp\',this.value)"></td>' +
        '</tr>';
    }).join('');
}

function updLayer(i, key, val) {
    layers[i][key] = parseFloat(val);
}

function addLayer() {
    const halfSpace = layers[layers.length - 1];
    layers.splice(layers.length - 1, 0, {
        Vs: Math.max(200, halfSpace.Vs - 200),
        h:  20,
        Ds: 0.05,
        Dp: 0.05,
    });
    renderLayers();
}

function removeLayer() {
    if (layers.length <= 2) {
        setStatus('Need at least 2 layers.', '#fbbf24');
        return;
    }
    layers.splice(layers.length - 2, 1);
    renderLayers();
}


// ══════════════════════════════════════════════════════════════════════════════
// INVERSION
// ══════════════════════════════════════════════════════════════════════════════
function runInversion() {
    if (!hvsrReady) return;

    const params = {
        Vs_init:  layers.map(function(L) { return L.Vs; }),
        h_init:   layers.map(function(L) { return L.h;  }),
        Ds:       layers.map(function(L) { return L.Ds; }),
        Dp:       layers.map(function(L) { return L.Dp; }),
        poisson:  parseFloat($('poisson').value)  || 0.4,
        fre1:     parseFloat($('fre1').value)     || 0.5,
        fre2:     parseFloat($('fre2').value)     || 30,
        Vfac:     parseFloat($('vfac').value)     || 2000,
        Hfac:     parseFloat($('hfac').value)     || 50,
        max_iter: parseInt($('max-iter').value)   || 200,
        n_freq:   parseInt($('n-freq').value)     || 300,
        // Objective weights
        alpha:    parseFloat($('alpha').value),
        beta:     parseFloat($('beta').value),
        gamma:    parseFloat($('gamma').value),
        smooth_window: parseInt($('smooth-win').value) || 0,
        step_vs:  parseFloat($('step-vs').value)  || 200,
        step_h:   parseFloat($('step-h').value)   || 5,
        tol:      parseFloat($('tol').value)       || 1e-3,
        patience: parseInt($('patience').value)    || 25,
        vs_min:   parseFloat($('vs-min').value)    || 150,
        vs_max:   parseFloat($('vs-max').value)    || 4500,
    };
    // Replace NaN (empty box) with defaults
    if (isNaN(params.alpha)) params.alpha = 0.3;
    if (isNaN(params.beta))  params.beta  = 0.3;
    if (isNaN(params.gamma)) params.gamma = 0.4;

    $('invert-btn').disabled = true;
    $('inv-progress').textContent = 'Running…';
    setStatus('Running inversion — this may take 10–60 s…', '#fbbf24');

    eel.run_inversion(params)(function(result) {
        $('invert-btn').disabled = false;
        if (!result || !result.success) {
            $('inv-progress').textContent = 'Failed';
            setStatus('⚠️  Inversion failed: ' + (result ? result.error : 'unknown'), '#f87171');
            return;
        }
        $('inv-progress').innerHTML =
            'Score: ' + result.score.toFixed(3) +
            ' · iters: ~' + result.n_iter;
        setStatus('Inversion done — score=' + result.score.toFixed(3), '#34d399');
        $('save-inv-btn').disabled = false;
        $('save-model-svg-btn').disabled = false;
        $('mcmc-btn').disabled     = false;

        // Reset MCMC display if it was showing previous results
        $('mc-stats-wrap').style.display = 'none';
        $('mc-progress').textContent = '';
        $('save-mcmc-btn').disabled  = true;

        // Refresh HVSR (now shows initial + final overlays)
        refreshHvsr();
        // Show velocity model (with current user-supplied axes if any)
        redrawModel();
    });
}


// ══════════════════════════════════════════════════════════════════════════════
// MCMC
// ══════════════════════════════════════════════════════════════════════════════
function runMcmc() {
    if (!hvsrReady) return;

    const tVal = $('mc-T').value;
    // Acceptance rule from radios
    let mcRule = 'mh';
    document.getElementsByName('mc-rule').forEach(function(r) {
        if (r.checked) mcRule = r.value;
    });
    // Read objective weights (same boxes as the NM inversion)
    let aVal = parseFloat($('alpha').value);  if (isNaN(aVal)) aVal = 0.3;
    let bVal = parseFloat($('beta').value);   if (isNaN(bVal)) bVal = 0.3;
    let gVal = parseFloat($('gamma').value);  if (isNaN(gVal)) gVal = 0.4;

    const params = {
        Ds:        layers.map(function(L) { return L.Ds; }),
        Dp:        layers.map(function(L) { return L.Dp; }),
        poisson:   parseFloat($('poisson').value)  || 0.4,
        fre1:      parseFloat($('fre1').value)     || 0.5,
        fre2:      parseFloat($('fre2').value)     || 30,
        Vfac:      parseFloat($('vfac').value)     || 2000,
        Hfac:      parseFloat($('hfac').value)     || 50,
        n_freq:    parseInt($('n-freq').value)     || 300,
        n_samp:    parseInt($('mc-n-samp').value)  || 1500,
        n_burn:    parseInt($('mc-n-burn').value)  || 500,
        step_size: parseFloat($('mc-step').value)  || 0.05,
        step_vs:   parseFloat($('mc-step-vs').value) || 1.0,
        step_h:    parseFloat($('mc-step-h').value)  || 1.0,
        T:         (tVal === '' || isNaN(parseFloat(tVal))) ? null : parseFloat(tVal),
        acceptance_rule: mcRule,
        // Objective weights (must match NM)
        alpha:     aVal,
        beta:      bVal,
        gamma:     gVal,
        smooth_window: parseInt($('smooth-win').value) || 0,
        vs_min:   parseFloat($('vs-min').value)    || 150,
        vs_max:   parseFloat($('vs-max').value)    || 4500,
    };

    $('mcmc-btn').disabled    = true;
    $('invert-btn').disabled  = true;
    $('mc-progress').textContent = 'Running MCMC…';
    setStatus('Running MCMC — this may take 30 s to several minutes…', '#fbbf24');

    eel.run_mcmc(params)(function(result) {
        $('mcmc-btn').disabled   = false;
        $('invert-btn').disabled = false;
        if (!result || !result.success) {
            $('mc-progress').textContent = 'Failed';
            setStatus('⚠️  MCMC failed: ' + (result ? result.error : 'unknown'), '#f87171');
            return;
        }
        const s = result.stats;
        $('mc-progress').innerHTML =
            'L_best=' + s.L_best.toFixed(3) +
            ' · L_mean=' + s.L_mean.toFixed(3) + '±' + s.L_std.toFixed(3) +
            '<br>' + s.n_post + ' samples · accept=' + (s.accept * 100).toFixed(0) + '% · T=' + s.T.toFixed(3);
        setStatus('MCMC done — ' + s.n_post + ' posterior samples (' +
                  (s.accept * 100).toFixed(0) + '% accept).', '#34d399');
        $('save-mcmc-btn').disabled = false;

        // Render per-layer stats table
        renderMcmcStats(s);

        // Replot HVSR (with envelope) and velocity model (with ensemble)
        refreshHvsr();
        redrawModel();
    });
}

function renderMcmcStats(s) {
    const tbody = $('mc-stats-tbody');
    const n = s.Vs.mean.length;
    let html = '';
    for (let i = 0; i < n; i++) {
        const vm = s.Vs.mean[i];
        const vs = s.Vs.std[i];
        const hm = s.h.mean[i];
        const hs = s.h.std[i];
        html += '<tr>' +
            '<td style="color:#9ca3af;">' + (i + 1) + '</td>' +
            '<td style="color:#34d399;">' + vm.toFixed(0) + ' ± ' + vs.toFixed(0) + '</td>' +
            '<td style="color:#34d399;">' + hm.toFixed(1) + ' ± ' + hs.toFixed(1) + '</td>' +
        '</tr>';
    }
    tbody.innerHTML = html;
    $('mc-stats-wrap').style.display = 'block';
}

function saveMcmc() {
    eel.save_mcmc_ensemble()(function(result) {
        setStatus(result && result.success
            ? 'MCMC ensemble saved (' + result.n_samples + ' samples) → ' + result.path
            : '⚠️  ' + (result ? result.error : 'Save failed'),
            result && result.success ? '#34d399' : '#f87171');
    });
}


// ══════════════════════════════════════════════════════════════════════════════
// VELOCITY MODEL PLOT (with manual axis controls)
// ══════════════════════════════════════════════════════════════════════════════
function _readAxisVal(id) {
    const v = $(id).value;
    if (v === '' || v === null) return null;
    const n = parseFloat(v);
    return isNaN(n) ? null : n;
}

function redrawModel() {
    eel.plot_velocity_models(
        _readAxisVal('vmin-ax'),
        _readAxisVal('vmax-ax'),
        _readAxisVal('dmin-ax'),
        _readAxisVal('dmax-ax')
    )(function(result) {
        if (!result || !result.success) {
            setStatus('⚠️  ' + (result ? result.error : 'Model plot failed'), '#f87171');
            return;
        }
        $('model-placeholder').style.display = 'none';
        const img = $('model-img');
        img.src = result.image;
        img.style.display = 'block';
        modelAutoBounds = {
            vmin: result.auto_v_min,
            vmax: result.auto_v_max,
            dmin: result.auto_d_min,
            dmax: result.auto_d_max,
        };
        // Fill placeholders so the user sees auto-bounds
        if ($('vmin-ax').value === '') $('vmin-ax').placeholder = result.auto_v_min.toFixed(0);
        if ($('vmax-ax').value === '') $('vmax-ax').placeholder = result.auto_v_max.toFixed(0);
        if ($('dmin-ax').value === '') $('dmin-ax').placeholder = result.auto_d_min.toFixed(0);
        if ($('dmax-ax').value === '') $('dmax-ax').placeholder = result.auto_d_max.toFixed(0);
    });
}

function resetModelAxes() {
    $('vmin-ax').value = '';
    $('vmax-ax').value = '';
    $('dmin-ax').value = '';
    $('dmax-ax').value = '';
    redrawModel();
}


// ══════════════════════════════════════════════════════════════════════════════
// SAVE
// ══════════════════════════════════════════════════════════════════════════════
function saveHvsr() {
    eel.save_hvsr_csv()(function(result) {
        setStatus(result && result.success
            ? 'Saved → ' + result.path
            : '⚠️  ' + (result ? result.error : 'Save failed'),
            result && result.success ? '#34d399' : '#f87171');
    });
}

function saveInversion() {
    eel.save_inversion_result()(function(result) {
        setStatus(result && result.success
            ? 'Saved → ' + result.path
            : '⚠️  ' + (result ? result.error : 'Save failed'),
            result && result.success ? '#34d399' : '#f87171');
    });
}

function saveHvsrSvg() {
    const fmin = parseFloat($('fmin').value) || 0.5;
    const fmax = parseFloat($('fmax').value) || 50;
    const showSamples = $('show-samples').checked;
    eel.save_hvsr_svg(fmin, fmax, showSamples)(function(result) {
        setStatus(result && result.success
            ? 'HVSR SVG saved → ' + result.path
            : '⚠️  ' + (result ? result.error : 'Save failed'),
            result && result.success ? '#34d399' : '#f87171');
    });
}

function saveModelSvg() {
    eel.save_model_svg(
        _readAxisVal('vmin-ax'),
        _readAxisVal('vmax-ax'),
        _readAxisVal('dmin-ax'),
        _readAxisVal('dmax-ax')
    )(function(result) {
        setStatus(result && result.success
            ? 'Model SVG saved → ' + result.path
            : '⚠️  ' + (result ? result.error : 'Save failed'),
            result && result.success ? '#34d399' : '#f87171');
    });
}


// ══════════════════════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded', function() {
    renderLayers();
});
