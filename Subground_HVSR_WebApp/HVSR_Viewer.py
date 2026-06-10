# HVSR_Viewer.py
# Web app for HVSR processing & inversion
# Algorithms ported from thecraigoneill/Subground_HVSR (MIT licence)

import eel
import io
import os
import re
import sys
import base64
import subprocess
import traceback
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks, savgol_filter
from scipy.interpolate import interp1d


# ══════════════════════════════════════════════════════════════════════════════
# FILE DIALOG SUBPROCESS HELPER
# ══════════════════════════════════════════════════════════════════════════════
HERE          = os.path.dirname(os.path.abspath(__file__))
DIALOG_SCRIPT = os.path.join(HERE, '_filedialog.py')
PYTHON_EXE    = sys.executable


def _run_dialog(args):
    try:
        proc = subprocess.run(
            [PYTHON_EXE, DIALOG_SCRIPT] + args,
            capture_output=True, text=True, timeout=300
        )
        return (proc.stdout or '').strip()
    except Exception as e:
        print(f"Dialog subprocess error: {e}")
        return ''


def _open_dialog(title, filetypes):
    return _run_dialog(['open', title, filetypes])


def _save_dialog(title, initial_name, filetypes):
    return _run_dialog(['save', title, initial_name, filetypes])


# ══════════════════════════════════════════════════════════════════════════════
# APP STATE
# ══════════════════════════════════════════════════════════════════════════════
state = {
    'file_path':    None,
    'header_lines': 33,
    'sampling':     512,
    'N_col':        0,
    'E_col':        1,
    'Z_col':        2,
    # Loaded raw data
    'time':         None,
    'N':            None,
    'E':            None,
    'Z':            None,
    # HVSR results
    'freq':         None,
    'hvsr':         None,    # normalised so baseline ≈ 1
    'hvsr_raw':     None,
    'PN':           None,
    'PE':           None,
    'PZ':           None,
    'norm_factor':  1.0,
    # Detected observed peaks (frequency and amplitude) — populated by
    # compute_hvsr / redetect_peaks, used by the objective to weight per-peak fits.
    'obs_pk_f':     None,
    'obs_pk_a':     None,
    # Peak-detection parameters (user-tunable without rerunning Welch)
    'pk_prom_lo':   0.2,     # prominence threshold below pk_split_hz
    'pk_prom_hi':   0.5,     # prominence threshold at/above pk_split_hz
    'pk_dist_hz':   2.0,     # minimum separation between peaks (Hz)
    'pk_split_hz':  20.0,    # frequency boundary between lo/hi prominence
    # Time-domain mask boxes — list of [t_min, t_max] seconds. Samples inside
    # these ranges are EXCLUDED from the HVSR Welch computation.
    'masks':        [],
    # Inversion results
    'Vs_init':      None,
    'h_init':       None,
    'Vs_final':     None,
    'h_final':      None,
    'hvsr_init':    None,
    'hvsr_final':   None,
    'inv_freq':     None,
    # MCMC ensemble (filled by run_mcmc)
    'Vs_post':      None,    # (n_post, n_layers)
    'h_post':       None,
    'L_post':       None,
    'hvsr_post':    None,    # (n_post, n_freq) — forward models for each posterior sample
    'mcmc_stats':   None,    # dict of per-layer percentiles
}


# ══════════════════════════════════════════════════════════════════════════════
# TROMINO HEADER AUTO-DETECTION
# ══════════════════════════════════════════════════════════════════════════════
_FS_REGEX = re.compile(r'Sampling\s*rate\s*:\s*\t*\s*([\d.]+)\s*Hz', re.IGNORECASE)


def _looks_like_data_line(line):
    """A data line has 3+ whitespace-separated tokens, first 3 parse as floats."""
    s = line.strip()
    if not s or s.startswith('#'):
        return False
    tokens = s.split()
    if len(tokens) < 3:
        return False
    try:
        for t in tokens[:3]:
            float(t)
        return True
    except (ValueError, TypeError):
        return False


def _detect_tromino_header(file_path, max_scan=200):
    """Scan first max_scan lines for sampling rate and data-start line index."""
    header_lines = None
    sampling     = None
    try:
        with open(file_path, 'r', encoding='unicode_escape', errors='replace') as f:
            for i, line in enumerate(f):
                if i >= max_scan and header_lines is not None:
                    break
                if sampling is None:
                    m = _FS_REGEX.search(line)
                    if m:
                        try:
                            sampling = float(m.group(1))
                        except ValueError:
                            pass
                if header_lines is None and _looks_like_data_line(line):
                    header_lines = i
                    if sampling is not None:
                        break
    except Exception as e:
        print(f"Header detection error: {e}")
    return header_lines, sampling


# ══════════════════════════════════════════════════════════════════════════════
# TROMINO FILE READER
# ══════════════════════════════════════════════════════════════════════════════
@eel.expose
def open_tromino_file():
    fp = _open_dialog("Select Tromino .dat file",
                      "DAT files=*.dat|Text files=*.txt|All files=*.*")
    if not fp:
        return {"success": False, "error": "Cancelled"}
    return _load_file(fp)


@eel.expose
def reload_file():
    if not state['file_path']:
        return {"success": False, "error": "No file loaded"}
    return _load_file(state['file_path'])


def _load_file(file_path):
    try:
        det_hdr, det_fs = _detect_tromino_header(file_path)
        if det_hdr is not None:
            state['header_lines'] = det_hdr
            print(f"  auto-detected header_lines = {det_hdr}")
        if det_fs is not None:
            state['sampling'] = det_fs
            print(f"  auto-detected sampling rate = {det_fs} Hz")

        header = int(state['header_lines'])
        n_col  = int(state['N_col'])
        e_col  = int(state['E_col'])
        z_col  = int(state['Z_col'])

        N = np.genfromtxt(file_path, skip_header=header, usecols=n_col,
                          encoding='unicode_escape', invalid_raise=False)
        E = np.genfromtxt(file_path, skip_header=header, usecols=e_col,
                          encoding='unicode_escape', invalid_raise=False)
        Z = np.genfromtxt(file_path, skip_header=header, usecols=z_col,
                          encoding='unicode_escape', invalid_raise=False)

        mask = ~(np.isnan(N) | np.isnan(E) | np.isnan(Z))
        N, E, Z = N[mask], E[mask], Z[mask]

        if len(N) == 0:
            return {"success": False,
                    "error": "No numeric data found — check header_lines and column indices"}

        fs = float(state['sampling'])
        t  = np.arange(len(N)) / fs

        # Reset HVSR + masks + inversion when a new file is loaded
        state.update({
            'file_path':    file_path,
            'N': N, 'E': E, 'Z': Z, 'time': t,
            'freq': None, 'hvsr': None, 'hvsr_raw': None,
            'PN': None,   'PE': None,   'PZ': None,
            'norm_factor': 1.0,
            'obs_pk_f': None, 'obs_pk_a': None,
            'masks':       [],
            'Vs_init': None, 'h_init': None,
            'Vs_final': None, 'h_final': None,
            'hvsr_init': None, 'hvsr_final': None,
            'inv_freq': None,
            'Vs_post': None, 'h_post': None, 'L_post': None,
            'hvsr_post': None, 'mcmc_stats': None,
        })

        fname = file_path.replace('\\', '/').split('/')[-1]
        print(f"Loaded: {fname}  |  {len(N)} samples  |  fs={fs} Hz  |  duration={len(N)/fs:.1f} s")

        return {
            "success":      True,
            "filename":     fname,
            "n_samples":    int(len(N)),
            "duration_s":   round(float(len(N) / fs), 3),
            "sampling":     fs,
            "header_lines": int(state['header_lines']),
            "auto_header":  det_hdr is not None,
            "auto_fs":      det_fs is not None,
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@eel.expose
def update_file_params(sampling, header_lines, n_col, e_col, z_col):
    state['sampling']     = int(sampling)
    state['header_lines'] = int(header_lines)
    state['N_col']        = int(n_col)
    state['E_col']        = int(e_col)
    state['Z_col']        = int(z_col)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# TIME-DOMAIN MASKS  (applied during compute_hvsr to exclude bad windows)
# ══════════════════════════════════════════════════════════════════════════════
@eel.expose
def add_mask(t_min, t_max):
    fa = float(min(t_min, t_max))
    fb = float(max(t_min, t_max))
    state['masks'].append([fa, fb])
    return state['masks']


@eel.expose
def clear_masks():
    state['masks'] = []
    return True


@eel.expose
def get_masks():
    return state['masks']


@eel.expose
def remove_mask(index):
    try:
        state['masks'].pop(int(index))
    except Exception:
        pass
    return state['masks']


def _build_keep_mask(time, masks):
    """Return bool array, True where time is OUTSIDE every mask box."""
    keep = np.ones_like(time, dtype=bool)
    for m in masks:
        try:
            a = float(m[0]); b = float(m[1])
            if b < a: a, b = b, a
            keep &= ~((time >= a) & (time <= b))
        except Exception:
            continue
    return keep


# ══════════════════════════════════════════════════════════════════════════════
# HVSR COMPUTATION (Welch periodogram, baseline-normalised so ≈ 1)
# ══════════════════════════════════════════════════════════════════════════════
@eel.expose
def compute_hvsr(window_seconds=10.0):
    if state['N'] is None:
        return {"success": False, "error": "No file loaded"}
    try:
        fs = float(state['sampling'])
        t  = state['time']
        N  = state['N']; E = state['E']; Z = state['Z']

        # Apply time-domain mask before Welch
        keep = _build_keep_mask(t, state['masks'])
        N_f, E_f, Z_f = N[keep], E[keep], Z[keep]
        if len(N_f) < int(fs):
            return {"success": False,
                    "error": "After masking, less than 1 s of data remains"}

        nps = int(float(window_seconds) * fs)
        nps = max(64, min(nps, len(N_f)))
        # Very high overlap (matches the original Subground_HVSR pipeline):
        # nps-10 means a new segment starts every 10 samples. Median averaging
        # across the many resulting overlapping segments preserves the local
        # spectral structure (sharp peaks remain sharp) that lower overlap
        # would smear. Slow (~30 s for 1440 s of data) but only runs once.
        nol = max(0, nps - 10)

        fN, PN = welch(N_f, fs=fs, nperseg=nps, noverlap=nol,
                       scaling='spectrum', average='median', window='triang')
        fE, PE = welch(E_f, fs=fs, nperseg=nps, noverlap=nol,
                       scaling='spectrum', average='median', window='triang')
        fZ, PZ = welch(Z_f, fs=fs, nperseg=nps, noverlap=nol,
                       scaling='spectrum', average='median', window='triang')

        PN, PE, PZ = np.sqrt(PN), np.sqrt(PE), np.sqrt(PZ)
        PZ_safe    = np.where(PZ > 1e-30, PZ, 1e-30)
        H          = (PN + PE) / 2.0
        hvsr_raw   = H / PZ_safe

        sel = fZ > 0
        fZ  = fZ[sel]
        PN, PE, PZ = PN[sel], PE[sel], PZ[sel]
        hvsr_raw   = hvsr_raw[sel]

        # ── Resample to log10-uniform frequency grid ─────────────────────
        # Welch output is linearly spaced (step = fs/nps Hz). On a log axis
        # this means high frequencies are massively over-represented: 1400
        # samples/decade at 100 Hz vs only 8 at 0.5 Hz. Resampling once here
        # gives uniform density across all decades for EVERY downstream use
        # — plots, peak detection, objective function, and exported CSV.
        # Use ~100 points per decade across the full usable range.
        f_lo = float(fZ[fZ > 0][0])
        f_hi = float(fZ[-1])
        n_log = max(200, int(np.log10(f_hi / f_lo) * 100))
        f_log = np.logspace(np.log10(f_lo), np.log10(f_hi), n_log)

        def _resamp(y):
            return np.interp(f_log, fZ, y)

        hvsr_raw = _resamp(hvsr_raw)
        PN       = _resamp(PN)
        PE       = _resamp(PE)
        PZ       = _resamp(PZ)
        fZ       = f_log

        # Normalise so the BASELINE (the floor of the curve away from peaks)
        # sits at 1.0. The median over-counts peak shoulders, so we use the
        # 10th percentile of the curve within an analysis band (default
        # 0.5–100 Hz, intersected with whatever the curve covers).
        norm_f_min = 0.5
        norm_f_max = 100.0
        band       = (fZ >= norm_f_min) & (fZ <= norm_f_max)
        if np.sum(band) >= 10:
            baseline = float(np.nanpercentile(hvsr_raw[band], 10))
        else:
            baseline = float(np.nanpercentile(hvsr_raw, 10))
        if not np.isfinite(baseline) or baseline <= 0:
            baseline = 1.0
        hvsr_norm = hvsr_raw / baseline

        state['freq']        = fZ
        state['hvsr']        = hvsr_norm
        state['hvsr_raw']    = hvsr_raw
        state['PN']          = PN
        state['PE']          = PE
        state['PZ']          = PZ
        state['norm_factor'] = baseline

        # Detect peaks using current parameter state (prom_lo, prom_hi, dist_hz)
        _run_peak_detection()

        # Clear stale inversion overlays — curve has changed
        for k in ('Vs_init', 'h_init', 'Vs_final', 'h_final',
                  'hvsr_init', 'hvsr_final', 'inv_freq',
                  'Vs_post', 'h_post', 'L_post', 'hvsr_post', 'mcmc_stats'):
            state[k] = None

        return {
            "success":     True,
            "n_points":    int(len(fZ)),
            "f_min":       float(fZ[0]),
            "f_max":       float(fZ[-1]),
            "hvsr_max":    float(np.nanmax(hvsr_norm)),
            "norm_factor": float(baseline),
            "kept_frac":   float(np.sum(keep)) / float(len(keep)),
            "n_peaks":     int(len(state['obs_pk_f'])),
            "peak_freqs":  state['obs_pk_f'],
            "peak_amps":   state['obs_pk_a'],
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ── Peak detection ─────────────────────────────────────────────────────────────
def _run_peak_detection():
    """Detect HVSR peaks using the current state parameters and store results.

    Uses frequency-scaled prominence: below `pk_split_hz` Hz use `pk_prom_lo`;
    at and above `pk_split_hz` Hz use `pk_prom_hi`.  Minimum peak separation is
    `pk_dist_hz` Hz converted to samples via the linear Welch frequency step.
    Runs on the normalised HVSR curve (state['hvsr']).
    """
    fZ        = state.get('freq')
    hvsr_norm = state.get('hvsr')
    if fZ is None or hvsr_norm is None:
        return

    prom_lo   = float(state.get('pk_prom_lo',  0.2))
    prom_hi   = float(state.get('pk_prom_hi',  0.5))
    dist_hz   = float(state.get('pk_dist_hz',  2.0))
    split_hz  = float(state.get('pk_split_hz', 20.0))

    try:
        # Frequency step of the Welch output (uniform linear spacing)
        df = float(fZ[1] - fZ[0]) if len(fZ) > 1 else 1.0
        # Convert Hz distance to samples (minimum 1)
        dist_samp = max(1, int(round(dist_hz / df)))

        band = (fZ >= 0.5) & (fZ <= 100.0)
        if np.sum(band) < 10:
            state['obs_pk_f'] = []
            state['obs_pk_a'] = []
            return

        f_band = fZ[band]
        h_band = hvsr_norm[band]

        # Low-frequency portion (< split_hz)
        lo_mask = f_band < split_hz
        lo_idx, _ = find_peaks(h_band, prominence=prom_lo, distance=dist_samp)
        lo_idx = lo_idx[lo_mask[lo_idx]]  # keep only those in the lo zone

        # High-frequency portion (>= split_hz)
        hi_mask = f_band >= split_hz
        hi_idx, _ = find_peaks(h_band, prominence=prom_hi, distance=dist_samp)
        hi_idx = hi_idx[hi_mask[hi_idx]]  # keep only those in the hi zone

        # Merge and sort by frequency
        all_idx = np.unique(np.concatenate([lo_idx, hi_idx]))
        all_idx = all_idx[np.argsort(f_band[all_idx])]

        state['obs_pk_f'] = f_band[all_idx].tolist()
        state['obs_pk_a'] = h_band[all_idx].tolist()

        print(f"  Peak detection: {len(all_idx)} peaks  "
              f"(prom_lo={prom_lo}, prom_hi={prom_hi}, dist={dist_hz} Hz, "
              f"split={split_hz} Hz)")
        print(f"    f={[round(x,2) for x in state['obs_pk_f']]}")
    except Exception as e:
        print(f"Peak detection failed: {e}")
        state['obs_pk_f'] = []
        state['obs_pk_a'] = []


@eel.expose
def redetect_peaks(prom_lo, prom_hi, dist_hz, split_hz=20.0):
    """Update peak-detection parameters and re-run detection on the current
    HVSR curve — without rerunning the Welch computation.
    Returns the new peak list so the UI can replot immediately.
    """
    if state.get('hvsr') is None:
        return {"success": False, "error": "Compute HVSR first"}
    state['pk_prom_lo']  = float(prom_lo)
    state['pk_prom_hi']  = float(prom_hi)
    state['pk_dist_hz']  = float(dist_hz)
    state['pk_split_hz'] = float(split_hz)
    _run_peak_detection()
    return {
        "success":   True,
        "n_peaks":   int(len(state['obs_pk_f'])),
        "peak_freqs": state['obs_pk_f'],
        "peak_amps":  state['obs_pk_a'],
    }


# ══════════════════════════════════════════════════════════════════════════════
# HVSR FORWARD MODEL — ported HV3 algorithm from Subground_HVSR
# ══════════════════════════════════════════════════════════════════════════════
def _hv3_component(c, ro, h, d, ex, fref, f):
    q  = 1.0 / (2.0 * np.asarray(d, dtype=float))
    ns = len(c); nf = len(f)

    qf = np.zeros((ns, nf), dtype=float)
    for j in range(ns):
        for i in range(nf):
            qf[j, i] = q[j] * f[i] ** ex

    idisp = 1 if fref > 0 else 0
    TR = np.zeros(ns - 1); AR = np.zeros(ns - 1)
    for I in range(ns - 1):
        TR[I] = h[I] / c[I]
        AR[I] = ro[I] * c[I] / (ro[I + 1] * c[I + 1])

    NSL = ns - 1
    FAC = np.zeros((NSL + 2, nf), dtype=np.complex128)
    for J in range(1, NSL + 1):
        for ii in range(nf):
            term = 2.0 / (1.0 + np.sqrt(1.0 + qf[J - 1, ii] ** (-2))) * (1.0 - 1j / qf[J - 1, ii])
            FAC[J - 1, ii] = np.sqrt(term)
    FAC[NSL, :] = 1.0

    AMP = np.zeros(nf)
    jpi = 1.0 / 3.14159
    Z   = np.zeros(NSL + 1, dtype=np.complex128)
    X   = np.zeros(NSL + 1, dtype=np.complex128)
    T   = np.zeros(ns, dtype=np.complex128)
    A   = np.zeros(ns, dtype=np.complex128)
    FI  = np.zeros(ns, dtype=np.complex128)

    for k in range(nf):
        X[0] = 1.0 + 0.0j; Z[0] = 1.0 + 0.0j
        ALGF = np.log(f[k] / fref) if fref > 0 else 0.0
        for J in range(2, NSL + 2):
            if idisp != 0 and qf[J - 2, k] != 0:
                FJM1 = 1 + jpi / qf[J - 2, k] * ALGF
                FJ   = 1 + jpi / qf[J - 1, k] * ALGF
            else:
                FJM1 = 1.0; FJ = 1.0
            T[J - 2]  = TR[J - 2] * FAC[J - 2, k] / FJM1
            A[J - 2]  = AR[J - 2] * FAC[J - 1, k] / FAC[J - 2, k] * FJM1 / FJ
            FI[J - 2] = 6.283186 * f[k] * T[J - 2]
            ARG       = 1j * FI[J - 2]
            CFI1      = np.exp(ARG)
            CFI2      = np.exp(-ARG)
            Z[J - 1] = ((1 + A[J - 2]) * CFI1 * Z[J - 2] + (1 - A[J - 2]) * CFI2 * X[J - 2]) * 0.5
            X[J - 1] = ((1 - A[J - 2]) * CFI1 * Z[J - 2] + (1 + A[J - 2]) * CFI2 * X[J - 2]) * 0.5
        AMP[k] = 1.0 / abs(Z[NSL]) if abs(Z[NSL]) > 1e-30 else 0.0
    return AMP


def forward_hvsr(Vs, Vp, ro, h, Ds, Dp, f, ex=0.0, fref=1.0):
    s_amp = _hv3_component(Vs, ro, h, Ds, ex, fref, f)
    p_amp = _hv3_component(Vp, ro, h, Dp, ex, fref, f)
    p_safe = np.where(np.abs(p_amp) > 1e-30, p_amp, 1e-30)
    return s_amp / p_safe


def _make_petro(Vs, poisson):
    Vs = np.asarray(Vs, dtype=float)
    Vp = Vs * np.sqrt((1.0 - poisson) / (0.5 - poisson))
    ro = 1740.0 * (Vp / 1000.0) ** 0.25
    return Vp, ro


# ══════════════════════════════════════════════════════════════════════════════
# INVERSION (Nelder-Mead)
# ══════════════════════════════════════════════════════════════════════════════
def _objective(Vs, h, hvsr_obs, freq_obs, freq_calc, Ds, Dp, poisson,
               alpha=0.3, beta=0.3, gamma=0.4, smooth_window=0,
               vs_min=150.0, vs_max=4500.0):
    """Three-term HVSR misfit, evaluated on a log-spaced frequency grid.

    Both the observed and modelled curves are interpolated onto `freq_calc`
    (which is constructed log-spaced in run_inversion / run_mcmc) so every
    sum gives equal weight per decade — no implicit linear-sampling bias.

      L_peak_amp : at each pre-detected observed peak frequency, the modelled
                   HVSR amplitude is compared to the observed amplitude in
                   log-space. Each peak counts equally regardless of its
                   absolute height, which is what lets a multi-peak curve
                   actually be fit by multiple model layers.

      L_logamp   : log-amplitude L1 over the whole band. Captures the
                   non-peak shape of the curve without taking away from the
                   peak fitting (its contribution at peak frequencies is
                   relatively small because log-amp ratios shrink there).

      L_grad     : sum-squared difference of np.gradient(hvsr) on the log-f
                   grid. This is the core peak-shape / width / slope term —
                   the gradient of a curve in log-f space directly encodes
                   peak width (sharp peak ⇒ large gradient over a few log-f
                   bins). Per Subground_HVSR paper, this resolves the
                   absolute Vs vs depth ambiguity.

    Final score: alpha*L_peak_amp + beta*L_logamp + gamma*L_grad

    Defaults alpha=0.3, beta=0.3, gamma=0.4 match the original Subground_HVSR
    paper weights (0.6*(0.5*peak + 0.5*all) + 0.4*grad).

    `smooth_window`: 0 (default) means no extra smoothing — invert the curve
    as it comes out of Welch. Set to an odd integer >= 5 to apply a
    Savitzky-Golay window to the observed curve (on the log-f grid) before
    computing L_logamp and L_grad. L_peak_amp is unaffected since the peak
    amplitudes were already extracted from the pre-smoothed curve in
    compute_hvsr. Use this when the observed curve is noisy and the
    gradient term is being dominated by noise rather than structure.
    """
    if np.any(Vs < vs_min) or np.any(Vs > vs_max) or np.any(h < 0.5):
        return 1e9
    Vp, ro = _make_petro(Vs, poisson)
    try:
        hvsr_c = forward_hvsr(Vs, Vp, ro, h, Ds, Dp, freq_calc)
    except Exception:
        return 1e9
    if np.any(~np.isfinite(hvsr_c)):
        return 1e9

    # Resample observed onto the log-spaced freq_calc grid
    fm_obs = interp1d(freq_obs, hvsr_obs, fill_value='extrapolate')
    hvsr_obs_g = fm_obs(freq_calc)
    if np.any(~np.isfinite(hvsr_obs_g)):
        return 1e9

    # Optional smoothing — only applied to the curve & gradient terms,
    # NOT to peak amplitudes (which use the already-smoothed peak values).
    if smooth_window and smooth_window >= 5:
        sw = int(smooth_window)
        if sw % 2 == 0:
            sw += 1
        if sw > len(hvsr_obs_g):
            sw = len(hvsr_obs_g) | 1
        try:
            hvsr_obs_sm = savgol_filter(hvsr_obs_g, window_length=sw, polyorder=3)
        except Exception:
            hvsr_obs_sm = hvsr_obs_g
    else:
        hvsr_obs_sm = hvsr_obs_g

    # ── 1. Per-peak amplitude (log-space) — uses raw peaks, no smoothing
    pk_f = state.get('obs_pk_f') or []
    pk_a = state.get('obs_pk_a') or []
    if len(pk_f) > 0:
        pk_freqs = np.asarray(pk_f, dtype=float)
        pk_amps  = np.asarray(pk_a, dtype=float)
        fm_c = interp1d(freq_calc, hvsr_c, fill_value='extrapolate')
        h_at_peaks = fm_c(pk_freqs)
        L_peak_amp = np.mean(
            (np.log(np.maximum(h_at_peaks, 0.1)) -
             np.log(np.maximum(pk_amps,    0.1))) ** 2
        )
    else:
        L_peak_amp = 0.0

    # ── 2. Log-amplitude L1 across whole curve (smoothed if requested)
    L_logamp = np.mean(
        np.abs(np.log(np.maximum(hvsr_c,      0.1)) -
               np.log(np.maximum(hvsr_obs_sm, 0.1)))
    )

    # ── 3. Gradient term — peak shape / width (smoothed if requested)
    L_grad = np.sum(
        (np.gradient(hvsr_c) - np.gradient(hvsr_obs_sm)) ** 2
    )

    return alpha * L_peak_amp + beta * L_logamp + gamma * L_grad


def _nelder_mead(Vs0, h0, hvsr_obs, freq_obs, freq_calc, Ds, Dp,
                 poisson, Vfac, Hfac, max_iter, status_cb=None,
                 vs_min=150.0, vs_max=4500.0, h_min=1.0,
                 alpha_w=0.3, beta_w=0.3, gamma_w=0.4, smooth_window=0,
                 step_vs=200.0, step_h=5.0, tol=1e-3, patience=25):
    """Nelder-Mead simplex optimisation for HVSR inversion.

    key parameters
    --------------
    step_vs   : initial simplex step size in m/s (physical units, independent of Vfac)
    step_h    : initial simplex step size in m   (physical units, independent of Hfac)
    tol       : relative improvement threshold — stop when the best score has
                not improved by more than `tol * best` over `patience` iterations.
                E.g. tol=1e-3 means <0.1% improvement triggers early stop.
    patience  : number of consecutive non-improving iterations before early stop.
    """
    dim_vs = len(Vs0); dim_h = len(h0); dim = dim_vs + dim_h

    def vec_to_params(x):
        Vs = np.clip(np.abs(x[:dim_vs] * Vfac), vs_min, vs_max)
        h  = np.clip(np.abs(x[dim_vs:] * Hfac), h_min, None)
        return Vs, h

    def f_score(x):
        Vs, h = vec_to_params(x)
        return _objective(Vs, h, hvsr_obs, freq_obs, freq_calc, Ds, Dp, poisson,
                          alpha=alpha_w, beta=beta_w, gamma=gamma_w,
                          smooth_window=smooth_window,
                          vs_min=vs_min, vs_max=vs_max)

    x_start = np.concatenate([Vs0 / Vfac, h0 / Hfac])

    # Initial simplex: step in PHYSICAL units converted to normalised space.
    # This makes the simplex independent of Vfac/Hfac — changing the scale
    # factors no longer shifts the initial search region.
    step_vs_n = float(step_vs) / float(Vfac)   # e.g. 200 m/s / 2000 = 0.10
    step_h_n  = float(step_h)  / float(Hfac)   # e.g.   5 m  /   50 = 0.10

    prev_best = f_score(x_start)
    res = [[x_start, prev_best]]
    for i in range(dim):
        x = np.copy(x_start)
        x[i] = x[i] + (step_vs_n if i < dim_vs else step_h_n)
        res.append([x, f_score(x)])

    alpha_nm, gamma_nm, rho_nm, sigma_nm = 1.0, 2.0, 0.5, 0.5
    no_improv = 0

    for it in range(max_iter):
        res.sort(key=lambda p: p[1])
        best = res[0][1]
        if status_cb and (it % 5 == 0):
            try: status_cb(it, max_iter, float(best))
            except Exception: pass

        # Relative improvement check
        improve_thr = tol * max(abs(prev_best), 1e-12)
        if best < prev_best - improve_thr:
            no_improv = 0; prev_best = best
        else:
            no_improv += 1
        if no_improv >= patience: break

        x0 = np.mean([p[0] for p in res[:-1]], axis=0)
        xr = x0 + alpha_nm * (x0 - res[-1][0])
        rscore = f_score(xr)
        if res[0][1] <= rscore < res[-2][1]:
            res[-1] = [xr, rscore]; continue
        if rscore < res[0][1]:
            xe = x0 + gamma_nm * (x0 - res[-1][0])
            escore = f_score(xe)
            res[-1] = [xe, escore] if escore < rscore else [xr, rscore]; continue
        xc = x0 + rho_nm * (x0 - res[-1][0])
        cscore = f_score(xc)
        if cscore < res[-1][1]:
            res[-1] = [xc, cscore]; continue
        x1 = res[0][0]; new_res = []
        for p in res:
            redx = x1 + sigma_nm * (p[0] - x1)
            new_res.append([redx, f_score(redx)])
        res = new_res

    res.sort(key=lambda p: p[1])
    Vs_b, h_b = vec_to_params(res[0][0])
    return Vs_b, h_b, res[0][1]


# ── MCMC sampler ──────────────────────────────────────────────────────────────
def _mcmc_walk(Vs0, h0, hvsr_obs, freq_obs, freq_calc, Ds, Dp,
               poisson, Vfac, Hfac, n=2000, n_burn=500,
               step_size=0.05, step_vs=1.0, step_h=1.0,
               vs_min=150.0, vs_max=4500.0, h_min=1.0, h_max=5000.0,
               status_cb=None, T=None,
               acceptance_rule='mh',
               alpha_w=0.3, beta_w=0.3, gamma_w=0.4, smooth_window=0):
    """Random-walk sampler in LOG-SPACE for Vs and h.

    Proposal: log10(Vs) and log10(h) get Gaussian perturbations with
    std = step_size * step_vs (Vs) and step_size * step_h (h). Working
    in log-space means a single fractional step produces sensible proposals
    at every scale, and the walker can never propose negative values.

    Acceptance rules:
      'mh'   — standard Metropolis-Hastings:
                 if L_prop <= L_cur          → accept
                 else accept with p = exp(-(L_prop - L_cur) / T)
      'dice' — original Subground rule:
                 if L_prop <= L_cur          → accept
                 else  d = 1 - |L_prop-L_cur| / max(L_prop, L_cur)
                       accept if d > uniform(0,1)
      'both' — alternate: dice rule on odd steps, MH on even.
               Tends to combine MH's basin-confinement with dice's exploration.

    Objective weights alpha_w/beta_w/gamma_w are forwarded to _objective.
    """
    dim_vs = len(Vs0); dim_h = len(h0); dim = dim_vs + dim_h

    def f_score_at(Vs_arr, h_arr):
        return _objective(Vs_arr, h_arr, hvsr_obs, freq_obs, freq_calc,
                          Ds, Dp, poisson,
                          alpha=alpha_w, beta=beta_w, gamma=gamma_w,
                          smooth_window=smooth_window,
                          vs_min=vs_min, vs_max=vs_max)

    Vs_cur = np.clip(np.asarray(Vs0, dtype=float).copy(), vs_min, vs_max)
    h_cur  = np.clip(np.asarray(h0,  dtype=float).copy(), h_min,  h_max)
    L_cur  = f_score_at(Vs_cur, h_cur)

    Vs_best = Vs_cur.copy(); h_best = h_cur.copy(); L_best = L_cur

    # Auto-temperature for MH if not supplied
    if T is None or (isinstance(T, float) and T <= 0):
        T = max(0.01, 0.04 * max(L_cur, 0.1))

    sd_vs = max(1e-4, step_size * step_vs)
    sd_h  = max(1e-4, step_size * step_h)

    n_total  = n_burn + n
    Vs_ens   = np.zeros((n_total, dim_vs))
    h_ens    = np.zeros((n_total, dim_h))
    L_ens    = np.zeros(n_total)
    accepted = np.zeros(n_total, dtype=bool)

    rng = np.random.default_rng()
    log_vs_min = np.log10(vs_min); log_vs_max = np.log10(vs_max)
    log_h_min  = np.log10(h_min);  log_h_max  = np.log10(h_max)

    rule = (acceptance_rule or 'mh').lower()
    if rule not in ('mh', 'dice', 'both'):
        rule = 'mh'

    for i in range(n_total):
        # Propose
        log_Vs_prop = np.log10(Vs_cur) + rng.normal(0, sd_vs, size=dim_vs)
        log_h_prop  = np.log10(h_cur)  + rng.normal(0, sd_h,  size=dim_h)
        log_Vs_prop = np.clip(log_Vs_prop, log_vs_min, log_vs_max)
        log_h_prop  = np.clip(log_h_prop,  log_h_min,  log_h_max)
        Vs_prop = 10 ** log_Vs_prop
        h_prop  = 10 ** log_h_prop

        L_prop = f_score_at(Vs_prop, h_prop)

        # Choose acceptance rule
        if rule == 'both':
            use_dice = (i % 2 == 0)   # alternate
        else:
            use_dice = (rule == 'dice')

        if L_prop <= L_cur:
            accept = True
        elif use_dice:
            denom = max(L_cur, L_prop, 1e-12)
            dice  = 1.0 - abs(L_prop - L_cur) / denom
            accept = dice > rng.uniform(0.0, 1.0)
        else:
            log_alpha_accept = -(L_prop - L_cur) / T
            accept = (log_alpha_accept > -50.0) and \
                     (rng.uniform(0.0, 1.0) < np.exp(log_alpha_accept))

        if accept:
            Vs_cur = Vs_prop
            h_cur  = h_prop
            L_cur  = L_prop
            if L_prop < L_best:
                L_best  = L_prop
                Vs_best = Vs_prop.copy()
                h_best  = h_prop.copy()

        Vs_ens[i]   = Vs_cur
        h_ens[i]    = h_cur
        L_ens[i]    = L_cur
        accepted[i] = accept

        if status_cb and (i % max(1, n_total // 40) == 0):
            try:
                status_cb(i, n_total, float(L_cur), float(L_best))
            except Exception:
                pass

    Vs_post = Vs_ens[n_burn:]
    h_post  = h_ens[n_burn:]
    L_post  = L_ens[n_burn:]

    return {
        'Vs_best': Vs_best, 'h_best': h_best, 'L_best': L_best,
        'Vs_post': Vs_post, 'h_post': h_post, 'L_post': L_post,
        'L_chain': L_ens,   'accept':   accepted,
        'T':       T,
    }


@eel.expose
def run_inversion(params):
    if state['hvsr'] is None:
        return {"success": False, "error": "Compute HVSR first"}
    try:
        Vs0      = np.asarray(params['Vs_init'], dtype=float)
        h0       = np.asarray(params['h_init'],  dtype=float)
        Ds       = np.asarray(params.get('Ds', [0.05] * len(Vs0)), dtype=float)
        Dp       = np.asarray(params.get('Dp', [0.05] * len(Vs0)), dtype=float)
        poisson  = float(params.get('poisson',  0.4))
        f1       = float(params.get('fre1',     0.5))
        f2       = float(params.get('fre2',     50.0))
        Vfac     = float(params.get('Vfac',     2000.0))
        Hfac     = float(params.get('Hfac',     50.0))
        max_iter = int(params.get('max_iter',   200))
        n_freq   = int(params.get('n_freq',     300))
        # Objective weights (user-tunable)
        alpha_w  = float(params.get('alpha',    0.3))
        beta_w   = float(params.get('beta',     0.3))
        gamma_w  = float(params.get('gamma',    0.4))
        sm_win   = int(params.get('smooth_window', 0))
        # Simplex step sizes in physical units (decoupled from scale factors)
        step_vs  = float(params.get('step_vs',  200.0))   # m/s
        step_h   = float(params.get('step_h',   5.0))     # m
        # Convergence tolerance (relative) and patience
        tol      = float(params.get('tol',      1e-3))
        patience = int(params.get('patience',   25))
        # Vs physical bounds (user-set)
        vs_min_u = float(params.get('vs_min',   150.0))
        vs_max_u = float(params.get('vs_max',   4500.0))

        if len(Vs0) != len(h0):
            return {"success": False, "error": "Vs_init and h_init must have same length"}
        if len(Vs0) < 2:
            return {"success": False, "error": "Need at least 2 layers"}

        f_calc = np.logspace(np.log10(f1), np.log10(f2), n_freq)

        freq_full = state['freq']
        hvsr_full = state['hvsr']
        keep = (freq_full >= f1) & (freq_full <= f2)
        if np.sum(keep) < 10:
            return {"success": False,
                    "error": "Too few observed HVSR points in band — widen fre1..fre2"}
        freq_obs = freq_full[keep]
        hvsr_obs = hvsr_full[keep]

        Vp_init, ro_init = _make_petro(Vs0, poisson)
        hvsr_init = forward_hvsr(Vs0, Vp_init, ro_init, h0, Ds, Dp, f_calc)

        progress_msgs = []
        def status_cb(it, total, score):
            progress_msgs.append(f"iter {it}/{total}  L1={score:.3f}")

        Vs_b, h_b, score = _nelder_mead(
            Vs0, h0, hvsr_obs, freq_obs, f_calc, Ds, Dp,
            poisson, Vfac, Hfac, max_iter, status_cb=status_cb,
            alpha_w=alpha_w, beta_w=beta_w, gamma_w=gamma_w,
            smooth_window=sm_win,
            step_vs=step_vs, step_h=step_h, tol=tol, patience=patience,
            vs_min=vs_min_u, vs_max=vs_max_u,
        )

        Vp_b, ro_b = _make_petro(Vs_b, poisson)
        hvsr_final = forward_hvsr(Vs_b, Vp_b, ro_b, h_b, Ds, Dp, f_calc)

        state['Vs_init']    = Vs0
        state['h_init']     = h0
        state['Vs_final']   = Vs_b
        state['h_final']    = h_b
        state['hvsr_init']  = hvsr_init
        state['hvsr_final'] = hvsr_final
        state['inv_freq']   = f_calc

        return {
            "success":  True,
            "Vs_final": Vs_b.tolist(),
            "h_final":  h_b.tolist(),
            "score":    float(score),
            "n_iter":   len(progress_msgs) * 5,
            "log":      progress_msgs[-10:],
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@eel.expose
def run_mcmc(params):
    """Run MCMC starting from the current Nelder-Mead best model.
    Returns posterior statistics per layer plus the chain log-misfit trace.
    """
    if state['hvsr'] is None:
        return {"success": False, "error": "Compute HVSR first"}
    if state['Vs_final'] is None:
        return {"success": False, "error": "Run inversion first — MCMC needs a starting model"}
    try:
        # Use the Nelder-Mead best as the seed; the table layers are already
        # frozen there.
        Vs0 = np.asarray(state['Vs_final'], dtype=float).copy()
        h0  = np.asarray(state['h_final'],  dtype=float).copy()
        Ds  = np.asarray(params.get('Ds',  [0.05] * len(Vs0)), dtype=float)
        Dp  = np.asarray(params.get('Dp',  [0.05] * len(Vs0)), dtype=float)
        poisson  = float(params.get('poisson',  0.4))
        f1       = float(params.get('fre1',     0.5))
        f2       = float(params.get('fre2',     50.0))
        Vfac     = float(params.get('Vfac',     2000.0))
        Hfac     = float(params.get('Hfac',     50.0))
        n_freq   = int(params.get('n_freq',     300))
        # Objective weights (shared with NM)
        alpha_w  = float(params.get('alpha',    0.3))
        beta_w   = float(params.get('beta',     0.3))
        gamma_w  = float(params.get('gamma',    0.4))
        sm_win   = int(params.get('smooth_window', 0))
        # Vs physical bounds
        vs_min_u = float(params.get('vs_min',   150.0))
        vs_max_u = float(params.get('vs_max',   4500.0))

        # MCMC-specific
        n_samp   = int(params.get('n_samp',     1500))
        n_burn   = int(params.get('n_burn',     500))
        step_sz  = float(params.get('step_size', 0.05))   # log-space fractional step
        step_vs  = float(params.get('step_vs',   1.0))    # Vs relative step
        step_h   = float(params.get('step_h',    1.0))    # h relative step
        T_in     = params.get('T', None)
        T        = float(T_in) if T_in not in (None, '', 'None') else None
        accept_rule = (params.get('acceptance_rule', 'mh') or 'mh').lower()
        if accept_rule not in ('mh', 'dice', 'both'):
            accept_rule = 'mh'

        f_calc    = np.logspace(np.log10(f1), np.log10(f2), n_freq)
        freq_full = state['freq']; hvsr_full = state['hvsr']
        keep      = (freq_full >= f1) & (freq_full <= f2)
        if np.sum(keep) < 10:
            return {"success": False, "error": "Too few HVSR points in band — widen fre1..fre2"}
        freq_obs  = freq_full[keep]
        hvsr_obs  = hvsr_full[keep]

        progress_msgs = []
        def status_cb(i, total, L_cur, L_best):
            progress_msgs.append(f"step {i}/{total}  L_cur={L_cur:.3f}  L_best={L_best:.3f}")

        result = _mcmc_walk(
            Vs0, h0, hvsr_obs, freq_obs, f_calc, Ds, Dp,
            poisson, Vfac, Hfac,
            n=n_samp, n_burn=n_burn,
            step_size=step_sz, step_vs=step_vs, step_h=step_h,
            T=T,
            acceptance_rule=accept_rule,
            alpha_w=alpha_w, beta_w=beta_w, gamma_w=gamma_w,
            smooth_window=sm_win,
            vs_min=vs_min_u, vs_max=vs_max_u,
            status_cb=status_cb,
        )

        Vs_best = result['Vs_best']
        h_best  = result['h_best']
        Vs_post = result['Vs_post']
        h_post  = result['h_post']
        L_post  = result['L_post']

        # Compute forward HVSR for every posterior sample (subsample if huge)
        # to keep plotting/storage manageable
        max_keep = 400
        if len(Vs_post) > max_keep:
            idx = np.linspace(0, len(Vs_post) - 1, max_keep).astype(int)
            Vs_post_sub = Vs_post[idx]
            h_post_sub  = h_post[idx]
            L_post_sub  = L_post[idx]
        else:
            Vs_post_sub = Vs_post
            h_post_sub  = h_post
            L_post_sub  = L_post

        Vp_best, ro_best = _make_petro(Vs_best, poisson)
        hvsr_best_curve  = forward_hvsr(Vs_best, Vp_best, ro_best, h_best, Ds, Dp, f_calc)

        hvsr_post = np.zeros((len(Vs_post_sub), len(f_calc)))
        for i, (Vs_i, h_i) in enumerate(zip(Vs_post_sub, h_post_sub)):
            Vp_i, ro_i  = _make_petro(Vs_i, poisson)
            hvsr_post[i] = forward_hvsr(Vs_i, Vp_i, ro_i, h_i, Ds, Dp, f_calc)

        # Per-layer percentiles for Vs and h
        def _pct(arr):
            return {
                "p05":  np.nanpercentile(arr, 5,  axis=0).tolist(),
                "p25":  np.nanpercentile(arr, 25, axis=0).tolist(),
                "p50":  np.nanpercentile(arr, 50, axis=0).tolist(),
                "p75":  np.nanpercentile(arr, 75, axis=0).tolist(),
                "p95":  np.nanpercentile(arr, 95, axis=0).tolist(),
                "mean": np.nanmean(arr, axis=0).tolist(),
                "std":  np.nanstd(arr,  axis=0).tolist(),
            }

        stats = {
            "Vs": _pct(Vs_post),
            "h":  _pct(h_post),
            "L_best":    float(result['L_best']),
            "L_mean":    float(np.nanmean(L_post)),
            "L_std":     float(np.nanstd(L_post)),
            "accept":    float(np.mean(result['accept'])),
            "n_post":    int(len(Vs_post)),
            "n_burn":    int(n_burn),
            "T":         float(result.get('T', 0.0)),
        }

        # Update the "best" model with the MCMC-improved best (might be better
        # than Nelder-Mead alone)
        state['Vs_final']   = Vs_best
        state['h_final']    = h_best
        state['hvsr_final'] = hvsr_best_curve
        state['inv_freq']   = f_calc
        # Store ensemble
        state['Vs_post']    = Vs_post_sub
        state['h_post']     = h_post_sub
        state['L_post']     = L_post_sub
        state['hvsr_post']  = hvsr_post
        state['mcmc_stats'] = stats

        return {
            "success":  True,
            "Vs_best":  Vs_best.tolist(),
            "h_best":   h_best.tolist(),
            "stats":    stats,
            "log":      progress_msgs[-12:],
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# PLOT RENDERING
# ══════════════════════════════════════════════════════════════════════════════
def _render_to_b64(fig):
    # tight_layout can warn for gridspec figures but still produces good output;
    # silence the warning while keeping the layout call.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')


def _axes_bbox(ax):
    bb = ax.get_position()
    return {"x0": bb.x0, "y0": bb.y0, "x1": bb.x1, "y1": bb.y1}


@eel.expose
def plot_waveform(channel='Z'):
    """Plot one channel with time-domain mask bands as red shaded regions."""
    if state['time'] is None:
        return {"success": False, "error": "No data"}
    try:
        t = state['time']
        if   channel == 'N': y = state['N']; col = '#60a5fa'; lbl = 'N–S'
        elif channel == 'E': y = state['E']; col = '#6bcb77'; lbl = 'E–W'
        else:                y = state['Z']; col = '#ffd93d'; lbl = 'Z (vertical)'

        t_full = t
        if len(t) > 8000:
            stride = max(1, len(t) // 8000)
            t_plot = t[::stride]; y_plot = y[::stride]
        else:
            t_plot = t; y_plot = y

        fig, ax = plt.subplots(figsize=(11, 2.6))
        fig.patch.set_facecolor('#1a1d22')
        ax.set_facecolor('#1a1d22')

        for m in state['masks']:
            try:
                a = float(m[0]); b = float(m[1])
                if b < a: a, b = b, a
                ax.axvspan(a, b, color='#ef4444', alpha=0.30, zorder=1)
            except Exception:
                continue

        ax.plot(t_plot, y_plot, color=col, linewidth=0.5, zorder=2)
        ax.set_xlabel("Time (s)", color='#a0a0b0', fontsize=9)
        ax.set_ylabel(lbl + " (mm/s)", color='#a0a0b0', fontsize=9)
        ax.tick_params(colors='#a0a0b0', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')
        ax.grid(True, alpha=0.2, color='#888')
        ax.set_xlim(t_full[0], t_full[-1])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fig.tight_layout()
        fig.canvas.draw()
        bbox = _axes_bbox(ax)
        ymin, ymax = ax.get_ylim()
        return {
            "success":   True,
            "image":     _render_to_b64(fig),
            "axes_norm": bbox,
            "data_bounds": {
                "xmin": float(t_full[0]),  "xmax": float(t_full[-1]),
                "ymin": float(ymin),       "ymax": float(ymax),
            },
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def _build_hvsr_figure(f_min=0.5, f_max=50.0, show_samples=False):
    """Build (but do not render) the HVSR figure. Returns a matplotlib Figure
    that the caller can either b64-encode (for inline display) or save to SVG.
    """
    f  = state['freq']
    h  = state['hvsr']
    PN = state['PN']; PE = state['PE']; PZ = state['PZ']

    fig = plt.figure(figsize=(11, 7.6))
    fig.patch.set_facecolor('#1a1d22')
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.10)

    # Top: spectra
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1a1d22')
    ax1.loglog(f, PN, color='#60a5fa', linewidth=1.2, alpha=0.9, label='N–S')
    ax1.loglog(f, PE, color='#6bcb77', linewidth=1.2, alpha=0.9, label='E–W')
    ax1.loglog(f, PZ, color='#ffd93d', linewidth=1.2, alpha=0.9, label='Z')
    ax1.set_xlim(f_min, f_max)
    ax1.set_ylabel("Spectral amp", color='#a0a0b0', fontsize=10)
    ax1.tick_params(colors='#a0a0b0', labelsize=8, labelbottom=False)
    for sp in ax1.spines.values():
        sp.set_edgecolor('#444')
    ax1.grid(True, which='both', alpha=0.2, color='#888')
    ax1.legend(loc='upper right', facecolor='#1e2128', edgecolor='#444',
               labelcolor='white', fontsize=8, ncol=3)

    # Bottom: HVSR
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#1a1d22')

    # MCMC envelope (drawn first so other lines sit on top)
    if state['hvsr_post'] is not None and state['inv_freq'] is not None:
        hp   = state['hvsr_post']
        invf = state['inv_freq']
        try:
            p05 = np.nanpercentile(hp, 5,  axis=0)
            p95 = np.nanpercentile(hp, 95, axis=0)
            p25 = np.nanpercentile(hp, 25, axis=0)
            p75 = np.nanpercentile(hp, 75, axis=0)
            ax2.fill_between(invf, p05, p95, color='#34d399',
                              alpha=0.12, zorder=2,
                              label='MCMC 90% band')
            ax2.fill_between(invf, p25, p75, color='#34d399',
                              alpha=0.22, zorder=2)
        except Exception:
            pass

    ax2.semilogx(f, h, color='#60a5fa', linewidth=2.0,
                 label='Observed (normalised)', zorder=4)

    # Frequency-sample dots (toggle from UI)
    if show_samples:
        in_band = (f >= f_min) & (f <= f_max)
        ax2.scatter(f[in_band], h[in_band], s=7, color='#93c5fd',
                    edgecolor='none', zorder=5, alpha=0.85,
                    label=f'Samples ({int(np.sum(in_band))})')

    if state['hvsr_init'] is not None:
        ax2.semilogx(state['inv_freq'], state['hvsr_init'],
                     color='#fbbf24', linewidth=1.2, linestyle='--',
                     label='Initial model', zorder=5)
    if state['hvsr_final'] is not None:
        ax2.semilogx(state['inv_freq'], state['hvsr_final'],
                     color='#34d399', linewidth=1.8, linestyle='-',
                     label='Inverted (best)', zorder=6)

    ax2.axhline(1.0, color='#6b7280', linewidth=0.6, linestyle=':', zorder=1)

    # Draw detected observed peaks (used by the objective for per-peak fits)
    pk_f = state.get('obs_pk_f') or []
    pk_a = state.get('obs_pk_a') or []
    if len(pk_f) > 0:
        ax2.scatter(pk_f, pk_a, marker='v', s=70, color='#f87171',
                    edgecolor='#fff', linewidth=0.7, zorder=8,
                    label=f'Detected peaks ({len(pk_f)})')

    in_band = (f >= f_min) & (f <= f_max)
    ymax = float(np.nanmax(h[in_band]) * 1.15) if np.any(in_band) else 5.0
    ymax = max(ymax, 2.5)
    ax2.set_xlim(f_min, f_max)
    ax2.set_ylim(0, ymax)
    ax2.set_xlabel("Frequency (Hz)", color='#a0a0b0')
    ax2.set_ylabel("HVSR amplitude (baseline = 1)", color='#a0a0b0')
    ax2.tick_params(colors='#a0a0b0')
    for sp in ax2.spines.values():
        sp.set_edgecolor('#444')
    ax2.grid(True, which='both', alpha=0.25, color='#888')
    ax2.legend(loc='upper right', facecolor='#1e2128', edgecolor='#444',
               labelcolor='white', fontsize=9)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig.tight_layout()
    fig.canvas.draw()
    return fig


@eel.expose
def plot_hvsr(f_min=0.5, f_max=50.0, show_samples=False):
    """Two stacked subplots: spectra (top) + HVSR (bottom, normalised baseline=1).
    If show_samples=True, draws small dots on each observed HVSR sample so the
    user can see the underlying frequency-sampling density.
    """
    if state['hvsr'] is None:
        return {"success": False, "error": "Compute HVSR first"}
    try:
        fig = _build_hvsr_figure(f_min, f_max, show_samples)
        return {"success": True, "image": _render_to_b64(fig)}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def _stair(Vs, h):
    """Build step-profile arrays for plotting Vs vs depth."""
    depth = []; vs = []
    d = 0.0
    for i in range(len(Vs)):
        depth.append(d); vs.append(float(Vs[i]))
        d += float(h[i])
        depth.append(d); vs.append(float(Vs[i]))
    return np.array(vs), np.array(depth)


def _build_model_figure(v_min=None, v_max=None, d_min=None, d_max=None):
    """Build (but do not render) the velocity-model figure.
    Returns (Figure, auto_bounds_dict).
    """
    VsI, DI = _stair(state['Vs_init'],  state['h_init'])
    VsF, DF = _stair(state['Vs_final'], state['h_final'])

    fig, ax = plt.subplots(figsize=(5.5, 7))
    fig.patch.set_facecolor('#1a1d22')
    ax.set_facecolor('#1a1d22')

    # MCMC ensemble: thin traces drawn FIRST (so best/initial sit on top)
    if state['Vs_post'] is not None and state['h_post'] is not None:
        Vs_post = state['Vs_post']
        h_post  = state['h_post']
        n_show  = min(80, len(Vs_post))
        idx     = np.linspace(0, len(Vs_post) - 1, n_show).astype(int)
        for k in idx:
            Vs_k, D_k = _stair(Vs_post[k], h_post[k])
            ax.plot(Vs_k, D_k, color='#60a5fa', linewidth=0.5,
                    alpha=0.15, zorder=1)
        try:
            Vs_med = np.nanmedian(Vs_post, axis=0)
            h_med  = np.nanmedian(h_post,  axis=0)
            Vsm, Dm = _stair(Vs_med, h_med)
            ax.plot(Vsm, Dm, color='#60a5fa', linewidth=1.4,
                    linestyle='-', alpha=0.8,
                    label='MCMC median', zorder=2)
        except Exception:
            pass

    ax.plot(VsI, DI, color='#fbbf24', linewidth=1.5, linestyle='--',
            label='Initial', zorder=3)
    ax.plot(VsF, DF, color='#34d399', linewidth=2.2,
            label='Inverted (best)', zorder=4)

    all_vs = np.concatenate([VsI, VsF])
    if state['Vs_post'] is not None:
        all_vs = np.concatenate([all_vs, state['Vs_post'].ravel()])
    h_init  = np.asarray(state['h_init'], dtype=float)
    h_final = np.asarray(state['h_final'], dtype=float)
    d_solid_i = float(np.sum(h_init[:-1]))  if len(h_init)  > 1 else float(h_init[0])
    d_solid_f = float(np.sum(h_final[:-1])) if len(h_final) > 1 else float(h_final[0])
    d_solid   = max(d_solid_i, d_solid_f) * 1.4

    def _pick(val, default):
        try:
            if val in (None, '', 'None'):
                return default
            return float(val)
        except Exception:
            return default

    auto_v_min = float(np.min(all_vs)) * 0.9
    auto_v_max = float(np.max(all_vs)) * 1.1
    auto_d_min = 0.0
    auto_d_max = float(max(d_solid, 5.0))

    v_min_v = _pick(v_min, auto_v_min)
    v_max_v = _pick(v_max, auto_v_max)
    d_min_v = _pick(d_min, auto_d_min)
    d_max_v = _pick(d_max, auto_d_max)

    ax.set_xlim(v_min_v, v_max_v)
    ax.set_ylim(d_max_v, d_min_v)
    ax.set_xlabel("Vs (m/s)", color='#a0a0b0')
    ax.set_ylabel("Depth (m)", color='#a0a0b0')
    ax.tick_params(colors='#a0a0b0')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')
    ax.grid(True, alpha=0.25, color='#888')
    ax.legend(loc='lower right', facecolor='#1e2128', edgecolor='#444',
              labelcolor='white', fontsize=9)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig.tight_layout()
    fig.canvas.draw()
    bounds = {
        "auto_v_min": auto_v_min, "auto_v_max": auto_v_max,
        "auto_d_min": auto_d_min, "auto_d_max": auto_d_max,
    }
    return fig, bounds


@eel.expose
def plot_velocity_models(v_min=None, v_max=None, d_min=None, d_max=None):
    """Plot initial + final Vs profiles with user-controllable axes."""
    if state['Vs_init'] is None:
        return {"success": False, "error": "Run inversion first"}
    try:
        fig, bounds = _build_model_figure(v_min, v_max, d_min, d_max)
        return {
            "success": True,
            "image":   _render_to_b64(fig),
            **bounds,
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════
@eel.expose
def save_hvsr_csv():
    if state['hvsr'] is None:
        return {"success": False, "error": "No HVSR data"}
    fp = _save_dialog("Save HVSR curve as CSV", "hvsr.csv", "CSV files=*.csv")
    if not fp:
        return {"success": False, "error": "Cancelled"}
    try:
        import csv
        with open(fp, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["freq_Hz", "hvsr_norm", "hvsr_raw", "PN", "PE", "PZ"])
            for i in range(len(state['freq'])):
                w.writerow([state['freq'][i], state['hvsr'][i],
                            state['hvsr_raw'][i],
                            state['PN'][i], state['PE'][i], state['PZ'][i]])
        return {"success": True, "path": fp}
    except Exception as e:
        return {"success": False, "error": str(e)}


@eel.expose
def save_inversion_result():
    if state['Vs_final'] is None:
        return {"success": False, "error": "Run inversion first"}
    fp = _save_dialog("Save inversion result", "inversion_result.csv", "CSV files=*.csv")
    if not fp:
        return {"success": False, "error": "Cancelled"}
    try:
        import csv
        with open(fp, 'w', newline='') as f:
            w = csv.writer(f)
            # If MCMC has run, include percentiles
            if state['mcmc_stats'] is not None:
                s = state['mcmc_stats']
                w.writerow(["layer",
                            "Vs_init", "h_init",
                            "Vs_best", "h_best",
                            "Vs_p05", "Vs_p25", "Vs_p50", "Vs_p75", "Vs_p95",
                            "Vs_mean", "Vs_std",
                            "h_p05",  "h_p25",  "h_p50",  "h_p75",  "h_p95",
                            "h_mean", "h_std"])
                for i in range(len(state['Vs_final'])):
                    w.writerow([
                        i + 1,
                        state['Vs_init'][i],  state['h_init'][i],
                        state['Vs_final'][i], state['h_final'][i],
                        s['Vs']['p05'][i], s['Vs']['p25'][i], s['Vs']['p50'][i],
                        s['Vs']['p75'][i], s['Vs']['p95'][i],
                        s['Vs']['mean'][i], s['Vs']['std'][i],
                        s['h']['p05'][i],  s['h']['p25'][i],  s['h']['p50'][i],
                        s['h']['p75'][i],  s['h']['p95'][i],
                        s['h']['mean'][i], s['h']['std'][i],
                    ])
            else:
                w.writerow(["layer", "Vs_init", "h_init", "Vs_final", "h_final"])
                for i in range(len(state['Vs_final'])):
                    w.writerow([i + 1,
                                state['Vs_init'][i],  state['h_init'][i],
                                state['Vs_final'][i], state['h_final'][i]])
        return {"success": True, "path": fp}
    except Exception as e:
        return {"success": False, "error": str(e)}


@eel.expose
def save_mcmc_ensemble():
    """Save the full MCMC posterior ensemble (every kept sample, per layer)."""
    if state['Vs_post'] is None:
        return {"success": False, "error": "Run MCMC first"}
    fp = _save_dialog("Save MCMC ensemble", "mcmc_ensemble.csv", "CSV files=*.csv")
    if not fp:
        return {"success": False, "error": "Cancelled"}
    try:
        import csv
        Vs_post = state['Vs_post']
        h_post  = state['h_post']
        L_post  = state['L_post']
        n_layers = Vs_post.shape[1]
        header = ['sample', 'L'] + \
                 [f'Vs_L{i+1}' for i in range(n_layers)] + \
                 [f'h_L{i+1}'  for i in range(n_layers)]
        with open(fp, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for k in range(len(Vs_post)):
                row = [k, float(L_post[k])] + \
                      [float(Vs_post[k, i]) for i in range(n_layers)] + \
                      [float(h_post[k, i])  for i in range(n_layers)]
                w.writerow(row)
        return {"success": True, "path": fp, "n_samples": int(len(Vs_post))}
    except Exception as e:
        return {"success": False, "error": str(e)}


@eel.expose
def save_hvsr_svg(f_min=0.5, f_max=50.0, show_samples=False):
    """Render the current HVSR figure (spectra + HVSR with overlays) to SVG."""
    if state['hvsr'] is None:
        return {"success": False, "error": "Compute HVSR first"}
    fp = _save_dialog("Save HVSR figure as SVG", "hvsr.svg", "SVG files=*.svg")
    if not fp:
        return {"success": False, "error": "Cancelled"}
    try:
        fig = _build_hvsr_figure(float(f_min), float(f_max), bool(show_samples))
        fig.savefig(fp, format='svg', facecolor=fig.get_facecolor())
        plt.close(fig)
        return {"success": True, "path": fp}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@eel.expose
def save_model_svg(v_min=None, v_max=None, d_min=None, d_max=None):
    """Render the velocity model figure (initial + best + MCMC traces) to SVG."""
    if state['Vs_init'] is None:
        return {"success": False, "error": "Run inversion first"}
    fp = _save_dialog("Save velocity model as SVG", "velocity_model.svg", "SVG files=*.svg")
    if not fp:
        return {"success": False, "error": "Cancelled"}
    try:
        fig, _bounds = _build_model_figure(v_min, v_max, d_min, d_max)
        fig.savefig(fp, format='svg', facecolor=fig.get_facecolor())
        plt.close(fig)
        return {"success": True, "path": fp}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════
eel.init('web')

if __name__ == '__main__':
    eel.start('index.html', size=(1500, 980))
