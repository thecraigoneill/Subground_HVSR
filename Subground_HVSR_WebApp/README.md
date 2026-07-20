HVSR Inversion Viewer
=====================

Web app for HVSR processing and 1D inversion. Algorithms ported from
Subground_HVSR (https://github.com/thecraigoneill/Subground_HVSR, MIT licence)
by Craig O'Neill.

Layout
------
- TOP    File controls bar (Open + auto-detected params + Compute HVSR)
- ROW 2  Waveform strip with channel radios + mask-drawing radio


<img width="1874" height="1133" alt="Screenshot 2026-07-16 at 10 28 58 AM" src="https://github.com/user-attachments/assets/cb0b020a-1d12-4caa-854e-9f137f5e94f7" />


- ROW 3  Two-panel HVSR plot (spectra above, HVSR below)  +
         Velocity model (with axis controls) and Initial Model table on right
- ROW 4  Bottom of right column: inversion settings + INVERT button

Usage
-----
1. Run from Terminal (on Mac) or Anaconda prompt:
       python HVSR_Viewer.py

2. Click "Open .dat File". Sampling rate and header length are detected
   automatically from the Tromino header. Override and click
   "Apply & Reload" if needed.

3. Mask noisy waveform sections (optional):
     - Click "Draw mask" radio in the Waveform row.
     - Click twice on the waveform; the time range is shaded red and excluded
       from the HVSR computation.
     - Switch back to "View" to read mouse-over time.

4. Click "Compute HVSR". The HVSR plot now shows:
     - Top: log-log of N, E, Z spectral amplitudes
     - Bottom: HVSR curve, normalised so the baseline ≈ 1
     - Dotted grey line at 1.0 is the baseline reference
  
<img width="1712" height="813" alt="Screenshot 2026-07-16 at 10 29 08 AM" src="https://github.com/user-attachments/assets/98bbc667-e230-4ea4-992c-486c6e8dcdec" />


5. Edit the Initial Model table (Vs, h, Ds, Dp per layer; last layer is the
   half-space — use a large h such as 1000 m).

6. Click "INVERT HVSR". Initial and final HVSR curves overlay on the
   central plot; initial and final Vs profiles appear in the model panel.

7. Adjust the velocity-model axes using Vs min/max and Depth min/max boxes
   to zoom into relevant depth or velocity ranges; click Apply axis. The
   default auto-bounds appear as placeholder text. Reset returns to auto.


<img width="1765" height="757" alt="Screenshot 2026-07-16 at 10 29 15 AM" src="https://github.com/user-attachments/assets/eec3638b-8a81-4895-b219-1a5b6e7b1972" />



9. Save HVSR CSV / Save inversion CSV to export results.

Requirements
------------
- Python 3.8+
- numpy, scipy, matplotlib, eel
- tkinter (in the standard library on most installs)

Install:  pip install eel numpy scipy matplotlib

Notes
-----
- Forward HVSR model: HV3 algorithm from Subground_HVSR
  (Albarello et al. / Herak / OpenHVSR lineage), ported to pure NumPy.
- Vp and density derived from Vs via Poisson ratio and a Gardner-style
  density relation (1740 * (Vp/1000)^0.25), following Subground_HVSR (for now).
- Observed HVSR is normalised by its median value so the baseline matches
  the simulated curve (~1.0).
- Time-domain masks are excluded from the Welch periodogram, so masking
  bad windows directly removes their contribution to the spectra.
- File dialogs run in a subprocess (_filedialog.py) so they appear in
  front of Chrome reliably on Windows and macOS (hopefully?)

