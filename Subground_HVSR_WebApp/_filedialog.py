"""
Standalone file dialog helper — runs in its own subprocess so it has its own
event loop and does not fight with Eel/Chrome for focus.

Usage:
    python _filedialog.py open    "Title"  "label1=*.ext|label2=*.*"
    python _filedialog.py save    "Title"  "default.ext"  "label1=*.ext"
    python _filedialog.py folder  "Title"

Prints chosen path to stdout. Empty stdout == cancelled.
On real errors, writes a message to stderr AND exits non-zero so the caller
can distinguish a genuine failure from a user cancel.

macOS note: uses the native `osascript` chooser and NEVER imports tkinter,
because conda/Homebrew Pythons on macOS frequently ship a broken _tkinter
that aborts on import — which previously surfaced as a bogus "Cancelled".
"""
import sys
import os
import json
import subprocess


def parse_filetypes(spec):
    """Parse 'Label=*.ext|Label2=*.*' into [(label, pattern), ...]."""
    if not spec:
        return [("All files", "*.*")]
    out = []
    for part in spec.split('|'):
        if '=' in part:
            lbl, pat = part.split('=', 1)
            out.append((lbl.strip(), pat.strip()))
    return out or [("All files", "*.*")]


def _extensions(spec):
    """Extract bare extensions (e.g. ['dat','txt']) from a filetype spec.

    Returns [] if a wildcard (*.* / *) is present, meaning "no restriction".
    """
    exts = []
    for _lbl, pat in parse_filetypes(spec):
        pat = pat.strip()
        if pat in ('*.*', '*', ''):
            return []                      # wildcard -> allow everything
        if pat.startswith('*.'):
            exts.append(pat[2:].lower())
        elif pat.startswith('.'):
            exts.append(pat[1:].lower())
    return exts


# ── macOS: native chooser via AppleScript (no tkinter dependency) ──────────────
def _osa(script):
    proc = subprocess.run(['osascript', '-e', script],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or '')
        # User pressed Cancel -> AppleScript error -128 / "User canceled."
        if 'User canceled' in err or 'User cancelled' in err or '-128' in err:
            return ''                      # treated as a clean cancel
        raise RuntimeError(err.strip() or f"osascript exited {proc.returncode}")
    return (proc.stdout or '').strip()


def _mac_open(title, spec):
    q = json.dumps(title)                  # AppleScript string literal (JSON-safe)
    exts = _extensions(spec)
    if exts:
        of_type = '{' + ', '.join(json.dumps(e) for e in exts) + '}'
        script = f'POSIX path of (choose file with prompt {q} of type {of_type})'
    else:
        script = f'POSIX path of (choose file with prompt {q})'
    return _osa(script)


def _mac_save(title, initfn):
    q = json.dumps(title)
    if initfn:
        script = (f'POSIX path of (choose file name with prompt {q} '
                  f'default name {json.dumps(initfn)})')
    else:
        script = f'POSIX path of (choose file name with prompt {q})'
    return _osa(script)


def _mac_folder(title):
    return _osa(f'POSIX path of (choose folder with prompt {json.dumps(title)})')


# ── Non-macOS: Tkinter chooser ─────────────────────────────────────────────────
def _tk_root():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    try:
        root.call('wm', 'attributes', '.', '-topmost', True)
    except Exception:
        pass
    root.lift()
    root.focus_force()
    root.update()
    return root


def _tk_open(title, spec):
    from tkinter import filedialog
    root = _tk_root()
    try:
        return filedialog.askopenfilename(parent=root, title=title,
                                          filetypes=parse_filetypes(spec)) or ''
    finally:
        try: root.destroy()
        except Exception: pass


def _tk_save(title, initfn, spec):
    from tkinter import filedialog
    root = _tk_root()
    defext = ''
    if initfn and '.' in initfn:
        defext = '.' + initfn.rsplit('.', 1)[-1]
    try:
        return filedialog.asksaveasfilename(
            parent=root, title=title, initialfile=initfn,
            defaultextension=defext, filetypes=parse_filetypes(spec)) or ''
    finally:
        try: root.destroy()
        except Exception: pass


def _tk_folder(title):
    from tkinter import filedialog
    root = _tk_root()
    try:
        return filedialog.askdirectory(parent=root, title=title) or ''
    finally:
        try: root.destroy()
        except Exception: pass


def main():
    if len(sys.argv) < 2:
        sys.exit(0)
    mode = sys.argv[1]
    is_mac = (sys.platform == 'darwin')

    try:
        if mode == 'open':
            title = sys.argv[2] if len(sys.argv) > 2 else 'Open'
            spec  = sys.argv[3] if len(sys.argv) > 3 else ''
            result = _mac_open(title, spec) if is_mac else _tk_open(title, spec)
        elif mode == 'save':
            title  = sys.argv[2] if len(sys.argv) > 2 else 'Save'
            initfn = sys.argv[3] if len(sys.argv) > 3 else ''
            spec   = sys.argv[4] if len(sys.argv) > 4 else ''
            result = _mac_save(title, initfn) if is_mac else _tk_save(title, initfn, spec)
        elif mode == 'folder':
            title  = sys.argv[2] if len(sys.argv) > 2 else 'Folder'
            result = _mac_folder(title) if is_mac else _tk_folder(title)
        else:
            result = ''
    except Exception as e:
        # Real failure (e.g. tkinter missing, osascript blocked): report it so
        # the app can show the true cause instead of a misleading "Cancelled".
        sys.stderr.write("Dialog error: " + str(e) + "\n")
        sys.stderr.flush()
        sys.exit(2)

    sys.stdout.write(result or '')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
