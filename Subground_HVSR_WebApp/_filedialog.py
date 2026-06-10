"""
Standalone file dialog helper — runs in its own subprocess so it has its own
Tkinter event loop and does not fight with Eel/Chrome for focus.

Usage:
    python _filedialog.py open    "Title"  "label1=*.ext|label2=*.*"
    python _filedialog.py save    "Title"  "default.ext"  "label1=*.ext"
    python _filedialog.py folder  "Title"

Prints chosen path to stdout. Empty == cancelled.
"""
import sys
import os
import subprocess

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    sys.exit(1)


def parse_filetypes(spec):
    if not spec:
        return [("All files", "*.*")]
    out = []
    for part in spec.split('|'):
        if '=' in part:
            lbl, pat = part.split('=', 1)
            out.append((lbl.strip(), pat.strip()))
    return out or [("All files", "*.*")]


def activate_self_macos():
    """Bring this Python process to the macOS foreground via osascript."""
    if sys.platform != 'darwin':
        return
    try:
        pid = str(os.getpid())
        script = (
            'tell application "System Events"\n'
            '  set frontmost of first process whose unix id is ' + pid + ' to true\n'
            'end tell'
        )
        subprocess.call(['osascript', '-e', script], timeout=2)
    except Exception:
        pass


def main():
    if len(sys.argv) < 2:
        sys.exit(0)
    mode = sys.argv[1]

    activate_self_macos()

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

    result = ''
    try:
        if mode == 'open':
            title  = sys.argv[2] if len(sys.argv) > 2 else 'Open'
            ftypes = parse_filetypes(sys.argv[3] if len(sys.argv) > 3 else '')
            result = filedialog.askopenfilename(parent=root, title=title, filetypes=ftypes)
        elif mode == 'save':
            title  = sys.argv[2] if len(sys.argv) > 2 else 'Save'
            initfn = sys.argv[3] if len(sys.argv) > 3 else ''
            ftypes = parse_filetypes(sys.argv[4] if len(sys.argv) > 4 else '')
            defext = ''
            if initfn and '.' in initfn:
                defext = '.' + initfn.rsplit('.', 1)[-1]
            result = filedialog.asksaveasfilename(
                parent=root, title=title,
                initialfile=initfn, defaultextension=defext, filetypes=ftypes
            )
        elif mode == 'folder':
            title  = sys.argv[2] if len(sys.argv) > 2 else 'Folder'
            result = filedialog.askdirectory(parent=root, title=title)
    except Exception as e:
        sys.stderr.write("Dialog error: " + str(e) + "\n")
        result = ''
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    sys.stdout.write(result or '')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
