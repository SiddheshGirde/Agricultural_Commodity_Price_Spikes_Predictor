r"""
ONE-TIME TOOL - run it once from the project root, check the output, then delete it (do not commit it).

Replaces hardcoded machine paths such as
    Path(r"C:\Users\siddh\Desktop\PBL\data\interim\news_clean.csv")
with
    (PROJECT_ROOT / "data/interim/news_clean.csv")
and adds `from config import PROJECT_ROOT` after the file's last top-level import, so the project
runs from any folder on any machine. config.PROJECT_ROOT is the folder above src/ (or $PBL_ROOT).

Usage (from the project root):
    python tools/make_portable.py --dry-run     # show what would change, touch nothing
    python tools/make_portable.py               # rewrite the files in place
    python tools/make_portable.py src/train.py  # only specific files

Safe to run twice (a second run changes nothing). Each rewritten file is compile-checked, and
anything still containing a hardcoded path is reported (exit code 1).
"""
import argparse
import ast
import difflib
import py_compile
import re
import sys
from pathlib import Path

ROOT_RAW = r"C:\\Users\\siddh\\Desktop\\PBL"          # regex for the path as written in a raw string
REST = r'((?:\\[^"\\]+)*)'                            # optional \sub\folder\file.csv
WRAPPED = re.compile(r'Path\(r"' + ROOT_RAW + REST + r'"\)')
BARE = re.compile(r'r"' + ROOT_RAW + REST + r'"')
LEFTOVER = re.compile(r"C:\\Users|C:/Users|siddh", re.IGNORECASE)
IMPORT_LINE = "from config import PROJECT_ROOT"


def rel_expr(rest: str) -> str:
    parts = [p for p in rest.split("\\") if p]
    return "PROJECT_ROOT" if not parts else 'PROJECT_ROOT / "' + "/".join(parts) + '"'


def convert(source: str):
    n = 0

    def wrapped(m):
        nonlocal n
        n += 1
        e = rel_expr(m.group(1))
        return e if e == "PROJECT_ROOT" else f"({e})"

    def bare(m):
        nonlocal n
        n += 1
        return f"str({rel_expr(m.group(1))})"

    out = WRAPPED.sub(wrapped, source)
    out = BARE.sub(bare, out)
    if n and IMPORT_LINE not in out and "import PROJECT_ROOT" not in out and "PROJECT_ROOT," not in out:
        tree = ast.parse(out)
        imports = [nd for nd in tree.body if isinstance(nd, (ast.Import, ast.ImportFrom))]
        lines = out.split("\n")
        at = imports[-1].end_lineno if imports else 0
        lines.insert(at, IMPORT_LINE)
        out = "\n".join(lines)
    return out, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    files = [Path(f) for f in a.files] or sorted(Path("src").glob("*.py")) + [Path("app.py")]
    skipped_config = {"config.py"}                       # config.py itself defines PROJECT_ROOT
    bad = 0
    for f in files:
        if not f.exists() or f.name in skipped_config:
            continue
        src = f.read_text(encoding="utf-8")
        new, n = convert(src)
        left = [i for i, ln in enumerate(new.split("\n"), 1) if LEFTOVER.search(ln)]
        if n == 0 and not left:
            print(f"  unchanged  {f}")
            continue
        print(f"  {'WOULD CHANGE' if a.dry_run else 'rewrote   '} {f}: {n} path(s) replaced")
        if a.dry_run and n:
            for ln in difflib.unified_diff(src.split("\n"), new.split("\n"), lineterm="", n=0):
                if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---")):
                    print("      " + ln)
        if not a.dry_run and n:
            f.write_text(new, encoding="utf-8")
            py_compile.compile(str(f), doraise=True)
        if left:
            bad += 1
            print(f"  !! {f} still contains a machine-specific path on line(s) {left} - edit by hand")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())