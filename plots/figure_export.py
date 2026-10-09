# plots/figure_export.py
"""
One gallery folder for EVERY figure of the project, each with the data points it shows (Oct 4, 2026).

After `install()`, every matplotlib `fig.savefig(path)` in the process ALSO:
    1. copies the image to  plots/figures/<source>__<name>.png
    2. writes               plots/figures/<source>__<name>.csv   - the data points actually drawn
<source> is the experiment folder the figure belongs to (outputs/<source>/plots/figures/x.png ->
"<source>__x.png"), so equally named figures of different phases never overwrite each other.
The original file is still written where it always was (nothing else changes).

The CSV is read back from the figure itself (lines, bars, scatter points, heat-map cells), so it is
exactly what the reader sees - one row per point:
    panel, series, kind, x, y, x_label, x_axis, y_axis
(x_label = the category name for bar / categorical axes; for heat maps x/y are the column/row
labels and `value` holds the cell.)
Used by plots/plot_results.py and every tool/experiment script that saves a figure.
"""
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GALLERY = PROJECT_ROOT / "plots" / "figures"
_INSTALLED = False


def _gallery_name(path):
    p = Path(path).resolve()
    try:
        rel = p.relative_to(PROJECT_ROOT / "outputs")
        src = rel.parts[0] if len(rel.parts) > 1 else ""
    except ValueError:
        src = p.parent.name
    if p.parent == GALLERY.resolve():
        return None                                   # already saved into the gallery
    return f"{src}__{p.stem}" if src else p.stem


def _is_num(txt):
    try:
        float(txt.replace("\u2212", "-").replace(",", ""))
        return True
    except ValueError:
        return False


def _tick_map(ticks, labels):
    """Category names of a categorical axis (numeric tick labels are ignored)."""
    return {round(float(t), 6): l.get_text() for t, l in zip(ticks, labels)
            if l.get_text() and not _is_num(l.get_text())}


def _label(m, v):
    if not m:
        return ""
    k = min(m, key=lambda t: abs(t - v))
    return m[k] if abs(k - v) < 0.5 else ""


def figure_points(fig):
    """Long-format rows of everything plotted in `fig`."""
    from matplotlib.container import BarContainer
    from matplotlib.collections import PathCollection
    from matplotlib.image import AxesImage
    rows = []
    for ai, ax in enumerate(fig.axes):
        panel = ax.get_title() or f"panel{ai + 1}"
        xa, ya = ax.get_xlabel(), ax.get_ylabel()
        xmap = _tick_map(ax.get_xticks(), ax.get_xticklabels())
        ymap = _tick_map(ax.get_yticks(), ax.get_yticklabels())
        base = {"panel": panel, "x_axis": xa, "y_axis": ya}
        in_bars = set()
        for ci, c in enumerate(ax.containers):
            if not isinstance(c, BarContainer):
                continue
            name = c.get_label() if c.get_label() and not c.get_label().startswith("_") else f"bars{ci + 1}"
            horiz = getattr(c, "orientation", "vertical") == "horizontal"
            for p in c.patches:
                in_bars.add(id(p))
                if horiz:
                    y = p.get_y() + p.get_height() / 2
                    rows.append({**base, "series": name, "kind": "barh", "x": p.get_width(), "y": y,
                                 "x_label": _label(ymap, y)})
                else:
                    x = p.get_x() + p.get_width() / 2
                    rows.append({**base, "series": name, "kind": "bar", "x": x, "y": p.get_height(),
                                 "x_label": _label(xmap, x)})
        for li, ln in enumerate(ax.get_lines()):
            xs, ys = ln.get_xdata(orig=True), ln.get_ydata(orig=True)
            lab = ln.get_label()
            name = lab if lab and not lab.startswith("_") else f"line{li + 1}"
            kind = "reference_line" if ln.get_transform() != ax.transData else "line"
            for x, y in zip(list(xs), list(ys)):
                try:
                    xf = float(x)
                except (TypeError, ValueError):
                    xf = x
                rows.append({**base, "series": name, "kind": kind, "x": xf, "y": y,
                             "x_label": _label(xmap, xf) if isinstance(xf, float) else str(x)})
        for si, col in enumerate(ax.collections):
            if isinstance(col, PathCollection):
                lab = col.get_label()
                name = lab if lab and not lab.startswith("_") else f"scatter{si + 1}"
                for x, y in col.get_offsets():
                    rows.append({**base, "series": name, "kind": "scatter", "x": float(x), "y": float(y),
                                 "x_label": _label(xmap, float(x))})
        for im in ax.images:
            if isinstance(im, AxesImage):
                a = im.get_array()
                for r in range(a.shape[0]):
                    for c in range(a.shape[1]):
                        rows.append({**base, "series": "heatmap", "kind": "cell",
                                     "x": _label(xmap, c) or c, "y": _label(ymap, r) or r,
                                     "value": float(a[r, c]), "x_label": _label(xmap, c)})
        for t in ax.texts:                                         # value labels / CD-diagram names
            txt = t.get_text().strip()
            if txt:
                x, y = t.get_position()
                rows.append({**base, "series": "text", "kind": "text", "x": x, "y": y, "x_label": txt})
    return rows


def export(fig, path):
    name = _gallery_name(path)
    if name is None:
        return
    try:
        import pandas as pd
        GALLERY.mkdir(parents=True, exist_ok=True)
        src = Path(path)
        if src.exists():
            shutil.copyfile(src, GALLERY / f"{name}{src.suffix or '.png'}")
        import warnings
        with warnings.catch_warnings():                            # masked (NaN) points -> empty cells, quietly
            warnings.simplefilter("ignore")
            rows = figure_points(fig)
        cols = ["panel", "series", "kind", "x", "y", "value", "x_label", "x_axis", "y_axis"]
        df = pd.DataFrame(rows)
        df = df.reindex(columns=[c for c in cols if c in df.columns or c in ("panel", "series", "kind", "x", "y")])
        df.to_csv(GALLERY / f"{name}.csv", index=False)
    except Exception as e:                                         # never break the real figure
        print(f"[figure_export] {path}: data export skipped ({type(e).__name__}: {e})")


def install():
    """Patch Figure.savefig once per process (also covers plt.savefig)."""
    global _INSTALLED
    if _INSTALLED:
        return
    from matplotlib.figure import Figure
    orig = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        out = orig(self, fname, *args, **kwargs)
        if isinstance(fname, (str, Path)):
            export(self, fname)
        return out

    Figure.savefig = savefig
    _INSTALLED = True
