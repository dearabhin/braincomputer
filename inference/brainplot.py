"""Render per-network brain-activation thumbnails from a TRIBE prediction.

TRIBE v2 predicts brain activity as ``(n_timesteps, n_vertices)`` on the fsaverage5
cortical surface. ``scoring.py`` collapses that into per-network scalar scores; here we
keep the spatial information and paint each functional network's vertices onto an inflated
brain, tinted by how strongly that network was driven (mean |activation| over time).

Output is a small base64-encoded PNG per network, embedded inline in the result JSON so the
frontend can show them as ``<img>`` thumbnails with no extra serving infrastructure.

Imported by bare name on the Modal worker (see CLAUDE.md gotcha) -- it must be listed in
``add_local_python_source`` in modal_app.py to be bundled into the image. nilearn +
matplotlib are already present (nilearn depends on matplotlib).
"""

from __future__ import annotations

import base64
import io
import os
from functools import lru_cache

import numpy as np

# Headless rendering: the GPU worker has no display, so force the Agg backend before any
# pyplot import. matplotlib's 3D surface rasterization works fine without a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from networks import NETWORK_KEYS, NETWORK_META, N_VERTS_PER_HEMI, get_masks  # noqa: E402

# Persist nilearn's fsaverage meshes on the Modal Volume so they download once, not on
# every cold start (mirrors NETWORKS_CACHE in networks.py).
NILEARN_DATA = os.environ.get("NILEARN_DATA", "/cache/nilearn")


@lru_cache(maxsize=1)
def _fsaverage():
    """fsaverage5 inflated surfaces + sulcal-depth background, cached per container."""
    from nilearn import datasets

    try:
        os.makedirs(NILEARN_DATA, exist_ok=True)
        data_dir = NILEARN_DATA
    except OSError:
        data_dir = None  # read-only fs: fall back to nilearn's default location
    return datasets.fetch_surf_fsaverage("fsaverage5", data_dir=data_dir)


def _render_one(stat_map: np.ndarray, fsavg) -> str:
    """Render left+right lateral views of a single per-vertex stat map -> data-URL PNG."""
    from nilearn import plotting

    fig, axes = plt.subplots(
        1, 2, figsize=(3.6, 2.0), dpi=100, subplot_kw={"projection": "3d"}
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    hemis = (
        ("left", fsavg.infl_left, fsavg.sulc_left, stat_map[:N_VERTS_PER_HEMI]),
        ("right", fsavg.infl_right, fsavg.sulc_right, stat_map[N_VERTS_PER_HEMI:]),
    )
    for ax, (hemi, mesh, bg, data) in zip(axes, hemis):
        plotting.plot_surf_stat_map(
            mesh, data, bg_map=bg, hemi=hemi, view="lateral",
            cmap="inferno", colorbar=False, threshold=1e-6,
            bg_on_data=True, axes=ax, figure=fig,
        )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_network_maps(preds: np.ndarray) -> list[dict]:
    """Per-network inflated-brain thumbnails as inline base64 PNGs.

    Returns ``[{"key", "label", "image"}, ...]`` in NETWORK_KEYS order. Best-effort: a
    render failure for one network (or all of them) is swallowed so it never breaks the
    job -- the maps are a nice-to-have on top of the scores.
    """
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim != 2:
        return []

    # Mean magnitude per vertex over time -> (n_vertices,), matching scoring's |activation|
    # convention (scoring._network_activation).
    vert_act = np.abs(preds).mean(axis=0)
    masks = get_masks()

    try:
        fsavg = _fsaverage()
    except Exception:
        return []

    out: list[dict] = []
    for key in NETWORK_KEYS:
        try:
            mask = masks.vertices(key)
            if mask.shape[0] != vert_act.shape[0]:
                continue
            stat = np.where(mask, vert_act, 0.0).astype(np.float32)
            out.append({
                "key": key,
                "label": NETWORK_META[key]["label"],
                "image": _render_one(stat, fsavg),
            })
        except Exception:
            continue  # skip this network, keep the rest
    return out
