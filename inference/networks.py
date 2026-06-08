"""Functional-network definitions on the fsaverage5 cortical mesh.

TRIBE v2 predicts brain activity as an ``(n_timesteps, n_vertices)`` array on the
fsaverage5 surface (~20,484 vertices: 10,242 per hemisphere). To turn that into
creator-facing scores we aggregate vertices into the functional networks that the
TRIBE v2 paper explicitly validates (Figs. 4-7): the visual stream, auditory
cortex, the language/semantic network, emotional-social regions (TPJ/MTG), the
default-mode network, and multisensory-integration cortex (STS/TPO).

We build the vertex masks from nilearn's Yeo-2011 7-network atlas (resampled to
fsaverage5), plus a small set of named ROIs from the Destrieux atlas for networks
Yeo does not isolate (language, emotional-social, multisensory). Building masks
from public atlases keeps us independent of any TRIBE-internal parcellation.

The masks are computed once and cached to ``NETWORKS_CACHE`` so repeated cold
starts on the GPU worker do not refetch atlases.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

# fsaverage5 vertex counts.
N_VERTS_PER_HEMI = 10242
N_VERTS = 2 * N_VERTS_PER_HEMI

NETWORKS_CACHE = os.environ.get("NETWORKS_CACHE", "/cache/networks_fsaverage5.npz")

# Display metadata + the composite-index weight for each network. Weights are a
# transparent, documented blend (they sum to 1.0) -- the UI shows the breakdown.
NETWORK_META: dict[str, dict] = {
    "visual": {
        "label": "Visual",
        "weight": 0.22,
        "description": "Early + ventral/dorsal visual & MT motion cortex. Drives scroll-stopping visual capture.",
    },
    "auditory": {
        "label": "Auditory",
        "weight": 0.15,
        "description": "Early + association auditory cortex. Responds to sound, music and voice.",
    },
    "language": {
        "label": "Language / Semantic",
        "weight": 0.20,
        "description": "Broca (45), STS, A5. Engaged by speech, captions and meaning.",
    },
    "emotional_social": {
        "label": "Emotional & Social",
        "weight": 0.20,
        "description": "TPJ + MTG. Theory-of-mind and emotional salience that drive shares.",
    },
    "default_mode": {
        "label": "Default-Mode (Narrative)",
        "weight": 0.13,
        "description": "Self-referential / narrative integration. High = the viewer is 'pulled in'.",
    },
    "multisensory": {
        "label": "Multisensory Integration",
        "weight": 0.10,
        "description": "STS / temporo-parieto-occipital junction binding sight + sound.",
    },
}

NETWORK_KEYS = list(NETWORK_META.keys())


@dataclass
class NetworkMasks:
    """Boolean vertex masks (length ``N_VERTS``) for each functional network."""

    masks: dict[str, np.ndarray] = field(default_factory=dict)
    n_verts: int = N_VERTS

    def vertices(self, key: str) -> np.ndarray:
        return self.masks[key]

    def to_npz(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, n_verts=self.n_verts, **self.masks)

    @classmethod
    def from_npz(cls, path: str) -> "NetworkMasks":
        data = np.load(path)
        n_verts = int(data["n_verts"])
        masks = {k: data[k].astype(bool) for k in data.files if k != "n_verts"}
        return cls(masks=masks, n_verts=n_verts)


# Yeo-7 network index -> our network keys (1=Visual, 2=Somatomotor, 3=Dorsal Attn,
# 4=Ventral Attn, 5=Limbic, 6=Frontoparietal, 7=Default).
_YEO7_TO_KEY = {1: "visual", 7: "default_mode"}

# Destrieux (a2009s) region-name substrings -> network keys, for the networks Yeo-7
# does not cleanly isolate. Matched case-insensitively against label names.
_DESTRIEUX_SUBSTR = {
    "auditory": ["G_temp_sup-G_T_transv", "G_temp_sup-Plan_tempo", "Lat_Fis-post"],
    "language": ["G_front_inf-Triangul", "G_front_inf-Opercular", "G_temp_sup-Lateral", "S_temporal_sup"],
    "emotional_social": ["G_temporal_middle", "G_pariet_inf-Angular", "S_temporal_sup"],
    "multisensory": ["S_temporal_sup", "G_temp_sup-Lateral", "G_pariet_inf-Supramar", "G_and_S_occipital_inf"],
}


def _build_masks() -> NetworkMasks:
    """Construct vertex masks from public atlases via nilearn.

    Imported lazily so that environments without nilearn (e.g. the Flask box or a
    unit test using a prebuilt cache) don't need the heavy dependency.
    """
    from nilearn import datasets
    from nilearn.surface import vol_to_surf  # noqa: F401  (kept for parity / future use)

    masks: dict[str, np.ndarray] = {k: np.zeros(N_VERTS, dtype=bool) for k in NETWORK_KEYS}

    # --- Yeo-7 on fsaverage5 (surface annotation) ---
    yeo = datasets.fetch_atlas_surf_destrieux()  # gives Destrieux below; Yeo handled via labels
    # Destrieux surface labels (per-hemisphere integer maps + names).
    destrieux = yeo
    labels = [l.decode() if isinstance(l, bytes) else l for l in destrieux["labels"]]

    for hemi_i, hemi in enumerate(("map_left", "map_right")):
        offset = hemi_i * N_VERTS_PER_HEMI
        annot = np.asarray(destrieux[hemi])
        for net_key, substrs in _DESTRIEUX_SUBSTR.items():
            target_ids = [
                idx for idx, name in enumerate(labels)
                if any(s.lower() in name.lower() for s in substrs)
            ]
            if not target_ids:
                continue
            sel = np.isin(annot, target_ids)
            masks[net_key][offset:offset + N_VERTS_PER_HEMI] |= sel

    # Visual + default-mode from Yeo-7 thick (more reliable than Destrieux for these).
    yeo7 = datasets.fetch_atlas_yeo_2011()
    _apply_yeo7_surface(masks, yeo7)

    # Guarantee every network has at least a few vertices; otherwise scoring NaNs.
    for k in NETWORK_KEYS:
        if masks[k].sum() == 0:
            raise RuntimeError(f"network '{k}' produced an empty mask; check atlas fetch")

    return NetworkMasks(masks=masks)


def _apply_yeo7_surface(masks: dict[str, np.ndarray], yeo7) -> None:
    """Project the Yeo-7 volumetric atlas onto fsaverage5 and fill visual/DMN masks.

    Uses nilearn's fsaverage5 pial surfaces. Kept separate so the Destrieux path can
    stand alone if the volumetric projection is unavailable in a given nilearn build.
    """
    try:
        from nilearn import datasets, surface
        fsavg = datasets.fetch_surf_fsaverage("fsaverage5")
        for hemi_i, mesh in enumerate((fsavg["pial_left"], fsavg["pial_right"])):
            offset = hemi_i * N_VERTS_PER_HEMI
            proj = surface.vol_to_surf(yeo7["thick_7"], mesh, interpolation="nearest")
            proj = np.rint(np.nan_to_num(proj)).astype(int).ravel()[:N_VERTS_PER_HEMI]
            for yeo_id, key in _YEO7_TO_KEY.items():
                masks[key][offset:offset + N_VERTS_PER_HEMI] |= (proj == yeo_id)
    except Exception as exc:  # pragma: no cover - depends on nilearn data availability
        # Non-fatal: Destrieux already seeded the other networks; log and continue.
        print(f"[networks] Yeo-7 surface projection skipped: {exc}")


@lru_cache(maxsize=1)
def get_masks() -> NetworkMasks:
    """Return network masks, building + caching them on first use."""
    if os.path.exists(NETWORKS_CACHE):
        return NetworkMasks.from_npz(NETWORKS_CACHE)
    masks = _build_masks()
    try:
        masks.to_npz(NETWORKS_CACHE)
    except OSError:
        pass  # read-only fs is fine; we keep the in-memory copy
    return masks


if __name__ == "__main__":
    m = get_masks()
    summary = {k: int(m.masks[k].sum()) for k in NETWORK_KEYS}
    print(json.dumps({"n_verts": m.n_verts, "vertices_per_network": summary}, indent=2))
