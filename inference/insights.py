"""Rule-based, plain-language insights from a score profile.

Translates the network scores + timeline (scoring.ScoreResult) into a short list of
creator-facing cards: what's working, what to try, and pacing tips. Deterministic
and explainable on purpose -- no second model, no hallucinated claims, and every
card maps to a concrete pattern in the predicted brain activity.
"""

from __future__ import annotations

import numpy as np

from scoring import ScoreResult

# Thresholds on the 0-100 scale (50 = typical content by construction).
HIGH = 65.0
LOW = 38.0


def _net(networks: list[dict]) -> dict[str, float]:
    return {n["key"]: n["score"] for n in networks}


def generate_insights(result: ScoreResult, modality: str, experimental: bool) -> list[dict]:
    cards: list[dict] = []
    s = _net(result.networks)

    # --- Headline on the composite index ---
    idx = result.engagement_index
    if idx >= HIGH:
        cards.append(_card("positive", "Strong overall neural engagement",
                           f"Predicted activation sits in the top band (index {idx:.0f}/100). "
                           "The brain networks tied to attention and meaning are well driven."))
    elif idx <= LOW:
        cards.append(_card("warning", "Low predicted engagement",
                           f"The composite index is {idx:.0f}/100. Consider a stronger hook and "
                           "clearer sensory or emotional cues — see the tips below."))
    else:
        cards.append(_card("tip", "Solid, with room to grow",
                           f"Index {idx:.0f}/100 — around typical. Targeted tweaks to the weakest "
                           "networks below can lift reach potential."))

    # --- Visual ---
    if s.get("visual", 0) >= HIGH:
        cards.append(_card("positive", "Scroll-stopping visuals",
                           "Strong early-visual and motion (MT) activation — good thumbnail / first-frame appeal."))
    elif s.get("visual", 100) <= LOW and modality in ("video", "image"):
        cards.append(_card("tip", "Sharpen the visuals",
                           "Weak visual-cortex drive. Add motion, contrast, faces or a bold first frame."))

    # --- Language / captions ---
    if s.get("language", 0) >= HIGH:
        cards.append(_card("positive", "Clear verbal hook",
                           "The language/semantic network is well engaged — speech or captions are landing."))
    elif s.get("language", 100) <= LOW:
        msg = ("Little language-network activation. Add a spoken hook, on-screen captions, "
               "or a punchy text overlay.")
        cards.append(_card("tip", "Add a verbal/caption hook", msg))

    # --- Emotional-social (drives shares) ---
    if s.get("emotional_social", 0) >= HIGH:
        cards.append(_card("positive", "Emotionally resonant",
                           "TPJ/MTG activation suggests emotional and social salience — content people share."))
    elif s.get("emotional_social", 100) <= LOW:
        cards.append(_card("tip", "Raise emotional salience",
                           "Flat emotional-social response. A relatable face, stakes, or a story beat can help."))

    # --- Auditory ---
    if modality in ("video", "audio") and s.get("auditory", 100) <= LOW:
        cards.append(_card("tip", "Use sound deliberately",
                           "Low auditory engagement. Music, a voice, or a sound effect early can hold attention."))

    # --- Multisensory binding ---
    if modality == "video" and s.get("multisensory", 0) >= HIGH:
        cards.append(_card("positive", "Sight and sound work together",
                           "Strong multisensory integration — your audio reinforces the visuals."))

    # --- Pacing / hook from the timeline ---
    cards.extend(_pacing_insights(result))

    # --- Experimental modality caveat ---
    if experimental:
        cards.append(_card("warning", "Experimental input",
                           f"{modality.title()}-only inputs are a degenerate case for a model built on "
                           "video+audio+text. Treat these scores as directional, not precise."))

    return cards


def _pacing_insights(result: ScoreResult) -> list[dict]:
    t = np.asarray(result.timeline.get("t", []), dtype=float)
    overall = np.asarray(result.timeline.get("overall", []), dtype=float)
    if overall.size < 3:
        return []
    cards: list[dict] = []

    # Hook: compare first ~20% to the peak.
    first_n = max(1, int(round(len(overall) * 0.2)))
    early = float(overall[:first_n].mean())
    peak = float(overall.max())
    peak_t = float(t[int(np.argmax(overall))]) if t.size else 0.0

    if peak > 0 and early < 0.8 * peak:
        cards.append(_card("tip", "Front-load your hook",
                           f"Engagement peaks around {peak_t:.0f}s but the opening is weaker. "
                           "Move your strongest moment into the first second or two."))
    elif early >= 0.9 * peak:
        cards.append(_card("positive", "Great opening",
                           "Engagement is strong from the very first frames — ideal for short-form retention."))

    # Drop-off: last 20% vs middle.
    if len(overall) >= 5:
        last_n = max(1, int(round(len(overall) * 0.2)))
        tail = float(overall[-last_n:].mean())
        mid = float(overall[first_n:-last_n].mean()) if len(overall) > 2 * first_n else float(overall.mean())
        if mid > 0 and tail < 0.7 * mid:
            cards.append(_card("tip", "Tighten the ending",
                               "Predicted engagement falls off near the end. Trim the tail or add a payoff/CTA."))
    return cards


def _card(severity: str, title: str, body: str) -> dict:
    return {"severity": severity, "title": title, "body": body}
