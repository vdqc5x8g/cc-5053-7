"""Audit tools for the sustained-vibration refactor.

These functions exist so the behaviour the refactor is supposed to
guarantee can be measured rather than asserted by eye:

* ``isolated_vs_accumulated`` - render each hit as its own response, then
  render both hits added to one growing modal state, and compare. Because
  the solver is linear, superposition must hold exactly; the function
  reports the residual and which modes cancelled in the overlap.
* ``replay_blocks`` - render one engine session with several block sizes
  and report worst-case differences, so a block edge that zeroed state
  would show up as a mismatch.
* ``trace_cancellation`` - per-mode stems for a pair of hits, exposing
  which modes are cancelled and which add, through the block edges.
* ``mode_tail_readback`` - instantaneous phase / amplitude / tail of each
  mode from the engine's live state.

Nothing here mutates the caller's engine; every experiment builds fresh
engines from described hits.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .synth import HitEvent, ModalEngine, ModalInstrument


def _fresh_engine(instrument: ModalInstrument, sample_rate_hz: float) -> ModalEngine:
    return ModalEngine(instrument, sample_rate_hz=sample_rate_hz)


def isolated_vs_accumulated(
    instrument: ModalInstrument,
    events: list[HitEvent],
    total_s: float,
    block_size: int = 512,
) -> dict:
    """Compare per-hit independent responses with one accumulated state.

    Independent: one engine per event, each struck at t=0, rendered for
    total_s (the first hit's tail is still present when the second arrives
    only in the accumulated run - isolated runs are referenced to their own
    own strike instant, so their placement on the shared time axis is
    exactly the event time).

    Accumulated: one engine, events submitted at their own times, rendered
    once in blocks. Linear superposition says the accumulated signal equals
    the sum of the isolated signals, delayed to each event time. The
    reported residual is max |accumulated - superposition| relative to the
    accumulated peak.
    """
    sr = instrument.sample_rate_hz
    n_total = int(round(total_s * sr))
    superposed = np.zeros(n_total, dtype=np.float64)
    isolated = []
    for ev in events:
        eng = _fresh_engine(instrument, sr)
        ev0 = dataclasses.replace(ev, time_s=0.0)
        eng.submit(ev0)
        buf = eng.render_blocks(n_total, block_size)
        start = int(round(ev.time_s * sr))
        if start < n_total:
            superposed[start:] += buf[: n_total - start]
        isolated.append(buf)

    acc = _fresh_engine(instrument, sr)
    for ev in events:
        acc.submit(ev)
    acc_out = acc.render_blocks(n_total, block_size)

    peak = float(np.max(np.abs(acc_out))) if acc_out.size else 0.0
    residual = float(np.max(np.abs(acc_out - superposed))) if acc_out.size else 0.0
    return {
        "sample_rate_hz": sr,
        "n_samples": n_total,
        "accumulated": acc_out,
        "superposed": superposed,
        "isolated": isolated,
        "peak": peak,
        "residual": residual,
        "residual_rel": residual / peak if peak > 0.0 else 0.0,
        "per_hit_reports": [r for r in acc.reports],
    }


def overlap_cancellation(
    reports,
    n_hits: int,
    rel_floor: float = 1e-6,
) -> dict:
    """Classify each mode by the sum of per-hit kicks over a hit sequence.

    ``reports`` is a list of HitReport (one per event). A mode whose summed
    kick is small relative to the largest single kick of that mode has been
    cancelled by the overlapping hits; a mode whose summed kick exceeds the
    largest single kick is reinforced. The distinction is reported per mode
    with the signed sum, so the caller can see which modes are on opposite
    sides of their nodal lines.
    """
    if not reports:
        return {"cancelled": [], "reinforced": [], "unchanged": [], "sums": []}
    n_modes = len(reports[0].modes)
    vectors = [np.array([m.delta_v_m_s for m in r.modes]) for r in reports]
    total = np.sum(vectors, axis=0)
    big = np.max(np.abs(vectors), axis=0)
    cancelled, reinforced, unchanged = [], [], []
    for i in range(n_modes):
        ratio = abs(total[i]) / big[i] if big[i] > 0.0 else 0.0
        if big[i] == 0.0:
            unchanged.append(i)
        elif ratio <= rel_floor:
            cancelled.append(i)
        elif ratio > 1.0 + 1e-9:
            reinforced.append(i)
        else:
            unchanged.append(i)
    return {
        "cancelled": cancelled,
        "reinforced": reinforced,
        "unchanged": unchanged,
        "sums": total.tolist(),
        "largest_single": big.tolist(),
    }


def replay_blocks(
    instrument: ModalInstrument,
    events: list[HitEvent],
    total_s: float,
    block_sizes: list[int],
) -> dict:
    """Render one session with several block lengths; report disagreement.

    Each block size runs on its own fresh engine with the same events. The
    reference is the largest block size. Any state zeroed at a block edge
    would appear as a difference of that block's span; the per-block
    maximum difference is returned so the location of a discrepancy is
    visible, not just its existence.
    """
    sr = instrument.sample_rate_hz
    n = int(round(total_s * sr))
    outputs = {}
    for bs in block_sizes:
        eng = _fresh_engine(instrument, sr)
        for ev in events:
            eng.submit(ev)
        outputs[bs] = eng.render_blocks(n, bs)
    ref_size = max(block_sizes)
    ref = outputs[ref_size]
    report = {"sample_rate_hz": sr, "n_samples": n, "reference_block": ref_size}
    comparisons = {}
    for bs, out in outputs.items():
        diff = np.abs(out - ref)
        # Per-block maxima over the reference size make edge effects local.
        edges = {}
        for k in range(0, n, ref_size):
            m = float(np.max(diff[k : k + ref_size])) if n > k else 0.0
            if m > 0.0:
                edges[k] = m
        comparisons[bs] = {
            "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
            "max_rel_diff": (
                float(np.max(diff)) / float(np.max(np.abs(ref)))
                if ref.size and np.max(np.abs(ref)) > 0.0
                else 0.0
            ),
            "diff_at_block_edges": edges,
        }
    report["comparisons"] = comparisons
    report["outputs"] = outputs
    return report


def trace_cancellation(
    instrument: ModalInstrument,
    events: list[HitEvent],
    total_s: float,
    mode_indices: list[int],
    block_size: int = 256,
) -> dict:
    """Per-mode stems through block edges for a sequence of hits.

    For each event, an isolated engine renders that event alone, and an
    accumulated engine renders all events into one voice. The per-mode
    stems of both are returned alongside their sums, so the caller can see
    a mode cancelled by superposition while a neighbour is reinforced, and
    can verify that a block edge does not reset either.
    """
    sr = instrument.sample_rate_hz
    n = int(round(total_s * sr))

    stems_isolated = []
    for ev in events:
        eng = _fresh_engine(instrument, sr)
        eng.submit(dataclasses.replace(ev, time_s=0.0))
        out, stems = eng.voice.render_block(n, stem_modes=mode_indices)
        stems_isolated.append(stems)

    acc = _fresh_engine(instrument, sr)
    for ev in events:
        acc.submit(ev)
    acc_out, acc_stems = acc.voice.render_block(n, stem_modes=mode_indices)

    blocks = [
        {
            "start": k,
            "end": min(k + block_size, n),
            "rms_accumulated": {
                i: float(np.sqrt(np.mean(acc_stems[i][k : k + block_size] ** 2)))
                for i in mode_indices
            },
        }
        for k in range(0, n, block_size)
    ]
    return {
        "sample_rate_hz": sr,
        "n_samples": n,
        "accumulated": acc_out,
        "accumulated_stems": acc_stems,
        "isolated_stems": stems_isolated,
        "blocks": blocks,
    }


def mode_tail_readback(
    engine: ModalEngine,
    mode_indices=None,
    n_tail: float = 1.0,
    floor_ratio: float | None = None,
) -> dict:
    """Live per-mode phase / amplitude / tail, plus the summed report vector.

    ``floor_ratio`` switches ``tail_s`` from the intrinsic e-folding time to
    "time until this mode falls below floor_ratio of the loudest mode",
    which reads 0 for a cancelled mode.
    """
    states = engine.body_state(n_tail=n_tail, floor_ratio=floor_ratio)
    idx = range(len(states)) if mode_indices is None else mode_indices
    rows = []
    for i in idx:
        s = states[i]
        mode = engine.instrument.modes[i]
        rows.append(
            {
                "index": i,
                "label": mode.label,
                "frequency_hz": mode.frequency_hz,
                "tau_s": mode.tau_s,
                "amplitude": s.amplitude,
                "phase_rad": s.phase_rad,
                "phase_deg": np.degrees(s.phase_rad),
                "tail_s": s.tail_s,
                "displacement": s.displacement,
                "velocity": s.velocity,
            }
        )
    return {"position": engine.position, "time_s": engine.time_s, "modes": rows}
