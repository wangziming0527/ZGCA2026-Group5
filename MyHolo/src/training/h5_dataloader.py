"""Simplified HDF5 motion cache backed by a PyTorch ``DataLoader``.

This module provides two core utilities:

* ``Hdf5MotionDataset`` – loads contiguous motion windows directly from HDF5
  shards using metadata stored in ``manifest.json``.
* ``MotionClipBatchCache`` – maintains a double-buffered cache of motion clips
  with deterministic swapping semantics suitable for high-throughput
  reinforcement learning.

Compared to the legacy slot-based prefetcher, this implementation keeps the
pipeline intentionally simple:

* A dataset-worker keeps shard handles open locally; no Ray dependency.
* Each cached batch has a fixed shape
  ``[max_num_clips, max_frame_length, feature_dims]``.
* Swapping a batch is handled via an O(1) pointer flip once the next batch is
  staged on the desired device (CPU or GPU).

The cache exposes helper methods that mirror the data access patterns required
by ``RefMotionCommand``:

* ``sample_env_assignments`` for initial clip/frame sampling.
* ``gather_state`` to fetch ``1 + n_future`` frames per environment.

All tensors returned by this module are ``torch.float32`` unless stated
otherwise; tensor shapes are noted explicitly in type annotations.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from loguru import logger
from tabulate import tabulate

import torch.distributed as dist  # type: ignore

from holomotion.src.utils import torch_utils

Tensor = torch.Tensor


def _configure_weighted_bins(
    keys: List[str],
    cfg: Mapping[str, Any],
    batch_size_for_log: int,
) -> Tuple[List[List[int]], List[float], List[Dict[str, Any]]]:
    """Common helper to parse config, assign bins, and compute batch fractions."""
    if batch_size_for_log <= 0:
        batch_size_for_log = 1

    cfg_local: Dict[str, Any] = dict(cfg or {})

    patterns_cfg = cfg_local.get("bin_regex_patterns")
    if not patterns_cfg:
        raise ValueError(
            "weighted_bin configuration requires 'bin_regex_patterns' "
            "(list of {regex, ratio}) to be configured"
        )

    compiled_patterns: List[Dict[str, Any]] = []
    ratios: List[float] = []
    for idx, entry in enumerate(patterns_cfg):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Entry {idx} in bin_regex_patterns must be a mapping, "
                f"got {type(entry)}"
            )
        regex_str = entry.get("regex", None)
        if not isinstance(regex_str, str) or not regex_str:
            raise ValueError(
                f"Entry {idx} in bin_regex_patterns is missing a non-empty "
                f"'regex' field"
            )
        ratio_val = entry.get("ratio", None)
        if ratio_val is None:
            raise ValueError(
                f"Entry {idx} in bin_regex_patterns is missing 'ratio'"
            )
        ratio_f = float(ratio_val)
        if ratio_f < 0.0 or ratio_f > 1.0:
            raise ValueError(
                f"Entry {idx} in bin_regex_patterns has invalid ratio "
                f"{ratio_f:.6f}; expected in [0.0, 1.0]"
            )
        compiled_patterns.append(
            {
                "name": str(entry.get("name", f"bin_{idx}")),
                "regex": regex_str,
                "compiled": re.compile(regex_str),
            }
        )
        ratios.append(ratio_f)

    sum_explicit = float(sum(ratios))
    if sum_explicit > 1.0 + 1.0e-6:
        raise ValueError(
            f"Sum of weighted-bin ratios is {sum_explicit:.6f} (> 1.0). "
            "Please reduce the ratios so that their sum is <= 1.0."
        )
    others_ratio = max(0.0, 1.0 - sum_explicit)

    if len(keys) == 0:
        raise ValueError(
            "weighted_bin configuration received an empty key set"
        )

    num_items_total = float(len(keys))
    num_explicit = len(compiled_patterns)
    bin_indices: List[List[int]] = [[] for _ in range(num_explicit + 1)]

    for idx, motion_key in enumerate(keys):
        assigned = False
        for b_idx, pat in enumerate(compiled_patterns):
            if pat["compiled"].search(motion_key):
                bin_indices[b_idx].append(idx)
                assigned = True
                break
        if not assigned:
            bin_indices[-1].append(idx)

    # Combine explicit ratios with implicit "others" ratio
    all_ratios: List[float] = list(ratios)
    all_ratios.append(others_ratio)

    # If all motion keys are covered by explicit regex bins, but the specified
    # ratios sum to less than 1.0, linearly reweight explicit ratios so that
    # they sum to 1.0 and disable the implicit "others" bin.
    others_count = len(bin_indices[-1])
    if others_count == 0 and others_ratio > 0.0 and sum_explicit > 0.0:
        scale = 1.0 / sum_explicit
        ratios = [r * scale for r in ratios]
        others_ratio = 0.0
        all_ratios = list(ratios)
        all_ratios.append(others_ratio)
        logger.info(
            "Weighted-bin: all regex bins cover the dataset; "
            "linearly reweighted explicit ratios to sum to 1.0 and disabled "
            "the implicit 'others' bin."
        )

    # Validate non-empty bins for any positive ratio (including others)
    for b_idx, r in enumerate(all_ratios):
        if r > 0.0 and len(bin_indices[b_idx]) == 0:
            if b_idx < num_explicit:
                name = compiled_patterns[b_idx]["name"]
                regex_s = compiled_patterns[b_idx]["regex"]
                raise ValueError(
                    f"Weighted-bin '{name}' (regex='{regex_s}') has ratio "
                    f"{r:.6f} but matched no motion keys"
                )
            raise ValueError(
                f"Weighted-bin 'others' has ratio {r:.6f} but matched no "
                "motion keys"
            )

    # Prepare logging summary using the configured cache batch size
    raw_counts_log = [ratio * batch_size_for_log for ratio in all_ratios]
    base_counts_log = [int(c) for c in raw_counts_log]
    residuals_log = [c - int(c) for c in raw_counts_log]
    remaining = batch_size_for_log - int(sum(base_counts_log))
    if remaining != 0:
        order = sorted(
            range(len(residuals_log)),
            key=lambda i: residuals_log[i],
            reverse=True,
        )
        idx_pos = 0
        while remaining > 0:
            j = order[idx_pos % len(order)]
            base_counts_log[j] += 1
            remaining -= 1
            idx_pos += 1
    batch_fractions_log = [
        float(c) / float(batch_size_for_log) for c in base_counts_log
    ]

    # Build specs using the final, actually used batch fractions
    specs: List[Dict[str, Any]] = []
    total_items = float(max(1, num_items_total))
    for b_idx in range(num_explicit):
        name = compiled_patterns[b_idx]["name"]
        regex_s = compiled_patterns[b_idx]["regex"]
        n = len(bin_indices[b_idx])
        ds_frac = float(n) / total_items
        bf = batch_fractions_log[b_idx]
        specs.append(
            {
                "name": name,
                "regex": regex_s,
                "ratio": bf,
                "count": n,
                "dataset_fraction": ds_frac,
                "batch_fraction": bf,
            }
        )
    # Others bin
    others_name = "others"
    others_regex = "<unmatched>"
    n_o = len(bin_indices[-1])
    ds_frac_o = float(n_o) / total_items
    bf_o = batch_fractions_log[-1]
    specs.append(
        {
            "name": others_name,
            "regex": others_regex,
            "ratio": bf_o,
            "count": n_o,
            "dataset_fraction": ds_frac_o,
            "batch_fraction": bf_o,
        }
    )

    return bin_indices, all_ratios, specs


def preview_weighted_bin_from_manifest(
    manifest_path: str | Sequence[str],
    batch_size: int,
    cfg: Mapping[str, Any],
) -> None:
    """Lightweight preview of weighted-bin sampling using manifest.json only.

    This helper is intended to be called at configuration time before any
    MotionClipBatchCache/DataLoader is constructed, so that invalid regex or
    ratio settings can fail fast without incurring the cost of cache setup.
    """
    if batch_size <= 0:
        batch_size = 1

    if isinstance(manifest_path, (str, os.PathLike)):
        manifest_paths: List[str] = [str(manifest_path)]
    else:
        manifest_paths = [str(p) for p in manifest_path]
    if len(manifest_paths) == 0:
        raise ValueError(
            "preview_weighted_bin_from_manifest requires at least one manifest path"
        )

    key_source: Dict[str, str] = {}
    for mp in manifest_paths:
        if not os.path.exists(mp):
            raise FileNotFoundError(
                f"HDF5 manifest not found at {mp}. "
                "Please set robot.motion.hdf5_root/train_hdf5_roots "
                "to the correct path."
            )
        with open(mp, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        clips = manifest.get("clips", {})
        if not clips:
            raise ValueError(
                f"Manifest at {mp} contains no clips; cannot preview "
                "weighted-bin sampling."
            )
        for key in clips.keys():
            if key in key_source:
                raise ValueError(
                    f"Duplicate motion clip key '{key}' found in multiple "
                    "manifests; clip keys must be globally unique."
                )
            key_source[key] = mp

    keys = list(key_source.keys())
    _, _, specs = _configure_weighted_bins(
        keys=keys,
        cfg=cfg,
        batch_size_for_log=batch_size,
    )

    table_rows = []
    for item in specs:
        table_rows.append(
            [
                item["name"],
                item["regex"],
                f"{item['ratio']:.4f}",
                int(item["count"]),
                f"{item['dataset_fraction']:.4f}",
                f"{item['batch_fraction']:.4f}",
            ]
        )
    headers = [
        "bin",
        "regex",
        "final_ratio",
        "num_clips",
        "clip_fraction",
        "batch_fraction",
    ]
    logger.info(
        "Weighted-bin config preview (manifest-level):\n"
        + tabulate(table_rows, headers=headers, tablefmt="simple_outline")
    )


class AbstractClipScorer:
    """Interface for clip score-based curriculum strategies."""

    def update(self, stats: Dict[str, float], step: int) -> None:
        raise NotImplementedError

    def probabilities(
        self, keys: List[str], step: int
    ) -> Optional[torch.Tensor]:
        """Optional: return normalized sampling probabilities for provided keys.
        If None is returned, the cache will compute probabilities by itself."""
        return None

    def scores(self, keys: List[str], step: int) -> torch.Tensor:
        """Return non-negative, unnormalized scores for the provided motion keys.
        Shape: [len(keys)]
        """
        raise NotImplementedError

    def on_sampled(
        self, keys: List[str], step: int, probs: Optional[torch.Tensor] = None
    ) -> None:
        """Notify scorer that the given keys were sampled at 'step'.
        probs: Optional vector of per-key sampling probabilities aligned with keys.
        """
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return


class AdvantageRecencyScorer(AbstractClipScorer):
    """EMA difficulty + recency bonus scorer (label-free).

    Score for key m:
      s_m = EMA_median_abs_advantage
      recency_bonus = 1 + kappa * min(1, steps_since_last/τ)
      progress_bonus = 1 + progress_beta * ema(relative_improvement)
      stagnation_decay = exp(-stagnation_beta * max(0, steps_since_improve)/stagnation_tau)
      S_m = (min(s_m, adv_cap) ** gamma) * recency_bonus * progress_bonus * stagnation_decay + epsilon
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        gamma: float = 1.5,
        kappa: float = 0.3,
        tau: int = 1000,
        epsilon: float = 1.0e-3,
        adv_cap: float = 0.0,
        progress_alpha: float = 0.1,
        progress_beta: float = 0.5,
        improve_threshold: float = 0.02,
        stagnation_tau: int = 2000,
        stagnation_beta: float = 0.5,
    ) -> None:
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.kappa = float(kappa)
        self.tau = int(max(1, tau))
        self.epsilon = float(max(1e-12, epsilon))
        self.adv_cap = float(max(0.0, adv_cap))
        self.progress_alpha = float(progress_alpha)
        self.progress_beta = float(progress_beta)
        self.improve_threshold = float(max(0.0, improve_threshold))
        self.stagnation_tau = int(max(1, stagnation_tau))
        self.stagnation_beta = float(stagnation_beta)
        self._ema: Dict[str, float] = {}
        self._last_step: Dict[str, int] = {}
        self._progress: Dict[str, float] = {}
        self._last_improve_step: Dict[str, int] = {}

    def update(self, stats: Dict[str, float], step: int) -> None:
        for k, v in stats.items():
            v_f = float(max(0.0, v))
            prev = self._ema.get(k, 1.0)
            ema = (1.0 - self.alpha) * prev + self.alpha * v_f
            self._ema[k] = ema
            # Track relative improvement
            rel_improve = 0.0
            if prev > 1.0e-8:
                rel_improve = max(0.0, (prev - ema) / prev)
            prog_prev = self._progress.get(k, 0.0)
            prog_new = (
                1.0 - self.progress_alpha
            ) * prog_prev + self.progress_alpha * rel_improve
            self._progress[k] = prog_new
            if rel_improve >= self.improve_threshold:
                self._last_improve_step[k] = int(step)
            # Do not touch last_step here; only set when actually sampled

    def scores(self, keys: List[str], step: int) -> torch.Tensor:
        out = []
        for k in keys:
            s = float(self._ema.get(k, 1.0))
            last = self._last_step.get(k, None)
            if last is None:
                since = self.tau
            else:
                since = max(0, step - last)
            recency = 1.0 + self.kappa * min(1.0, since / float(self.tau))
            # Saturate extreme advantage to avoid bad/outlier data dominating
            s_core = max(0.0, s)
            if self.adv_cap > 0.0:
                s_core = min(s_core, self.adv_cap)
            # Progress and stagnation
            prog = float(self._progress.get(k, 0.0))
            progress_bonus = 1.0 + self.progress_beta * prog
            last_impr = self._last_improve_step.get(k, None)
            if last_impr is None:
                since_impr = self.stagnation_tau
            else:
                since_impr = max(0, step - last_impr)
            stagnation_decay = float(
                torch.exp(
                    torch.tensor(
                        -self.stagnation_beta
                        * since_impr
                        / float(self.stagnation_tau),
                        dtype=torch.float32,
                    )
                ).item()
            )
            score = (
                s_core**self.gamma
            ) * recency * progress_bonus * stagnation_decay + self.epsilon
            out.append(score)
        return torch.tensor(out, dtype=torch.float32)

    def on_sampled(
        self, keys: List[str], step: int, probs: Optional[torch.Tensor] = None
    ) -> None:
        for k in keys:
            self._last_step[k] = int(step)

    def state_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "kappa": self.kappa,
            "tau": self.tau,
            "epsilon": self.epsilon,
            "adv_cap": self.adv_cap,
            "progress_alpha": self.progress_alpha,
            "progress_beta": self.progress_beta,
            "improve_threshold": self.improve_threshold,
            "stagnation_tau": self.stagnation_tau,
            "stagnation_beta": self.stagnation_beta,
            "ema": self._ema,
            "last_step": self._last_step,
            "progress": self._progress,
            "last_improve_step": self._last_improve_step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.alpha = float(state.get("alpha", self.alpha))
        self.gamma = float(state.get("gamma", self.gamma))
        self.kappa = float(state.get("kappa", self.kappa))
        self.tau = int(state.get("tau", self.tau))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.adv_cap = float(state.get("adv_cap", self.adv_cap))
        self.progress_alpha = float(
            state.get("progress_alpha", self.progress_alpha)
        )
        self.progress_beta = float(
            state.get("progress_beta", self.progress_beta)
        )
        self.improve_threshold = float(
            state.get("improve_threshold", self.improve_threshold)
        )
        self.stagnation_tau = int(
            state.get("stagnation_tau", self.stagnation_tau)
        )
        self.stagnation_beta = float(
            state.get("stagnation_beta", self.stagnation_beta)
        )
        self._ema = dict(state.get("ema", {}))
        self._last_step = dict(state.get("last_step", {}))
        self._progress = dict(state.get("progress", {}))
        self._last_improve_step = dict(state.get("last_improve_step", {}))


class Exp3ProgressScorer(AbstractClipScorer):
    """EXP3 over clips using learning progress as reward.

    - Keeps EMA difficulty S_t(m) from median(|adv|) (fed via update()).
    - Progress reward r_t(m) = clamp((S_{t-1}-S_t)/max(S_{t-1}, eps), 0, 1).
    - EXP3 weights w(m) updated with importance-corrected reward r̂ = r / p(m).
    """

    def __init__(
        self,
        *,
        ema_alpha: float = 0.05,
        eta: float = 0.2,
        gamma: float = 0.1,
        eps: float = 1.0e-6,
    ) -> None:
        self.ema_alpha = float(ema_alpha)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self._ema: Dict[str, float] = {}
        self._log_weights: Dict[
            str, float
        ] = {}  # default lazily to 0.0 → weight=1.0
        self._last_p_sampled: Dict[str, float] = {}
        self._last_step: Dict[str, int] = {}
        self._population_size_by_step: Dict[int, int] = {}

    def probabilities(self, keys: List[str], step: int) -> torch.Tensor:
        if len(keys) == 0:
            return torch.zeros(0, dtype=torch.float32)
        # record candidate set size for this step only if not set yet
        if step not in self._population_size_by_step:
            self._population_size_by_step[step] = len(keys)
        lw = []
        for k in keys:
            lw.append(float(self._log_weights.get(k, 0.0)))
        lw_t = torch.tensor(lw, dtype=torch.float32)
        # stable softmax over log-weights
        lw_t = lw_t - lw_t.max()
        p_core = torch.softmax(lw_t, dim=0)
        if self.gamma > 0.0:
            uni = torch.full_like(p_core, 1.0 / len(keys))
            p = (1.0 - self.gamma) * p_core + self.gamma * uni
        else:
            p = p_core
        p = torch.clamp(p, min=1.0e-12)
        p = p / p.sum()
        return p

    def on_sampled(
        self, keys: List[str], step: int, probs: Optional[torch.Tensor] = None
    ) -> None:
        if probs is None:
            # Cannot importance-correct; still record step.
            for k in keys:
                self._last_step[k] = int(step)
            return
        probs = probs.detach().cpu().float()
        for i, k in enumerate(keys):
            self._last_p_sampled[k] = float(
                max(1.0e-12, float(probs[i].item()))
            )
            self._last_step[k] = int(step)

    def update(self, stats: Dict[str, float], step: int) -> None:
        # Update EMA difficulty; compute progress and apply EXP3 updates using last-sampled p.
        for k, v in stats.items():
            v_f = float(max(0.0, v))
            prev = float(self._ema.get(k, 1.0))
            ema = (1.0 - self.ema_alpha) * prev + self.ema_alpha * v_f
            self._ema[k] = ema

            # progress in [0,1]
            denom = max(prev, self.eps)
            progress = max(0.0, min(1.0, (prev - ema) / denom))

            if k in self._last_p_sampled:
                p = float(self._last_p_sampled.pop(k))
                # scale update by candidate set size at sampling step (if available)
                k_step = int(self._last_step.get(k, step))
                pop_size = int(self._population_size_by_step.get(k_step, 1))
                scale = self.eta / max(1, pop_size)
                delta = scale * (progress / max(p, 1.0e-12))
                lw_old = float(self._log_weights.get(k, 0.0))
                lw_new = lw_old + float(delta)
                # clamp to keep numbers well-behaved; probabilities computed via softmax are shift-invariant
                if lw_new > 50.0:
                    lw_new = 50.0
                elif lw_new < -50.0:
                    lw_new = -50.0
                self._log_weights[k] = lw_new
            # else: not sampled this round; only EMA is updated

    def scores(self, keys: List[str], step: int) -> torch.Tensor:
        # For compatibility: return EMA difficulty (non-negative).
        out = [float(max(0.0, self._ema.get(k, 1.0))) for k in keys]
        return torch.tensor(out, dtype=torch.float32)

    def state_dict(self) -> dict:
        return {
            "ema_alpha": self.ema_alpha,
            "eta": self.eta,
            "gamma": self.gamma,
            "eps": self.eps,
            "ema": self._ema,
            "log_weights": self._log_weights,
            "last_step": self._last_step,
            "population_size_by_step": self._population_size_by_step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.ema_alpha = float(state.get("ema_alpha", self.ema_alpha))
        self.eta = float(state.get("eta", self.eta))
        self.gamma = float(state.get("gamma", self.gamma))
        self.eps = float(state.get("eps", self.eps))
        self._ema = dict(state.get("ema", {}))
        # Backward compatibility: support both 'log_weights' and legacy 'weights'
        if "log_weights" in state:
            self._log_weights = dict(state.get("log_weights", {}))
        else:
            # convert legacy positive weights to log-space
            w = dict(state.get("weights", {}))
            self._log_weights = {
                k: float(
                    torch.log(torch.tensor(max(1.0e-12, float(v)))).item()
                )
                for k, v in w.items()
            }
        self._last_step = dict(state.get("last_step", {}))
        self._population_size_by_step = dict(
            state.get("population_size_by_step", {})
        )


class Exp3CombinedProgressScorer(AbstractClipScorer):
    """EXP3 with combined actor+critic progress.

    - S_A(m): EMA of median(|adv|) per clip
    - S_D(m): EMA of RMS-TD per clip
    - p_A = rel_drop(S_A), p_D = rel_drop(S_D)
    - reward r = sqrt(p_A * p_D) if include_critic_progress else p_A
    - EXP3 update on log-weights with IPS and |K|-scaled step size
    """

    def __init__(
        self,
        *,
        ema_alpha: float = 0.2,
        eta: float = 0.5,
        gamma: float = 0.1,
        eps: float = 1.0e-6,
        include_critic_progress: bool = True,
    ) -> None:
        self.ema_alpha = float(ema_alpha)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.include_critic_progress = bool(include_critic_progress)
        self._ema_adv: Dict[str, float] = {}
        self._ema_td: Dict[str, float] = {}
        self._log_weights: Dict[str, float] = {}
        self._last_p_sampled: Dict[str, float] = {}
        self._last_step: Dict[str, int] = {}
        self._population_size_by_step: Dict[int, int] = {}
        # last-step diagnostics
        self._last_prog_a: Dict[str, float] = {}
        self._last_prog_d: Dict[str, float] = {}
        self._last_reward: Dict[str, float] = {}

    def probabilities(self, keys: List[str], step: int) -> torch.Tensor:
        if len(keys) == 0:
            return torch.zeros(0, dtype=torch.float32)
        if step not in self._population_size_by_step:
            self._population_size_by_step[step] = len(keys)
        lw = torch.tensor(
            [float(self._log_weights.get(k, 0.0)) for k in keys],
            dtype=torch.float32,
        )
        lw = lw - lw.max()
        p_core = torch.softmax(lw, dim=0)
        if self.gamma > 0.0:
            uni = torch.full_like(p_core, 1.0 / len(keys))
            p = (1.0 - self.gamma) * p_core + self.gamma * uni
        else:
            p = p_core
        p = torch.clamp(p, min=1.0e-12)
        return p / p.sum()

    def on_sampled(
        self, keys: List[str], step: int, probs: Optional[torch.Tensor] = None
    ) -> None:
        if probs is None:
            for k in keys:
                self._last_step[k] = int(step)
            return
        probs = probs.detach().cpu().float()
        for i, k in enumerate(keys):
            self._last_p_sampled[k] = float(
                max(1.0e-12, float(probs[i].item()))
            )
            self._last_step[k] = int(step)

    def update(self, stats: Dict[str, float], step: int) -> None:
        # Backward-compatible: use actor progress only
        self.update_combined(stats, {}, step)

    def update_combined(
        self,
        adv_stats: Dict[str, float],
        td_stats: Dict[str, float],
        step: int,
    ) -> None:
        keys = set(adv_stats.keys()) | set(td_stats.keys())
        for k in keys:
            # Update EMAs
            if k in adv_stats:
                v_a = float(max(0.0, adv_stats[k]))
                prev_a = float(self._ema_adv.get(k, 1.0))
                ema_a = (1.0 - self.ema_alpha) * prev_a + self.ema_alpha * v_a
                self._ema_adv[k] = ema_a
                denom_a = max(prev_a, self.eps)
                p_a = max(0.0, min(1.0, (prev_a - ema_a) / denom_a))
                self._last_prog_a[k] = p_a
            else:
                p_a = float(self._last_prog_a.get(k, 0.0))

            if k in td_stats:
                v_d = float(max(0.0, td_stats[k]))
                prev_d = float(self._ema_td.get(k, 1.0))
                ema_d = (1.0 - self.ema_alpha) * prev_d + self.ema_alpha * v_d
                self._ema_td[k] = ema_d
                denom_d = max(prev_d, self.eps)
                p_d = max(0.0, min(1.0, (prev_d - ema_d) / denom_d))
                self._last_prog_d[k] = p_d
            else:
                p_d = float(self._last_prog_d.get(k, 0.0))

            # Combined reward
            if self.include_critic_progress:
                r = float(torch.sqrt(torch.tensor(p_a * p_d)).item())
            else:
                r = p_a
            self._last_reward[k] = r

            # EXP3 log-weight update if sampled
            if k in self._last_p_sampled:
                p = float(self._last_p_sampled.pop(k))
                k_step = int(self._last_step.get(k, step))
                pop_size = int(self._population_size_by_step.get(k_step, 1))
                scale = self.eta / max(1, pop_size)
                delta = scale * (r / max(p, 1.0e-12))
                lw_old = float(self._log_weights.get(k, 0.0))
                lw_new = lw_old + float(delta)
                if lw_new > 50.0:
                    lw_new = 50.0
                elif lw_new < -50.0:
                    lw_new = -50.0
                self._log_weights[k] = lw_new

    def scores(self, keys: List[str], step: int) -> torch.Tensor:
        out = [float(max(0.0, self._ema_adv.get(k, 1.0))) for k in keys]
        return torch.tensor(out, dtype=torch.float32)

    def state_dict(self) -> dict:
        return {
            "ema_alpha": self.ema_alpha,
            "eta": self.eta,
            "gamma": self.gamma,
            "eps": self.eps,
            "include_critic_progress": self.include_critic_progress,
            "ema_adv": self._ema_adv,
            "ema_td": self._ema_td,
            "log_weights": self._log_weights,
            "last_step": self._last_step,
            "population_size_by_step": self._population_size_by_step,
            "last_prog_a": self._last_prog_a,
            "last_prog_d": self._last_prog_d,
            "last_reward": self._last_reward,
        }

    def load_state_dict(self, state: dict) -> None:
        self.ema_alpha = float(state.get("ema_alpha", self.ema_alpha))
        self.eta = float(state.get("eta", self.eta))
        self.gamma = float(state.get("gamma", self.gamma))
        self.eps = float(state.get("eps", self.eps))
        self.include_critic_progress = bool(
            state.get("include_critic_progress", self.include_critic_progress)
        )
        self._ema_adv = dict(state.get("ema_adv", {}))
        self._ema_td = dict(state.get("ema_td", {}))
        self._log_weights = dict(state.get("log_weights", {}))
        self._last_step = dict(state.get("last_step", {}))
        self._population_size_by_step = dict(
            state.get("population_size_by_step", {})
        )
        self._last_prog_a = dict(state.get("last_prog_a", {}))
        self._last_prog_d = dict(state.get("last_prog_d", {}))
        self._last_reward = dict(state.get("last_reward", {}))


class Exp3PoseProgressScorer(AbstractClipScorer):
    """EXP3 with pose-error (MPJPE/MPKPE) progress as reward.

    - S_P(m): EMA of combined pose error per clip (provided by caller)
    - p_P = rel_drop(S_P) clipped to [0, progress_clip]
    - reward r = p_P
    - EXP3 update on log-weights with IPS and |K|-scaled step size
    """

    def __init__(
        self,
        *,
        ema_alpha: float = 0.2,
        eta: float = 0.5,
        gamma: float = 0.1,
        eps: float = 1.0e-6,
        progress_clip: float = 0.25,
    ) -> None:
        self.ema_alpha = float(ema_alpha)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.progress_clip = float(max(0.0, progress_clip))
        self._ema_pose: Dict[str, float] = {}
        self._log_weights: Dict[str, float] = {}
        self._last_p_sampled: Dict[str, float] = {}
        self._last_step: Dict[str, int] = {}
        self._population_size_by_step: Dict[int, int] = {}
        self._last_prog_p: Dict[str, float] = {}
        self._last_reward: Dict[str, float] = {}

    def probabilities(self, keys: List[str], step: int) -> torch.Tensor:
        if len(keys) == 0:
            return torch.zeros(0, dtype=torch.float32)
        if step not in self._population_size_by_step:
            self._population_size_by_step[step] = len(keys)
        lw = torch.tensor(
            [float(self._log_weights.get(k, 0.0)) for k in keys],
            dtype=torch.float32,
        )
        lw = lw - lw.max()
        p_core = torch.softmax(lw, dim=0)
        if self.gamma > 0.0:
            uni = torch.full_like(p_core, 1.0 / len(keys))
            p = (1.0 - self.gamma) * p_core + self.gamma * uni
        else:
            p = p_core
        p = torch.clamp(p, min=1.0e-12)
        return p / p.sum()

    def on_sampled(
        self, keys: List[str], step: int, probs: Optional[torch.Tensor] = None
    ) -> None:
        if probs is None:
            for k in keys:
                self._last_step[k] = int(step)
            return
        probs = probs.detach().cpu().float()
        for i, k in enumerate(keys):
            self._last_p_sampled[k] = float(
                max(1.0e-12, float(probs[i].item()))
            )
            self._last_step[k] = int(step)

    def update(self, stats: Dict[str, float], step: int) -> None:
        # stats: per-key combined pose error (non-negative)
        for k, v in stats.items():
            v_f = float(max(0.0, v))
            prev = float(self._ema_pose.get(k, 1.0))
            ema = (1.0 - self.ema_alpha) * prev + self.ema_alpha * v_f
            self._ema_pose[k] = ema
            denom = max(prev, self.eps)
            p = (prev - ema) / denom
            # clip symmetric to be robust to noise (allow minor negative)
            p = float(max(-self.progress_clip, min(self.progress_clip, p)))
            # reward is non-negative progress only
            r = float(max(0.0, p))
            self._last_prog_p[k] = (
                r  # store non-negative progress actually rewarded
            )
            self._last_reward[k] = r
            if k in self._last_p_sampled:
                prob = float(self._last_p_sampled.pop(k))
                k_step = int(self._last_step.get(k, step))
                pop_size = int(self._population_size_by_step.get(k_step, 1))
                scale = self.eta / max(1, pop_size)
                delta = scale * (r / max(prob, 1.0e-12))
                lw_old = float(self._log_weights.get(k, 0.0))
                lw_new = lw_old + float(delta)
                if lw_new > 50.0:
                    lw_new = 50.0
                elif lw_new < -50.0:
                    lw_new = -50.0
                self._log_weights[k] = lw_new

    def scores(self, keys: List[str], step: int) -> torch.Tensor:
        out = [float(max(0.0, self._ema_pose.get(k, 1.0))) for k in keys]
        return torch.tensor(out, dtype=torch.float32)

    def state_dict(self) -> dict:
        return {
            "ema_alpha": self.ema_alpha,
            "eta": self.eta,
            "gamma": self.gamma,
            "eps": self.eps,
            "progress_clip": self.progress_clip,
            "ema_pose": self._ema_pose,
            "log_weights": self._log_weights,
            "last_step": self._last_step,
            "population_size_by_step": self._population_size_by_step,
            "last_prog_p": self._last_prog_p,
            "last_reward": self._last_reward,
        }

    def load_state_dict(self, state: dict) -> None:
        self.ema_alpha = float(state.get("ema_alpha", self.ema_alpha))
        self.eta = float(state.get("eta", self.eta))
        self.gamma = float(state.get("gamma", self.gamma))
        self.eps = float(state.get("eps", self.eps))
        self.progress_clip = float(
            state.get("progress_clip", self.progress_clip)
        )
        self._ema_pose = dict(state.get("ema_pose", {}))
        self._log_weights = dict(state.get("log_weights", {}))
        self._last_step = dict(state.get("last_step", {}))
        self._population_size_by_step = dict(
            state.get("population_size_by_step", {})
        )
        self._last_prog_p = dict(state.get("last_prog_p", {}))
        self._last_reward = dict(state.get("last_reward", {}))




import torch
import numpy as np
from torch.utils.data import Sampler
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re

class CurriculumHierarchicalSampler(Sampler):
    def __init__(
        self,
        dataset,
        gamma_decay: float = 0.9993,
        laplace_eps: float = 1.0,
        laplace_kappa: float = 2.0,
        temp_start: float = 3.0,
        temp_end: float = 0.15,
        gamma_start: float = 0.15,
        gamma_end: float = 0.05,
        decay_start: int = 5000,
        decay_end: int = 35000,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.gamma_decay = float(gamma_decay)
        self.laplace_eps = float(laplace_eps)
        self.laplace_kappa = float(laplace_kappa)
        self.temp_start = float(temp_start)
        self.temp_end = float(max(0.01, temp_end))
        self.gamma_start = float(gamma_start)
        self.gamma_end = float(gamma_end)
        self.decay_start = int(decay_start)
        self.decay_end = int(decay_end)
        self.rng = np.random.default_rng(seed)
        self.current_iteration: int = 0
        
        self._clip_A: Dict[str, float] = {}
        self._clip_F: Dict[str, float] = {}
        self._motion_A: Dict[str, float] = {}
        self._motion_F: Dict[str, float] = {}
        
        self.motion_to_bins = defaultdict(list)
        self.bin_to_idx = {}
        
        # ======== [新增：建立严谨的物理时间轴字典] ========
        self.m_key_to_bins_info = defaultdict(list)
        
        for idx, window in enumerate(self.dataset.windows):
            m_key = window.raw_motion_key
            b_key = window.motion_key
            self.motion_to_bins[m_key].append(b_key)
            self.bin_to_idx[b_key] = idx
            
            # 使用正则精准提取 Bin 的起始帧，例如 "___start_300_len_550" 提取出 300
            match = re.search(r"__start_(\d+)_", b_key)
            start_frame = int(match.group(1)) if match else 0
            self.m_key_to_bins_info[m_key].append((start_frame, b_key))
            
            if m_key not in self._motion_A:
                self._motion_A[m_key] = 0.0
                self._motion_F[m_key] = 0.0
            self._clip_A[b_key] = 0.0
            self._clip_F[b_key] = 0.0
            
        # 必须对时间轴进行严格的从小到大排序，这是“沿途结算”的数学基础
        for m_key in self.m_key_to_bins_info:
            self.m_key_to_bins_info[m_key].sort(key=lambda x: x[0])
            
        self.motion_keys = list(self._motion_A.keys())
        self._initial_m_probs: Optional[np.ndarray] = None
        self._initial_b_probs: Dict[str, np.ndarray] = {}
        self._prob_min_ratio = 0.75
        self._prob_max_ratio = 80.0
        self._recompute_probs()

    @staticmethod
    def _clamp_probs_with_ratio_bounds(
        probs: np.ndarray,
        base_probs: np.ndarray,
        min_ratio: float,
        max_ratio: float,
    ) -> np.ndarray:
        """Project probabilities with per-item ratio bounds while keeping sum=1."""
        p = np.asarray(probs, dtype=np.float64)
        b = np.asarray(base_probs, dtype=np.float64)
        if p.ndim != 1 or b.ndim != 1 or len(p) != len(b) or len(p) == 0:
            return probs

        lower = b * float(min_ratio)
        upper = b * float(max_ratio)
        # Safety: keep feasible interval strictly positive.
        lower = np.maximum(lower, 1.0e-12)
        upper = np.maximum(upper, lower)

        # Iterative simplex projection with bounds.
        q = np.clip(p, lower, upper)
        active = np.ones_like(q, dtype=bool)
        target_sum = 1.0
        for _ in range(16):
            fixed_sum = q[~active].sum()
            remain = target_sum - fixed_sum
            active_count = int(active.sum())
            if active_count <= 0:
                break
            candidate = q[active] + (remain - q[active].sum()) / active_count
            clamped = np.clip(candidate, lower[active], upper[active])
            q[active] = clamped
            new_active = active.copy()
            new_active[active] = (candidate >= lower[active]) & (candidate <= upper[active])
            if np.array_equal(new_active, active):
                break
            active = new_active

        q = np.clip(q, lower, upper)
        q_sum = q.sum()
        if q_sum <= 0.0:
            return probs
        q = q / q_sum
        return q.astype(np.float64)

    def _get_decayed_value(self, start_val: float, end_val: float, step: int) -> float:
        if step <= self.decay_start: return start_val
        if step >= self.decay_end: return end_val
        progress = (step - self.decay_start) / (self.decay_end - self.decay_start)
        return start_val + progress * (end_val - start_val)

    def _calc_probabilities(self, attempts: Dict[str, float], failures: Dict[str, float], keys_list: List[str]) -> np.ndarray:
        current_temp = self._get_decayed_value(self.temp_start, self.temp_end, self.current_iteration)
        current_gamma = self._get_decayed_value(self.gamma_start, self.gamma_end, self.current_iteration)
        
        # 贝叶斯平滑得分计算: (F + eps) / (A + kappa)
        scores = torch.tensor([
            (failures[k] + self.laplace_eps) / (attempts[k] + self.laplace_kappa) 
            for k in keys_list
        ], dtype=torch.float32)
        
        scaled_scores = scores / current_temp
        scaled_scores = scaled_scores - scaled_scores.max()
        p_core = torch.softmax(scaled_scores, dim=0)
        
        if current_gamma > 0.0:
            uni = torch.full_like(p_core, 1.0 / len(keys_list))
            p = (1.0 - current_gamma) * p_core + current_gamma * uni
        else:
            p = p_core
            
        p = torch.clamp(p, min=1.0e-12)
        p = p / p.sum()
        return p.numpy()

    def _recompute_probs(self) -> None:
        m_probs = self._calc_probabilities(self._motion_A, self._motion_F, self.motion_keys)
        if self._initial_m_probs is None:
            self._initial_m_probs = np.asarray(m_probs, dtype=np.float64)
        self.m_probs = self._clamp_probs_with_ratio_bounds(
            np.asarray(m_probs, dtype=np.float64),
            self._initial_m_probs,
            self._prob_min_ratio,
            self._prob_max_ratio,
        )
        self.b_probs_dict = {}
        for m_key, b_keys in self.motion_to_bins.items():
            b_probs = self._calc_probabilities(self._clip_A, self._clip_F, b_keys)
            if m_key not in self._initial_b_probs:
                self._initial_b_probs[m_key] = np.asarray(b_probs, dtype=np.float64)
            self.b_probs_dict[m_key] = self._clamp_probs_with_ratio_bounds(
                np.asarray(b_probs, dtype=np.float64),
                self._initial_b_probs[m_key],
                self._prob_min_ratio,
                self._prob_max_ratio,
            )

    def set_iteration(self, global_rl_iter: int) -> None:
        if self.current_iteration != global_rl_iter:
            self.current_iteration = global_rl_iter
            # Iteration 更新后，立刻根据新的温度重新计算一次概率，保证退火实时生效
            self._recompute_probs()

    def update_scores(self, events: List[tuple], global_rl_iter: int = None) -> None:
        """
        核心结算函数：只有在 Clip 结束时 (Reset) 才会被调用触发。
        """
        # 1. 更新全局 Iteration
        if global_rl_iter is not None:
            self.current_iteration = global_rl_iter
        else:
            self.current_iteration += 1

        if not events:
            return
            
        # 2. 全局遗忘机制 (时间衰减)
        for k in self._motion_A:
            self._motion_A[k] *= self.gamma_decay
            self._motion_F[k] *= self.gamma_decay
        for k in self._clip_A:
            self._clip_A[k] *= self.gamma_decay
            self._clip_F[k] *= self.gamma_decay
            
        # 3. 滑动窗口马后炮结算
        for event in events:
            # 兼容解析 4 元组 (严谨模式) 或 3 元组 (旧版回退)
            if len(event) == 4:
                m_key, start_b_key, is_failure, progress_frames = event
            else:
                m_key, start_b_key, is_failure = event
                progress_frames = 0
                
            # [宏观] Motion 级结算：只要玩了一局，A就+1；如果失败了，F就加上失败权重。
            if m_key in self._motion_A:
                self._motion_A[m_key] += 1.0
                self._motion_F[m_key] += is_failure

            # [微观] Bin 级滑动窗口结算
            if progress_frames > 0 and m_key in self.m_key_to_bins_info:
                # 解析出这局游戏的出发点
                match = re.search(r"__start_(\d+)_", start_b_key)
                base_start = int(match.group(1)) if match else 0
                
                # 计算出这局游戏的阵亡点 (或通关点)
                end_frame = base_start + progress_frames
                
                # 在时间轴上框出所有“活过”的关卡 (Bins)
                traversed_b_keys = []
                for bin_start, b_key in self.m_key_to_bins_info[m_key]:
                    # 只要 Bin 的起点被这局游戏的存活区间覆盖，就算路过
                    if base_start <= bin_start <= end_frame:
                        traversed_b_keys.append(b_key)
                
                if traversed_b_keys:
                    total_traversed = len(traversed_b_keys)
                    for i, b_key in enumerate(traversed_b_keys):
                        if b_key in self._clip_A:
                            # 凡是路过的 Bin，尝试次数 A 统统 +1
                            self._clip_A[b_key] += 1.0  
                            
                            # 【核心防抖】只有“最后踩到的那个 Bin”，并且这局游戏是“失败”结尾，才背锅 (F+1)
                            is_last_bin = (i == total_traversed - 1)
                            if is_failure > 0.0 and is_last_bin:
                                self._clip_F[b_key] += is_failure
                else:
                    # 极端容错：没匹配到任何遍历序列，仅结算起点
                    if start_b_key in self._clip_A:
                        self._clip_A[start_b_key] += 1.0
                        self._clip_F[start_b_key] += is_failure
            else:
                # 旧版本兼容：没有提供 progress_frames，直接给起点背锅
                if start_b_key in self._clip_A:
                    self._clip_A[start_b_key] += 1.0
                    self._clip_F[start_b_key] += is_failure
                
        # 结算完毕后，立刻根据最新分数重构概率树
        self._recompute_probs()

    def __iter__(self):
        while True:
            sampled_m_key = self.rng.choice(self.motion_keys, p=self.m_probs)
            b_keys = self.motion_to_bins[sampled_m_key]
            sampled_b_key = self.rng.choice(b_keys, p=self.b_probs_dict[sampled_m_key])
            yield self.bin_to_idx[sampled_b_key]

    def __len__(self):
        return len(self.dataset.windows)
        
    def state_dict(self) -> dict:
        # ====== [1] Motion 宏观层级诊断信息 ======
        motion_scores = {
            k: (self._motion_F[k] + self.laplace_eps) / (self._motion_A[k] + self.laplace_kappa)
            for k in self.motion_keys
        }
        motion_probs = {}
        if hasattr(self, 'm_probs'):
            for k, p in zip(self.motion_keys, self.m_probs):
                motion_probs[k] = float(p)
                
        # ====== [2] Bin (Clip) 微观层级诊断信息 ======
        clip_scores = {
            k: (self._clip_F[k] + self.laplace_eps) / (self._clip_A[k] + self.laplace_kappa)
            for k in self._clip_A.keys()
        }
        clip_probs = {}
        if hasattr(self, 'b_probs_dict'):
            for m_key, b_keys in self.motion_to_bins.items():
                if m_key in self.b_probs_dict:
                    for k, p in zip(b_keys, self.b_probs_dict[m_key]):
                        # 这里的概率是 P(Bin | Motion)，即在当前动作下的局部抽取概率
                        clip_probs[k] = float(p)

        # ====== [3] 提取 Hard Cases 榜单方便快速查看 ======
        top_hard_motions = sorted(motion_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        top_hard_clips = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            # 核心计数器 (断点续训必须)
            "clip_A": self._clip_A,
            "clip_F": self._clip_F,
            "motion_A": self._motion_A,
            "motion_F": self._motion_F,
            "current_iteration": self.current_iteration,
            
            # Motion 级上帝视角
            "diagnostics_motion_scores": motion_scores,
            "diagnostics_motion_probs": motion_probs,
            "diagnostics_top_20_hard_motions": dict(top_hard_motions),
            
            # Bin (Clip) 级上帝视角
            "diagnostics_clip_scores": clip_scores,
            "diagnostics_clip_probs": clip_probs,
            "diagnostics_top_20_hard_clips": dict(top_hard_clips),
        }
    def load_state_dict(self, state: dict) -> None:
        self.current_iteration = int(state.get("current_iteration", 0))
        
        # 1. 加载新版 (贝叶斯平滑) 的断点文件
        if "clip_A" in state:
            self._clip_A = dict(state.get("clip_A", {}))
            self._clip_F = dict(state.get("clip_F", {}))
            self._motion_A = dict(state.get("motion_A", {}))
            self._motion_F = dict(state.get("motion_F", {}))
            
        # 2. 兼容老版本 (EMA) 的断点文件，并自动转换为等效的贝叶斯计数
        elif "clip_ema" in state or "motion_ema" in state:
            from loguru import logger
            logger.warning("检测到旧版 EMA Curriculum 权重，正在自动映射为贝叶斯平滑格式...")
            
            old_clip_ema = state.get("clip_ema", {})
            old_motion_ema = state.get("motion_ema", {})
            
            # 赋予一个“虚拟的初始尝试次数”（比如 100 次）
            # 这样既能继承之前的失败率，又能给算法一定的置信度惯性
            base_A = 100.0
            
            for k, ema_score in old_clip_ema.items():
                self._clip_A[k] = base_A
                # EMA score 本质上是失败率，失败次数 = 失败率 * 总尝试次数
                self._clip_F[k] = float(ema_score) * base_A
                
            for k, ema_score in old_motion_ema.items():
                self._motion_A[k] = base_A
                self._motion_F[k] = float(ema_score) * base_A
                
        else:
            # 既没有新 key 也没有旧 key，维持初始化时的 0 状态
            pass

        # 诊断信息 (diagnostics_xxx) 只是给人看的日志，不需要读取，
        # 因为下面这行 _recompute_probs() 会根据 A 和 F 重新严格计算一遍当前的概率树
        self._recompute_probs()

class HierarchicalFailureScorer(AbstractClipScorer):
    """Two-stage Adaptive Sampling Scorer with Linear Annealing."""
    def __init__(
        self,
        *,
        clip_alpha: float = 0.05,
        motion_alpha: float = 0.02,
        motion_weight: float = 0.5,
        temp_start: float = 3.0,
        temp_end: float = 0.15,
        gamma_start: float = 0.15,
        gamma_end: float = 0.05,
        decay_start: int = 5000,
        decay_end: int = 30000,
        eps: float = 1.0e-6,
    ) -> None:
        self.clip_alpha = float(clip_alpha)
        self.motion_alpha = float(motion_alpha)
        self.motion_weight = float(motion_weight)
        
        # 衰减参数配置
        self.temp_start = float(temp_start)
        self.temp_end = float(max(0.01, temp_end))
        self.gamma_start = float(gamma_start)
        self.gamma_end = float(gamma_end)
        self.decay_start = int(decay_start)
        self.decay_end = int(decay_end)
        
        self.eps = float(eps)

        self._clip_ema: Dict[str, float] = {}
        self._motion_ema: Dict[str, float] = {}
        self._last_step: Dict[str, int] = {}

        # [新增] 显式追踪真实的 RL Iteration
        self.current_iteration: int = 0
        
    def _get_motion_key(self, clip_key: str) -> str:
        if "__start_" in clip_key:
            return clip_key.split("__start_", 1)[0]
        return clip_key

    def _get_decayed_value(self, start_val: float, end_val: float, step: int) -> float:
        """根据当前 step 计算线性衰减的参数值"""
        if step <= self.decay_start:
            return start_val
        if step >= self.decay_end:
            return end_val
        
        progress = (step - self.decay_start) / (self.decay_end - self.decay_start)
        return start_val + progress * (end_val - start_val)

    def update(self, stats: Dict[str, float], step: int) -> None:
        self.current_iteration += 1
        motion_buffer: Dict[str, List[float]] = {}
        for c_key, v in stats.items():
            v_f = float(max(0.0, v))
            c_prev = float(self._clip_ema.get(c_key, 1.0))
            self._clip_ema[c_key] = (1.0 - self.clip_alpha) * c_prev + self.clip_alpha * v_f
            
            m_key = self._get_motion_key(c_key)
            if m_key not in motion_buffer:
                motion_buffer[m_key] = []
            motion_buffer[m_key].append(v_f)
            
        for m_key, m_vals in motion_buffer.items():
            m_mean = sum(m_vals) / max(1, len(m_vals))
            m_prev = float(self._motion_ema.get(m_key, 1.0))
            self._motion_ema[m_key] = (1.0 - self.motion_alpha) * m_prev + self.motion_alpha * m_mean

    def scores(self, keys: List[str], step: int) -> torch.Tensor:
        out = []
        for c_key in keys:
            m_key = self._get_motion_key(c_key)
            c_score = float(self._clip_ema.get(c_key, 1.0))
            m_score = float(self._motion_ema.get(m_key, 1.0))
            combined = (m_score ** self.motion_weight) * (c_score ** (1.0 - self.motion_weight))
            out.append(combined)
        return torch.tensor(out, dtype=torch.float32)

    def probabilities(self, keys: List[str], step: int) -> torch.Tensor:
        if len(keys) == 0:
            return torch.zeros(0, dtype=torch.float32)
        
        raw_scores = self.scores(keys, step)
        
        # [修改] 使用真实的 RL iteration 计算衰减，而不是底层的环境采样 step
        current_temp = self._get_decayed_value(self.temp_start, self.temp_end, self.current_iteration)
        current_gamma = self._get_decayed_value(self.gamma_start, self.gamma_end, self.current_iteration)
        
        scaled_scores = raw_scores / current_temp
        scaled_scores = scaled_scores - scaled_scores.max()
        p_core = torch.softmax(scaled_scores, dim=0)
        
        if current_gamma > 0.0:
            uni = torch.full_like(p_core, 1.0 / len(keys))
            p = (1.0 - current_gamma) * p_core + current_gamma * uni
        else:
            p = p_core
            
        p = torch.clamp(p, min=1.0e-12)
        return p / p.sum()

    def on_sampled(self, keys: List[str], step: int, probs: Optional[torch.Tensor] = None) -> None:
        for k in keys:
            self._last_step[k] = int(step)

    def state_dict(self) -> dict:
        return {
            "clip_ema": self._clip_ema,
            "motion_ema": self._motion_ema,
            "ema": self._clip_ema,
            "last_step": self._last_step,
            "current_iteration": self.current_iteration,  # [新增] 保存迭代进度
        }

    def load_state_dict(self, state: dict) -> None:
        self._clip_ema = dict(state.get("clip_ema", state.get("ema", {})))
        self._motion_ema = dict(state.get("motion_ema", {}))
        self._last_step = dict(state.get("last_step", {}))
        self.current_iteration = int(state.get("current_iteration", 0))  # [新增] 恢复迭代进度

MANDATORY_DATASETS = {
    "dof_pos": "dof_pos",
    "dof_vel": "dof_vel",
    "rg_pos": "global_translation",
    "rb_rot": "global_rotation_quat",
    "body_vel": "global_velocity",
    "body_ang_vel": "global_angular_velocity",
}


class _WorldFrameZUpNormalizer:
    """Apply a fixed world-frame normalization to prefixed motion tensors in-place."""

    def __init__(
        self,
        *,
        arrays: Dict[str, Tensor],
        offset_xy: Tensor,  # [3], z==0
        q_flat_xyzw: Tensor,  # [T*B, 4] in XYZW
        ref_rg_pos_shape: torch.Size,
        ref_rb_rot_shape: torch.Size,
    ) -> None:
        self._arrays = arrays
        self._offset_xy = offset_xy
        self._q_flat_wxyz = torch_utils.xyzw_to_wxyz(q_flat_xyzw)
        self._ref_rg_pos_shape = ref_rg_pos_shape
        self._ref_rb_rot_shape = ref_rb_rot_shape

    def apply(self, prefix: str) -> None:
        pos_key = f"{prefix}rg_pos"
        rot_key = f"{prefix}rb_rot"
        vel_key = f"{prefix}body_vel"
        ang_key = f"{prefix}body_ang_vel"
        if (
            pos_key not in self._arrays
            or rot_key not in self._arrays
            or vel_key not in self._arrays
            or ang_key not in self._arrays
        ):
            return

        pos = self._arrays[pos_key]
        rot = self._arrays[rot_key]
        vel = self._arrays[vel_key]
        ang = self._arrays[ang_key]
        if (
            pos.shape != self._ref_rg_pos_shape
            or rot.shape != self._ref_rb_rot_shape
        ):
            return

        # Center XY using canonical offset.
        pos[..., 0] -= self._offset_xy[0]
        pos[..., 1] -= self._offset_xy[1]

        # Rotate vectors using shared quaternion utilities (WXYZ convention).
        pos_flat = pos.reshape(-1, 3)
        vel_flat = vel.reshape(-1, 3)
        ang_flat = ang.reshape(-1, 3)
        pos[:] = torch_utils.quat_apply(
            self._q_flat_wxyz, pos_flat
        ).reshape_as(pos)
        vel[:] = torch_utils.quat_apply(
            self._q_flat_wxyz, vel_flat
        ).reshape_as(vel)
        ang[:] = torch_utils.quat_apply(
            self._q_flat_wxyz, ang_flat
        ).reshape_as(ang)

        # Rotate orientations: q' = q_heading_inv * q.
        rot_flat_xyzw = rot.reshape(-1, 4)
        rot_flat_wxyz = torch_utils.xyzw_to_wxyz(rot_flat_xyzw)
        rot_out_wxyz = torch_utils.quat_mul(self._q_flat_wxyz, rot_flat_wxyz)
        rot[:] = torch_utils.wxyz_to_xyzw(rot_out_wxyz).reshape_as(rot)


def _normalize_window_world_frame(arrays: Dict[str, Tensor]) -> None:
    """Normalize a motion window into a canonical z-up world frame in-place.

    Behavior:
    - Uses the canonical root (body 0) at frame 0 from `ref_*` to:
      - Subtract its XY position from all body positions (Z is unchanged).
      - Remove its yaw around +Z from all body orientations.
    - Applies the same SE(3) transform to:
      - Positions: {ref_,ft_ref_}rg_pos[...]
      - Rotations: {ref_,ft_ref_}rb_rot[...]
      - Linear velocities: {ref_,ft_ref_}body_vel[...]
      - Angular velocities: {ref_,ft_ref_}body_ang_vel[...]
    """
    if "ref_rg_pos" not in arrays or "ref_rb_rot" not in arrays:
        raise ValueError("ref_rg_pos and ref_rb_rot are required")
    if "ref_body_vel" not in arrays or "ref_body_ang_vel" not in arrays:
        raise ValueError("ref_body_vel and ref_body_ang_vel are required")

    rg_pos = arrays["ref_rg_pos"]
    rb_rot = arrays["ref_rb_rot"]

    # Root pose at frame 0, body 0 (XYZW quaternion, z-up).
    p_root0 = rg_pos[0, 0]  # [3]
    q_root0 = rb_rot[0, 0]  # [4]

    # Compute XY offset from root at frame 0 (will be applied in _apply_to_set).
    offset_xy = p_root0.clone()
    offset_xy[2] = 0.0

    # Extract yaw from q_root0 (XYZW) using z-up convention.
    x = q_root0[0]
    y = q_root0[1]
    z = q_root0[2]
    w = q_root0[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = w * w + x * x - y * y - z * z
    yaw0 = torch.atan2(siny_cosp, cosy_cosp)

    # Quaternion for rotation around +Z by -yaw0 (remove initial heading).
    half = -0.5 * yaw0
    sin_half = torch.sin(half)
    cos_half = torch.cos(half)
    q_heading_inv = torch.stack(
        [
            torch.zeros_like(sin_half),
            torch.zeros_like(sin_half),
            sin_half,
            cos_half,
        ],
        dim=-1,
    )  # [4], XYZW

    t, b, _ = rg_pos.shape
    q_flat = q_heading_inv.view(1, 1, 4).expand(t, b, 4).reshape(-1, 4)
    normalizer = _WorldFrameZUpNormalizer(
        arrays=arrays,
        offset_xy=offset_xy,
        q_flat_xyzw=q_flat,
        ref_rg_pos_shape=rg_pos.shape,
        ref_rb_rot_shape=rb_rot.shape,
    )

    for pfx in ("ref_", "ft_ref_"):
        normalizer.apply(pfx)


@dataclass
class MotionWindow:
    """Metadata describing a contiguous motion window within an HDF5 shard."""

    motion_key: str  # unique per window
    shard_index: int
    start: int
    length: int
    raw_motion_key: str  # original clip key
    window_index: int


@dataclass
class MotionClipSample:
    """In-memory representation of a motion window.

    Attributes:
        motion_key: Unique window identifier (includes slice info).
        raw_motion_key: Original clip identifier from manifest.
        tensors: Mapping from tensor name to data tensor of shape
            ``[window_length, ...]`` (float32 unless specified otherwise).
        length: Number of valid frames contained in the sample (``<=``
            ``max_frame_length``).
    """

    motion_key: str
    raw_motion_key: str
    tensors: Dict[str, Tensor]
    length: int


@dataclass
class ClipBatch:
    """Batch of motion clips ready for consumption by the environment.

    Attributes:
        tensors: Mapping from tensor name to tensor with shape
            ``[batch_size, max_frame_length, ...]`` placed on the staging
            device.
        lengths: Valid frame counts per clip ``[batch_size]``.
        motion_keys: List of motion keys corresponding to each clip.
        max_frame_length: Fixed length configured for the cache.
    """

    tensors: Dict[str, Tensor]
    lengths: Tensor
    motion_keys: List[str]
    raw_motion_keys: List[str]
    max_frame_length: int

    # ======== [新增: 赋予 ClipBatch 搬运到 GPU 的能力] ========
    def to(self, device: torch.device, non_blocking: bool = False) -> "ClipBatch":
        """Move all tensors in the batch to the specified device."""
        return ClipBatch(
            tensors={
                k: v.to(device, non_blocking=non_blocking) 
                for k, v in self.tensors.items()
            },
            lengths=self.lengths.to(device, non_blocking=non_blocking),
            motion_keys=self.motion_keys,
            raw_motion_keys=self.raw_motion_keys,
            max_frame_length=self.max_frame_length,
        )
    # ==========================================================

    @staticmethod
    def collate_fn(samples: List[MotionClipSample]) -> "ClipBatch":
        if len(samples) == 0:
            raise ValueError(
                "ClipBatch collate_fn received an empty sample list"
            )

        max_frame_length = max(
            sample.tensors["ref_dof_pos"].shape[0] for sample in samples
        )
        max_frame_length = int(max_frame_length)

        batched_tensors: Dict[str, Tensor] = {}
        lengths = torch.zeros(len(samples), dtype=torch.long)
        motion_keys = []
        raw_motion_keys = []

        for batch_idx, sample in enumerate(samples):
            lengths[batch_idx] = sample.length
            motion_keys.append(sample.motion_key)
            raw_motion_keys.append(sample.raw_motion_key)

            for name, tensor in sample.tensors.items():
                if name not in batched_tensors:
                    pad_shape = (
                        len(samples),
                        max_frame_length,
                    ) + tensor.shape[1:]
                    batched_tensors[name] = torch.zeros(
                        pad_shape,
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )

                target = batched_tensors[name]
                valid_frames = sample.length
                target[batch_idx, :valid_frames] = tensor

                if valid_frames < max_frame_length and valid_frames > 0:
                    target[batch_idx, valid_frames:] = tensor[valid_frames - 1]

        return ClipBatch(
            tensors=batched_tensors,
            lengths=lengths,
            motion_keys=motion_keys,
            raw_motion_keys=raw_motion_keys,
            max_frame_length=max_frame_length,
        )


def _cache_collate_fn(
    samples: List[MotionClipSample],
    mode: str,
    batch_size: int,
) -> ClipBatch:
    """Collate function for motion cache DataLoader (supports validation padding)."""
    if mode == "val" and batch_size > len(samples) and len(samples) > 0:
        extra = batch_size - len(samples)
        gen = torch.Generator()
        idx = torch.randint(0, len(samples), size=(extra,), generator=gen)
        padded = list(samples)
        for i in idx.tolist():
            padded.append(samples[i])
        return ClipBatch.collate_fn(padded)
    return ClipBatch.collate_fn(samples)


class InfiniteDistributedSampler(DistributedSampler):
    """Distributed sampler that yields an infinite stream by cycling epochs."""

    def __iter__(self):
        # Infinite stream by cycling epochs
        while True:
            self.set_epoch(getattr(self, "_epoch", 0))
            for idx in super().__iter__():
                yield idx
            self._epoch = getattr(self, "_epoch", 0) + 1


class InfiniteRandomSampler(Sampler[int]):
    """Random sampler that yields infinite reshuffled passes over the dataset."""

    def __init__(self, data_source: Dataset, seed: int = 0) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        # Yield infinite permutations of indices
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(self.data_source), generator=g)
            for idx in perm.tolist():
                yield int(idx)
            self.epoch += 1

    def __len__(self) -> int:
        # Large sentinel to satisfy components that query length
        return 2**31 - 1


class WeightedBinInfiniteSampler(Sampler[int]):
    """Infinite sampler that respects regex-based weighted bins over indices."""

    def __init__(
        self,
        dataset_len: int,
        bin_indices: List[List[int]],
        ratios: List[float],
        batch_size: int,
        seed: int,
    ) -> None:
        self._ds_len = int(max(0, dataset_len))
        self._bins = [torch.tensor(b, dtype=torch.long) for b in bin_indices]
        self._ratios = list(ratios)
        self._batch_size = int(max(1, batch_size))
        self._seed = int(seed)
        self._epoch = 0

        raw_counts = [r * float(self._batch_size) for r in self._ratios]
        base_counts = [int(c) for c in raw_counts]
        residuals = [c - int(c) for c in raw_counts]
        remaining = self._batch_size - int(sum(base_counts))
        if remaining != 0:
            order = sorted(
                range(len(residuals)),
                key=lambda i: residuals[i],
                reverse=True,
            )
            idx_pos = 0
            while remaining > 0:
                j = order[idx_pos % len(order)]
                base_counts[j] += 1
                remaining -= 1
                idx_pos += 1
        self._counts = [max(0, int(c)) for c in base_counts]

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            batch: List[int] = []
            for bin_idx, count in zip(self._bins, self._counts):
                if count <= 0 or bin_idx.numel() == 0:
                    continue
                choice = torch.randint(
                    0,
                    int(bin_idx.numel()),
                    size=(count,),
                    generator=g,
                )
                selected = bin_idx[choice].tolist()
                batch.extend(int(x) for x in selected)

            if not batch:
                # Fallback: uniform over dataset indices
                if self._ds_len == 0:
                    raise ValueError(
                        "WeightedBinInfiniteSampler cannot sample from an empty dataset"
                    )
                all_idx = torch.randint(
                    0,
                    self._ds_len,
                    size=(self._batch_size,),
                    generator=g,
                )
                batch = [int(x) for x in all_idx.tolist()]

            if len(batch) > self._batch_size:
                batch = batch[: self._batch_size]
            elif len(batch) < self._batch_size:
                pad = self._batch_size - len(batch)
                if pad > 0:
                    batch.extend(batch[:pad])

            perm = torch.randperm(len(batch), generator=g)
            for idx in perm.tolist():
                yield int(batch[idx])
            self._epoch += 1

    def __len__(self) -> int:
        return 2**31 - 1


class Hdf5MotionDataset(Dataset[MotionClipSample]):
    def __init__(
        self,
        manifest_path: str | Sequence[str],
        max_frame_length: int,
        min_window_length: int = 1,
        window_stride: Optional[int] = None, 
        handpicked_motion_names: Optional[List[str]] = None,
        handpicked_motion_txt: Optional[str] = None,
        handpicked_motion_json: Optional[str] = None,             # [新增] 接收 JSON 路径
        handpicked_motion_cluster_ids: Optional[List[int]] = None, # [新增] 接收想要提取的 cluster_id 列表
        excluded_motion_names: Optional[List[str]] = None,
        world_frame_normalization: bool = True,
        allowed_prefixes: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        
        # 【修改 1】：删除之前写的强制设为 None 的“断电”代码
        # handpicked_motion_names = None
        # handpicked_motion_txt = None

        if max_frame_length <= 0:
            raise ValueError("max_frame_length must be positive")

        self.max_frame_length = int(max_frame_length)
        self.min_window_length = int(min_window_length)

        self.disable_windowing = window_stride is not None and int(window_stride) <= 0
        if self.disable_windowing:
            self.window_stride = self.max_frame_length
        else:
            self.window_stride = int(window_stride) if window_stride is not None and window_stride > 0 else self.max_frame_length
        
        # ====== [修改 2] 融合 list, txt 以及 json 三种白名单读取方式 ======
        whitelist_names = set()
        
        # 1. 从原本的 List 载入
        if handpicked_motion_names is not None:
            whitelist_names.update(handpicked_motion_names)
            
        # 2. 从 txt 文件载入
        if handpicked_motion_txt is not None:
            if not os.path.exists(handpicked_motion_txt):
                raise FileNotFoundError(f"Whitelist txt file not found at {handpicked_motion_txt}")
            with open(handpicked_motion_txt, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  
                        whitelist_names.add(line)

        # 3. [新增] 从 JSON 文件和指定的 Cluster IDs 载入
        if handpicked_motion_json is not None and handpicked_motion_cluster_ids is not None:
            if not os.path.exists(handpicked_motion_json):
                raise FileNotFoundError(f"Whitelist json file not found at {handpicked_motion_json}")
            with open(handpicked_motion_json, 'r', encoding='utf-8') as f:
                import json
                cluster_data = json.load(f)
                clusters = cluster_data.get("clusters", {})
                # 遍历传入的 ID 列表
                for c_id in handpicked_motion_cluster_ids:
                    c_id_str = str(c_id) # JSON 中的键是字符串
                    if c_id_str in clusters:
                        motions = clusters[c_id_str].get("motions", [])
                        whitelist_names.update(motions)
                    else:
                        logger.warning(f"Cluster ID {c_id_str} 未在 {handpicked_motion_json} 中找到。")
        
        # 将合并去重后的集合转为列表，如果为空则设为 None 以便后续判断
        self.handpicked_motion_names = list(whitelist_names) if whitelist_names else None
        # ==========================================================

        self.excluded_motion_names = (
            set(excluded_motion_names)
            if excluded_motion_names is not None
            else None
        )
        # ... 后面的代码保持不变 ...
        self._world_frame_normalization_enabled = bool(
            world_frame_normalization
        )
        self._allowed_prefixes: Tuple[str, ...] = ("ref_", "ft_ref_")

        # Normalize manifest path(s) to a list for aggregation.
        if isinstance(manifest_path, (str, os.PathLike)):
            manifest_paths: List[str] = [str(manifest_path)]
        else:
            manifest_paths = [str(p) for p in manifest_path]
        if len(manifest_paths) == 0:
            raise ValueError("At least one manifest_path must be provided")

        # Aggregate shards and clips across one or many manifests into a single
        # logical dataset. Clip keys must be globally unique.
        self.hdf5_root = os.path.dirname(manifest_paths[0])
        self._manifest_paths: List[str] = manifest_paths
        self._shard_paths: List[str] = []
        self.shards: List[Dict[str, Any]] = []
        self.clips: Dict[str, Dict[str, Any]] = {}

        for mp in manifest_paths:
            if not os.path.exists(mp):
                raise FileNotFoundError(
                    f"HDF5 manifest not found at {mp}. "
                    "Please set robot.motion.hdf5_root/train_hdf5_roots "
                    "to the correct path."
                )
            with open(mp, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)

            root = os.path.dirname(mp)
            shards_local = list(manifest.get("hdf5_shards", []))
            clips_local = manifest.get("clips", {})

            shard_offset = len(self.shards)
            for shard_meta in shards_local:
                self.shards.append(shard_meta)
                rel = shard_meta.get("file", None)
                if not isinstance(rel, str) or not rel:
                    raise ValueError(
                        f"Shard entry in manifest {mp} is missing a valid 'file' field"
                    )
                self._shard_paths.append(os.path.join(root, rel))

            for key, meta in clips_local.items():
                if key in self.clips:
                    raise ValueError(
                        f"Duplicate motion clip key '{key}' found in multiple "
                        "manifests; clip keys must be globally unique."
                    )
                meta_global = dict(meta)
                meta_global["shard"] = (
                    int(meta_global.get("shard", 0)) + shard_offset
                )
                self.clips[key] = meta_global

        if len(self.shards) == 0:
            raise ValueError(
                f"No HDF5 shards listed in manifests: {', '.join(manifest_paths)}"
            )

        self.windows: List[MotionWindow] = self._enumerate_windows()
        if len(self.windows) == 0:
            raise ValueError(
                "No motion windows satisfy the requested frame length constraints"
            )

        # LRU cache of open HDF5 shard handles; size is bounded to avoid
        # unbounded host-memory usage from per-file raw chunk caches.
        self._file_handles: "OrderedDict[int, h5py.File]" = OrderedDict()
        max_open_env = os.getenv("HOLOMOTION_HDF5_MAX_OPEN_SHARDS")
        if max_open_env is None:
            self._max_open_files = 64
        else:
            self._max_open_files = max(1, int(max_open_env))

    def _enumerate_windows(self) -> List[MotionWindow]:
        windows: List[MotionWindow] = []
        motion_cnt = 0
        for motion_key, meta in self.clips.items():
            # ====== [修改] 白名单子串匹配逻辑 ======
            if self.handpicked_motion_names is not None:
                is_matched = False
                for allowed_name in self.handpicked_motion_names:
                    # 判断白名单内的名称是否是 motion_key 的子串
                    if allowed_name in motion_key:
                        is_matched = True
                        break
                    
                
                # 如果遍历完所有白名单规则都没有匹配上，则跳过该片段
                if not is_matched:
                    continue
            # ========================================
                
            if (
                self.excluded_motion_names is not None
                and motion_key in self.excluded_motion_names
            ):
                continue
            
            # 筛到了想要的动作
            motion_cnt += 1

            shard_index = int(meta.get("shard", 0))
            start = int(meta.get("start", 0))
            length = int(meta.get("length", 0))

            if length <= 0:
                continue

            remaining = length
            offset = 0
            window_index = 0
            # ======== 修复点：如果是禁用切窗模式，直接添加一整个 Clip 后跳过循环 ========
            if self.disable_windowing:
                if length >= self.min_window_length:
                    unique_key = f"{motion_key}__start_{start}_len_{length}"
                    windows.append(
                        MotionWindow(
                            motion_key=unique_key,
                            shard_index=shard_index,
                            start=start,
                            length=length,
                            raw_motion_key=motion_key,
                            window_index=0,
                        )
                    )
                continue # 处理完这个 Clip，直接 continue 下一个
            # ======================================================================
            # [修改] 使用滑动窗口进行切片
            while offset < length:
                window_length = min(self.max_frame_length, length - offset)
                if window_length >= self.min_window_length:
                    win_start = start + offset
                    unique_key = f"{motion_key}__start_{win_start}_len_{window_length}"
                    windows.append(
                        MotionWindow(
                            motion_key=unique_key,
                            shard_index=shard_index,
                            start=win_start,
                            length=window_length,
                            raw_motion_key=motion_key,
                            window_index=window_index,
                        )
                    )
                    window_index += 1           
                # [关键] 按配置的 stride 步进，而不是按 window_length 步进
                offset += self.window_stride

        logger.info(f"成功加载并筛选出 {motion_cnt} 个 motion.")                
        logger.info(f"成功加载并筛选出 {len(windows)} 个 motion windows.")

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> MotionClipSample:
        window = self.windows[index]
        shard_handle = self._get_shard_handle(window.shard_index)
        start, end = window.start, window.start + window.length

        arrays: Dict[str, Tensor] = {}

        # ================= [新增: 灵活的键名匹配函数] =================
        def _find_actual_key(handle, base_name, prefix):
            """尝试多种格式组合：带前缀单数、带前缀复数、无前缀单数、无前缀复数"""
            candidates = [
                f"{prefix}{base_name}",     # 例: ref_dof_vel
                f"{prefix}{base_name}s",    # 例: ref_dof_vels
                base_name,                  # 例: dof_vel
                f"{base_name}s"             # 例: dof_vels
            ]
            for cand in candidates:
                if cand in handle:
                    return cand
            return None
        # ==========================================================

        # Mandatory reference source: ref_*
        for logical_name, dataset_name in MANDATORY_DATASETS.items():
            # [修改] 使用灵活匹配代替硬编码
            actual_key = _find_actual_key(shard_handle, dataset_name, "ref_")
            if actual_key is None:
                raise KeyError(
                    f"Missing mandatory dataset '{dataset_name}' (tried permutations with/without 'ref_' and 's') "
                    f"in shard index {window.shard_index}"
                )
            np_array = shard_handle[actual_key][start:end]
            arrays[f"ref_{logical_name}"] = torch.from_numpy(np_array).to(
                torch.float32
            )

        # Optional filtered reference source: ft_ref_*
        for logical_name, dataset_name in MANDATORY_DATASETS.items():
            # [修改] 使用灵活匹配代替硬编码
            actual_key = _find_actual_key(shard_handle, dataset_name, "ft_ref_")
            if actual_key is not None:
                np_array = shard_handle[actual_key][start:end]
                arrays[f"ft_ref_{logical_name}"] = torch.from_numpy(
                    np_array
                ).to(torch.float32)

        if "frame_flag" in shard_handle:
        # ... 后面的代码保持完全不变 ...
            frame_flag_np = shard_handle["frame_flag"][start:end]
            frame_flag = torch.from_numpy(frame_flag_np).to(torch.long)
        else:
            frame_flag = torch.ones(window.length, dtype=torch.long)
            if window.length > 1:
                frame_flag[0] = 0
                frame_flag[-1] = 2
            elif window.length == 1:
                # Single-frame window: mark as both start and end (use 2 for end)
                frame_flag[0] = 2
        arrays["frame_flag"] = frame_flag

        if self._world_frame_normalization_enabled:
            _normalize_window_world_frame(arrays)

        # Derived root_* for ref_* (after normalization)
        arrays["ref_root_pos"] = arrays["ref_rg_pos"][:, 0, :]
        arrays["ref_root_rot"] = arrays["ref_rb_rot"][:, 0, :]
        arrays["ref_root_vel"] = arrays["ref_body_vel"][:, 0, :]
        arrays["ref_root_ang_vel"] = arrays["ref_body_ang_vel"][:, 0, :]

        # Derived root_* for optional ft_ref_* (after normalization)
        if (
            "ft_ref_rg_pos" in arrays
            and "ft_ref_rb_rot" in arrays
            and "ft_ref_body_vel" in arrays
            and "ft_ref_body_ang_vel" in arrays
        ):
            arrays["ft_ref_root_pos"] = arrays["ft_ref_rg_pos"][:, 0, :]
            arrays["ft_ref_root_rot"] = arrays["ft_ref_rb_rot"][:, 0, :]
            arrays["ft_ref_root_vel"] = arrays["ft_ref_body_vel"][:, 0, :]
            arrays["ft_ref_root_ang_vel"] = arrays["ft_ref_body_ang_vel"][
                :, 0, :
            ]

        return MotionClipSample(
            motion_key=window.motion_key,
            raw_motion_key=window.raw_motion_key,
            tensors=arrays,
            length=window.length,
        )

    def _get_shard_handle(self, shard_index: int) -> h5py.File:
        if shard_index in self._file_handles:
            handle = self._file_handles.pop(shard_index)
            if handle.id:
                # Mark as most recently used.
                self._file_handles[shard_index] = handle
                return handle

        if shard_index < 0 or shard_index >= len(self._shard_paths):
            raise IndexError(
                f"Shard index {shard_index} out of range for "
                f"{len(self._shard_paths)} available shards"
            )
        shard_path = self._shard_paths[shard_index]
        # Open with SWMR and a configurable raw chunk cache to speed up repeated reads.
        # The default cache size (in bytes) can be overridden via the
        # HOLOMOTION_HDF5_RDCC_NBYTES environment variable.
        rdcc_nbytes_env = os.getenv("HOLOMOTION_HDF5_RDCC_NBYTES")
        if rdcc_nbytes_env is None:
            rdcc_nbytes = 256 * 1024 * 1024  # 256MB default
        else:
            rdcc_nbytes = int(rdcc_nbytes_env)
        handle = h5py.File(
            shard_path,
            "r",
            libver="latest",
            swmr=True,
            rdcc_nbytes=rdcc_nbytes,
            rdcc_w0=0.75,
        )
        # Enforce LRU limit on the number of simultaneously open shard files.
        if (
            self._max_open_files is not None
            and len(self._file_handles) >= self._max_open_files
        ):
            old_index, old_handle = self._file_handles.popitem(last=False)
            old_handle.close()
        self._file_handles[shard_index] = handle
        return handle

    def close(self) -> None:
        """Close all open HDF5 shard handles for this dataset."""
        for handle in self._file_handles.values():
            if handle.id:
                handle.close()
        self._file_handles.clear()


class MotionClipBatchCache:
    """Double-buffered motion cache for RL training and evaluation with Async Global Curriculum."""

    def __init__(
        self,
        train_dataset: Hdf5MotionDataset,
        *,
        val_dataset: Optional[Hdf5MotionDataset] = None,
        batch_size: int,
        stage_device: Optional[torch.device] = None,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        sampler_rank: int = 0,
        sampler_world_size: int = 1,
        allowed_prefixes: Optional[Sequence[str]] = None,
        swap_interval_steps: Optional[int] = None,
        force_timeout_on_swap: bool = True,
        curriculum_cfg: Optional[Dict[str, Any]] = None,  # [新增] 接收课程学习配置
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
            
        # ======== [新增: 挂载 Curriculum 配置] ========
        self.curriculum_cfg = curriculum_cfg or {}
        # ============================================

        self._datasets = {
            "train": train_dataset,
            "val": val_dataset if val_dataset is not None else train_dataset,
        }
        self._mode = "train"
        self._seed = int(time.time_ns() & 0x7FFFFFFF)
        self._batch_size = batch_size
        
        # ======== [修复: 确保 stage_device 是 torch.device 对象] ========
        if stage_device is not None:
            self._stage_device = torch.device(stage_device) if isinstance(stage_device, str) else stage_device
        else:
            self._stage_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ================================================================
            
        self._num_workers = num_workers
        self._prefetch_factor = prefetch_factor
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers
        self._sampler_rank = sampler_rank
        self._sampler_world_size = sampler_world_size
        self._allowed_prefixes = allowed_prefixes

        self.swap_interval_steps = (
            swap_interval_steps
            if swap_interval_steps is not None
            else train_dataset.max_frame_length
        )
        self.force_timeout_on_swap = force_timeout_on_swap

        self._loaders: Dict[str, DataLoader] = {}
        self._iterators = {}
        self._sampler = None
        self._train_sampler = None
        self._curriculum_sampler = None  # [新增] 专门存放我们的层级采样器指针

        self._weighted_bin_enabled: bool = False
        self._weighted_bin_bins: Optional[List[List[int]]] = None
        self._weighted_bin_ratios: Optional[List[float]] = None
        self._weighted_bin_specs: Optional[List[Dict[str, Any]]] = None

        self._copy_stream = None
        self._pending_ready_event = None
        self._current_ready_event = None
        self._next_ready_event = None

        self._build_dataloader()
        self._prime_buffers()

        self.swap_index = 0
        self._step_counter = 0
        
    def _build_dataloader(self) -> None:
        dataset = self._datasets[self._mode]
        ds_len = len(dataset)
        
        if self._mode == "val":
            if self._sampler_world_size > 1:
                self._sampler = DistributedSampler(
                    dataset,
                    num_replicas=self._sampler_world_size,
                    rank=self._sampler_rank,
                    shuffle=False,
                    drop_last=False,
                )
            else:
                self._sampler = None
        else:
            # ======== [新增: 激活全局两级异步 Sampler] ========
            curr_enabled = self.curriculum_cfg.get("enabled", True)
            if curr_enabled:
                seed = self._seed + self._sampler_rank * 100003
                sampler = CurriculumHierarchicalSampler(
                    dataset=dataset,
                    # --- 替换为新的贝叶斯平滑参数 ---
                    gamma_decay=self.curriculum_cfg.get("gamma_decay", 0.99),
                    laplace_eps=self.curriculum_cfg.get("laplace_eps", 1.0),
                    laplace_kappa=self.curriculum_cfg.get("laplace_kappa", 2.0),
                    # ------------------------------
                    temp_start=self.curriculum_cfg.get("temp_start", 3.0),
                    temp_end=self.curriculum_cfg.get("temp_end", 0.15),
                    gamma_start=self.curriculum_cfg.get("gamma_start", 0.15),
                    gamma_end=self.curriculum_cfg.get("gamma_end", 0.05),
                    decay_start=self.curriculum_cfg.get("decay_start", 5000),
                    decay_end=self.curriculum_cfg.get("decay_end", 35000),
                    seed=seed,
                )
                self._curriculum_sampler = sampler  # 明确锚定专有指针！
                self._train_sampler = sampler
                self._sampler = sampler
                logger.info(f"Mounted CurriculumHierarchicalSampler on rank {self._sampler_rank}")
            # ===================================================
            elif (
                self._weighted_bin_enabled
                and self._weighted_bin_bins is not None
                and self._weighted_bin_ratios is not None
            ):
                seed = self._seed + self._sampler_rank * 100003
                # [修复]: 补齐 batch_size，并将 bin_ratios 改正为 ratios
                self._sampler = WeightedBinInfiniteSampler(
                    dataset_len=ds_len,
                    bin_indices=self._weighted_bin_bins,
                    ratios=self._weighted_bin_ratios,
                    batch_size=self._batch_size,
                    seed=seed,
                )
                self._train_sampler = self._sampler
            else:
                seed = self._seed + self._sampler_rank * 100003
                # [修复]: 必须传入 data_source=dataset，而非 dataset_len
                self._sampler = InfiniteRandomSampler(
                    data_source=dataset, seed=seed
                )
                self._train_sampler = self._sampler

        # [修复]: 换回正确的 _cache_collate_fn 拼装函数
        loader_kwargs = {
            "batch_size": self._batch_size,
            "num_workers": self._num_workers,
            "pin_memory": self._pin_memory,
            "collate_fn": lambda batch: _cache_collate_fn(
                batch, mode=self._mode, batch_size=self._batch_size
            ),
        }

        if self._prefetch_factor is not None and self._num_workers > 0:
            loader_kwargs["prefetch_factor"] = self._prefetch_factor

        if self._num_workers > 0:
            loader_kwargs["persistent_workers"] = self._persistent_workers

        if self._sampler is not None:
            loader_kwargs["sampler"] = self._sampler
            loader_kwargs["shuffle"] = False
        else:
            loader_kwargs["shuffle"] = True

        self._loaders[self._mode] = DataLoader(
            self._datasets[self._mode], **loader_kwargs
        )
        self._iterators[self._mode] = iter(self._loaders[self._mode])

    def set_mode(self, mode: str) -> None:
        if mode not in ["train", "val"]:
            raise ValueError(f"Invalid mode: {mode}")
        if self._mode == mode:
            return
        self._mode = mode
        if self._mode == "train":
            self._sampler = getattr(self, "_train_sampler", None)
        else:
            self._build_dataloader()
        self._prime_buffers()

    def _prime_buffers(self) -> None:
        # [新增] 辅助函数：安全地获取下一个 batch，耗尽则重置
        def safe_next_batch():
            try:
                return next(self._iterators[self._mode])
            except StopIteration:
                self._iterators[self._mode] = iter(self._loaders[self._mode])
                return next(self._iterators[self._mode])
        if self._stage_device.type == "cuda":
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream(device=self._stage_device)
            self._current_ready_event = torch.cuda.Event(
                enable_timing=False, blocking=False
            )
            self._next_ready_event = torch.cuda.Event(
                enable_timing=False, blocking=False
            )
            self._pending_ready_event = torch.cuda.Event(
                enable_timing=False, blocking=False
            )
            
            with torch.cuda.stream(self._copy_stream):
                batch_0 = safe_next_batch()  # [修改] 替换 next(...)
                self._current_batch = batch_0.to(self._stage_device, non_blocking=True)
                self._current_ready_event.record(self._copy_stream)

                batch_1 = safe_next_batch()  # [修改] 替换 next(...)
                self._next_batch = batch_1.to(self._stage_device, non_blocking=True)
                self._next_ready_event.record(self._copy_stream)

                batch_2 = safe_next_batch()  # [修改] 替换 next(...)
                self._pending_batch = batch_2.to(self._stage_device, non_blocking=True)
                self._pending_ready_event.record(self._copy_stream)
        else:
            self._current_batch = safe_next_batch().to(self._stage_device) # [修改]
            self._next_batch = safe_next_batch().to(self._stage_device)    # [修改]
            self._pending_batch = safe_next_batch().to(self._stage_device) # [修改]

    @property
    def current_batch(self):
        if self._stage_device.type == "cuda":
            self._current_ready_event.synchronize()
        return self._current_batch

    @property
    def clip_count(self) -> int:
        batch = self.current_batch
        return int(batch.lengths.shape[0])

    @property
    def num_batches(self) -> int:
        loader = self._loaders.get(self._mode, None)
        if loader is None:
            return 1
        try:
            return max(1, int(len(loader)))
        except Exception:
            return 1

    def lengths_for_indices(self, indices: torch.Tensor) -> torch.Tensor:
        batch = self.current_batch
        return batch.lengths[indices]

    def motion_keys_for_indices(self, indices: torch.Tensor) -> List[str]:
        batch = self.current_batch
        idx_np = indices.detach().cpu().numpy()
        return [batch.motion_keys[i] for i in idx_np]

    def sample_env_assignments(
        self,
        num_envs: int,
        n_future_frames: int,
        device: torch.device,
        *,
        deterministic_start: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.current_batch
        lengths = batch.lengths.to(device)

        if num_envs <= 0:
            raise ValueError("num_envs must be positive")

        self._step_counter += 1
        total = int(lengths.shape[0])
        if total == 0:
            raise ValueError("Cannot sample from an empty batch.")

        # ======== [修改: 极速 GPU 无放回盲抽] ========
        if num_envs == total:
            # 完美匹配，100% 榨干缓存，不浪费任何一个动作
            clip_indices = torch.randperm(total, device=device)
        elif num_envs < total:
            # 环境少于缓存，取子集
            clip_indices = torch.randperm(total, device=device)[:num_envs]
        else:
            # 环境多于缓存（如 8192 个 Env 对应 4096 缓存），必须有放回或循环使用
            repeats = num_envs // total
            remainder = num_envs % total
            perms = [torch.randperm(total, device=device) for _ in range(repeats)]
            if remainder > 0:
                perms.append(torch.randperm(total, device=device)[:remainder])
            clip_indices = torch.cat(perms)
        # ===================================================

        # 帧内抖动逻辑 (利用 stride 模拟 Mosaic)
        max_valid = torch.clamp(lengths[clip_indices] - 1 - n_future_frames, min=0)
        dataset = self._datasets.get(self._mode)
        stride = getattr(dataset, "window_stride", batch.max_frame_length)

        if deterministic_start:
            frame_starts = torch.zeros_like(max_valid)
        else:
            # ================= [修改此处逻辑] =================
            # 判断是否禁用了切窗。如果是，则在整段动作 [0, max_valid] 内随机初始化
            if getattr(dataset, "disable_windowing", False):
                # max_jitter = max_valid
                # 维持 window_stride=-1 的配置，但取消随机抖动，强制从动作的开头(0帧)初始化
                max_jitter = torch.zeros_like(max_valid)
            else:
                # 否则，维持原有的 Mosaic 抖动逻辑，限制在 stride 范围内
                stride = getattr(dataset, "window_stride", batch.max_frame_length)
                max_jitter = torch.clamp(torch.full_like(max_valid, stride - 1), max=max_valid)
            
            rand = torch.rand_like(max_jitter, dtype=torch.float32)
            frame_starts = torch.floor(rand * (max_jitter + 1).float()).to(torch.long)
            # =================================================

        # 这里就是之前可能不小心漏掉的 return 语句
        return clip_indices, frame_starts

    def gather_state(
        self,
        clip_indices: torch.Tensor,
        frame_indices: torch.Tensor,
        n_future_frames: int,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Gather (1 + n_future_frames) of data for the specified clips and frames."""
        # ======== [修复: 自动推断 device, 兼容旧接口调用] ========
        if device is None:
            device = clip_indices.device
        # ========================================================
            
        batch = self.current_batch
        num_envs = clip_indices.shape[0]
        seq_len = n_future_frames + 1

        # 构造二维索引张量: [num_envs, seq_len]
        t_offsets = torch.arange(seq_len, device=device).unsqueeze(0)
        t_indices = frame_indices.unsqueeze(1) + t_offsets
        c_indices = clip_indices.unsqueeze(1).expand(num_envs, seq_len)

        out = {}
        for name, tensor in batch.tensors.items():
            # tensor shape: [batch_size, max_frame_length, ...]
            out[name] = tensor[c_indices, t_indices].to(device)

        return out

    def advance(self) -> None:
        # [新增]
        def safe_next_batch():
            try:
                return next(self._iterators[self._mode])
            except StopIteration:
                self._iterators[self._mode] = iter(self._loaders[self._mode])
                return next(self._iterators[self._mode])

        if self._stage_device.type == "cuda":
            self._next_ready_event.synchronize()
            self._current_batch = self._next_batch
            self._current_ready_event = self._next_ready_event
            self._next_batch = getattr(self, "_pending_batch", None)
            self._next_ready_event = getattr(self, "_pending_ready_event", None)

            with torch.cuda.stream(self._copy_stream):
                new_batch = safe_next_batch()  # [修改] 替换 next(...)
                self._pending_batch = new_batch.to(self._stage_device, non_blocking=True)
                
                if not hasattr(self, "_pending_ready_event") or self._pending_ready_event is None:
                    self._pending_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                self._pending_ready_event.record(self._copy_stream)
        else:
            self._current_batch = self._next_batch
            self._next_batch = getattr(self, "_pending_batch", None)
            if self._next_batch is None:
                self._next_batch = safe_next_batch().to(self._stage_device) # [修改]
            self._pending_batch = safe_next_batch().to(self._stage_device)  # [修改]

        self.swap_index += 1
        self._step_counter = 0
        # ======== [新增: 低频 Rank 0 强制对齐] ========
        # 每当缓存完成一轮替换 (约几千步)，让 Rank 0 的上帝视角概率树强制覆盖其他卡
        self.sync_curriculum_from_rank0()
        # ==========================================

    # =========================================================================
    #                   [新增: 全局异步 Curriculum 核心接口] 
    # =========================================================================

    # 在 MotionClipBatchCache 类中，新增以下方法：
    def set_global_iteration(self, global_rl_iter: int) -> None:
        self._current_rl_iter = global_rl_iter
        sampler = getattr(self, "_curriculum_sampler", None)
        if sampler is not None and hasattr(sampler, "set_iteration"):
            sampler.set_iteration(global_rl_iter)

    def update_global_scores(self, events: List[tuple]) -> None:
        sampler = getattr(self, "_curriculum_sampler", None)
        if sampler is not None and hasattr(sampler, "update_scores"):
            # 获取当前保存的 RL iteration 透传进去
            current_iter = getattr(self, "_current_rl_iter", None)
            sampler.update_scores(events, global_rl_iter=current_iter)

    def save_curriculum_state(self, path: str) -> bool:
        """原生纯净接口：提取专属 Sampler 状态并正名保存"""
        import json
        sampler = getattr(self, "_curriculum_sampler", None)
        if sampler is not None and hasattr(sampler, "state_dict"):
            state = sampler.state_dict()
            
            # --- [新增] 计算当下的实时温度和探索率，方便在 JSON 中直接监控 ---
            current_temp = 3.0
            current_gamma = 0.15
            if hasattr(sampler, '_get_decayed_value'):
                current_temp = sampler._get_decayed_value(sampler.temp_start, sampler.temp_end, sampler.current_iteration)
                current_gamma = sampler._get_decayed_value(sampler.gamma_start, sampler.gamma_end, sampler.current_iteration)
            # -------------------------------------------------------------

            state["config"] = {
                "scorer": sampler.__class__.__name__,
                
                # 1. 贝叶斯平滑常量
                "gamma_decay": getattr(sampler, "gamma_decay", 0.9993),
                "laplace_eps": getattr(sampler, "laplace_eps", 1.0),
                "laplace_kappa": getattr(sampler, "laplace_kappa", 2.0),
                
                # 2. 退火边界常量 (Hyperparameters)
                "temp_start": getattr(sampler, "temp_start", 3.0),
                "temp_end": getattr(sampler, "temp_end", 0.15),
                "gamma_start": getattr(sampler, "gamma_start", 0.15),
                "gamma_end": getattr(sampler, "gamma_end", 0.05),
                "decay_start": getattr(sampler, "decay_start", 5000),
                "decay_end": getattr(sampler, "decay_end", 35000),
                
                # 3. 实时状态监控 (Dynamic Variables - 随 Iteration 变化)
                "REALTIME_current_temp": current_temp,
                "REALTIME_current_gamma": current_gamma
            }
            
            with open(path, "w") as f:
                json.dump(state, f, indent=4)
            return True
        return False

    def load_curriculum_state(self, path: str) -> None:
        """断点续训接口：精准恢复全局难度字典"""
        import json
        import os
        sampler = getattr(self, "_curriculum_sampler", None)
        if sampler is not None and os.path.exists(path):
            with open(path, "r") as f:
                state = json.load(f)
            sampler.load_state_dict(state)
            logger.info(f"Loaded valid curriculum state from {path}")

    def sync_curriculum_from_rank0(self) -> None:
        """
        [解决 DDP 多卡分歧]
        利用 torch.distributed 强制要求所有其他显卡 (Rank 1...N)
        覆写为 主卡 (Rank 0) 当前最新的 Curriculum 记分牌。
        这保证了多卡分布式训练下真正的 "全局统一课程"。
        """
        sampler = getattr(self, "_curriculum_sampler", None)
        if sampler is None:
            return
            
        import torch.distributed as dist
        if not dist.is_initialized():
            return
            
        # 准备一个承载数据的列表
        state_list = [None]
        
        # Rank 0 把自己的记分牌放进去
        if dist.get_rank() == 0:
            state_list[0] = sampler.state_dict()
            
        # 广播给所有人
        dist.broadcast_object_list(state_list, src=0)
        
        # 其他人接收并强制更新自己的记分牌
        if dist.get_rank() != 0 and state_list[0] is not None:
            sampler.load_state_dict(state_list[0])
