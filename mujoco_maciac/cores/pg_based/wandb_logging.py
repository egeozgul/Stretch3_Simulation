"""Shared wandb init + eval-logging helpers for all pg_based algorithms.

Before this module, each algorithm (MacIAC, MacCAC, ACAC, MacIAICC, ...)
maintained its own copy of the wandb.init and per-eval log_dict construction
code. Feature coverage drifted (MacIAC logged a dozen metrics; MacCAC logged
two) and the `project` field was set to `save_dir` so every hyperparameter
combination created its OWN wandb project - clicking through 432 projects is
infeasible.

This module centralizes two things:

1. `init_wandb_run(alg_name, save_dir, ...)` pins the wandb project to the
   algorithm class name (e.g. "MacIAC"), with the run's display name set to
   the sweep's save_dir (which already encodes the hyperparameters the sweep
   varied, e.g. "mac_iac_overcooked_D_dual_policy_sweep__a_lr-0.0003_c_lr-0.001_train_freq-8").
   All of one algorithm's runs now live under one dashboard.

2. `build_eval_log_dict(...)` builds the same rich per-eval metric dict that
   MacIAC used to build inline, using hasattr/getattr probes so algorithms
   whose envs_runner doesn't collect every optional field (per-agent-per-
   instruction compliance, episode diagnostics, instruction_counter, etc.)
   still get partial logging without AttributeError.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb

_WANDB_API_KEY = "1953b06a2318828bc531085d9e76a250f82840fd"
_WANDB_ENTITY = "wwlin1198"


def _safe_key(text: str, max_len: int = 50) -> str:
    """Strip apostrophes and spaces from a human instruction string so it's
    usable as a wandb metric key (which must not contain spaces/quotes).
    Truncates to `max_len` to keep key panels readable."""
    return text.replace("'", "").replace(" ", "_")[:max_len]


def init_wandb_run(
    alg_name: str,
    save_dir: str,
    config: Dict[str, Any],
    instr_enabled: bool,
    run_id: int = 0,
    extra_tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> None:
    """Initialize one wandb run under the algorithm-level project.

    Parameters
    ----------
    alg_name : str
        Algorithm class name used as the wandb `project`: e.g. 'MacIAC',
        'MacCAC', 'ACAC', 'MacIAICC'. Stable across sweeps so every run of a
        given algorithm appears in one dashboard.
    save_dir : str
        Sweep-specific directory that already encodes the hyperparameters
        varied in this sweep (e.g.
        'mac_iac_overcooked_D_dual_policy_sweep__a_lr-0.0003_c_lr-0.001_train_freq-8').
        Used as the prefix of the wandb run `name`; the final run name also
        carries `__seed<run_id>__<YYYYMMDD-HHMMSS>` so every seed shows up as
        its own distinct row (not collapsed into a mean/confidence band).
    config : dict
        Hyperparameter dict logged to wandb.config (for post-hoc filtering).
    instr_enabled : bool
        Whether instruction-conditioning is active this run. Controls the
        default tag set.
    run_id : int
        Seed index, also added to config for filtering.
    extra_tags : list[str], optional
        Extra tags beyond the default instructions_{enabled,disabled} tag.
    notes : str, optional
        Free-form notes attached to the run; defaults to an auto-generated
        string mentioning the algorithm and instruction switch mode.
    """
    tags: List[str] = ["instructions_enabled" if instr_enabled else "instructions_disabled"]
    if extra_tags:
        tags.extend(extra_tags)

    if notes is None:
        switch_mode = os.environ.get("INSTRUCTION_SWITCH_MODE", "stochastic")
        notes = (
            f"{alg_name} with instruction switching ({switch_mode} mode)"
            if instr_enabled
            else f"{alg_name} without instructions"
        )

    # Timestamp goes into the display name so multiple seeds of the same
    # hparam config show up as distinct runs, and so that a restarted run
    # (after preemption) doesn't visually collide with the original.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Run name: sweep-level save_dir + seed index + launch timestamp.
    # Example: mac_iac_overcooked_D_..._train_freq-8__seed0__20260218-153401
    run_name = f"{save_dir}__seed{run_id}__{timestamp}"

    # Augment config so the run carries its identity metadata explicitly,
    # independent of the display name. Useful when the wandb UI truncates
    # long sweep names.
    cfg = dict(config)
    cfg.setdefault("run_id", run_id)
    cfg.setdefault("save_dir", save_dir)
    cfg.setdefault("alg", alg_name)
    cfg.setdefault("launch_timestamp", timestamp)

    # Surface the instruction-shaping env-vars in wandb config so runs
    # with different penalty / duration / cadence are filterable in the
    # UI. Defaults match the runner-side defaults; we still record them
    # explicitly so unset == "default value at training time" is visible.
    cfg.setdefault("instruction_enabled", os.environ.get("INSTRUCTION_ENABLED", "0") == "1")
    cfg.setdefault("instruction_switch_mode", os.environ.get("INSTRUCTION_SWITCH_MODE", "stochastic"))
    cfg.setdefault("instruction_provided_prob",
                   float(os.environ.get("INSTRUCTION_PROVIDED_PROB", "0.00347")))
    cfg.setdefault("instruction_penalty",
                   float(os.environ.get("INSTRUCTION_PENALTY", "-50.0")))
    cfg.setdefault("instruction_duration_steps",
                   int(os.environ.get("INSTRUCTION_DURATION_STEPS", "0")))
    cfg.setdefault("use_chain_break", os.environ.get("USE_CHAIN_BREAK", "0") == "1")
    cfg.setdefault("use_value_cancellation",
                   os.environ.get("USE_VALUE_CANCELLATION", "0") == "1")
    raw_instructions = os.environ.get("OVERCOOKED_INSTRUCTIONS")
    if raw_instructions:
        # Stored as a string list — wandb config handles list nicely.
        cfg.setdefault(
            "instructions",
            [s for s in raw_instructions.split("||") if s],
        )

    project_name = os.environ.get("WANDB_PROJECT", alg_name)

    wandb.login(key=_WANDB_API_KEY)
    wandb.init(
        project=project_name,
        entity=_WANDB_ENTITY,
        name=run_name,
        # No `group=` on purpose: we want every seed / every relaunch to
        # appear as its own row in the wandb UI instead of being collapsed
        # into a single mean+confidence band.
        config=cfg,
        notes=notes,
        tags=tags,
        job_type="training",
        reinit=True,
    )

    # Pin every plot to use 'Episode' as the x-axis instead of wandb's
    # auto-incrementing internal step counter. Each call to wandb.log({...,
    # 'Episode': N, ...}) will then place all sibling metrics at x=N. The
    # glob 'step_metric' applies to every key, including ones added later
    # by build_eval_log_dict (Compliance/*, Returns_*, etc.).
    wandb.define_metric("Episode")
    wandb.define_metric("*", step_metric="Episode")


def build_eval_log_dict(
    *,
    epi_count: int,
    eval_return: float,
    envs_runner,
    instruction_texts: Optional[List[str]] = None,
    encoder_agent=None,
    instr_active: bool = True,
) -> Dict[str, Any]:
    """Assemble a per-eval wandb log dict.

    Returns a dict suitable for direct `wandb.log(...)` with at minimum
    'Episode' and 'Returns'. Adds optional sections when the corresponding
    envs_runner buffers are populated, so algorithms can share this helper
    regardless of which diagnostics their envs_runner records.

    Parameters
    ----------
    epi_count : int
        Current episode count.
    eval_return : float
        Mean eval return across eval_num_epi evaluation episodes.
    envs_runner : object
        Parallel-env runner; helper probes these optional attributes:
          - instruction_provider          (required for any instruction metric)
          - eval_compliance               (list[float], per-episode compliance)
          - eval_compliance_per_instruction           (list[dict[str, float]])
          - eval_compliance_per_agent_instruction     (list[dict[(int,str), float]])
          - eval_episode_instructions     (list[(return, inst_text_or_None)])
          - eval_episode_diagnostics      (list[dict] with 'completed', 'horizon_truncated', 'episode_len', 'instruction')
          - instruction_counter           (dict[str, int])
    instruction_texts : list[str], optional
        Instruction strings used this run. Required for embedding cosine/L2
        metrics.
    encoder_agent : object, optional
        Agent whose `.actor_net.encode_instruction(text)` produces an embedding.
        Pass None (e.g. for ACAC's one-hot encoding) to skip embedding metrics.
    instr_active : bool
        Whether instructions are currently provided (stochastic switching may
        temporarily disable them). Controls whether compliance metrics are
        logged.
    """
    log_dict: Dict[str, Any] = {"Episode": int(epi_count), "Returns": float(eval_return)}
    provider_on = getattr(envs_runner, "instruction_provider", None) is not None

    # ---- compliance ----------------------------------------------------------
    avg_compliance = 0.0
    per_instr_compliance: Dict[str, float] = {}
    eval_compliance = getattr(envs_runner, "eval_compliance", None)
    if provider_on and eval_compliance:
        avg_compliance = float(np.mean(eval_compliance))
        ep_per_instr = getattr(envs_runner, "eval_compliance_per_instruction", None)
        if ep_per_instr:
            instr_rates: Dict[str, List[float]] = {}
            for epi_per_instr in ep_per_instr:
                for inst_text, rate in epi_per_instr.items():
                    instr_rates.setdefault(inst_text, []).append(rate)
            per_instr_compliance = {k: float(np.mean(v)) for k, v in instr_rates.items()}

    if provider_on and eval_compliance is not None and instr_active:
        log_dict["Instruction_Compliance"] = avg_compliance
        if instruction_texts:
            for inst_text in instruction_texts:
                log_dict[f"Compliance/{_safe_key(inst_text)}"] = per_instr_compliance.get(inst_text, 0.0)
        else:
            for inst_text, rate in per_instr_compliance.items():
                log_dict[f"Compliance/{_safe_key(inst_text)}"] = rate

        agent_per_instr = getattr(envs_runner, "eval_compliance_per_agent_instruction", None)
        if agent_per_instr:
            agent_rates: Dict[tuple, List[float]] = {}
            for epi_data in agent_per_instr:
                for key, rate in epi_data.items():
                    agent_rates.setdefault(key, []).append(rate)
            for (agent_idx, inst_text), rates in agent_rates.items():
                log_dict[f"Compliance_Agent{agent_idx}/{_safe_key(inst_text, 40)}"] = float(np.mean(rates))
            # Per-agent overall compliance (averaged across instructions).
            agent_overall: Dict[int, List[float]] = {}
            for (agent_idx, _), rates in agent_rates.items():
                agent_overall.setdefault(agent_idx, []).extend(rates)
            for agent_idx, rates in agent_overall.items():
                if rates:
                    log_dict[f"Compliance_Agent{agent_idx}/Overall"] = float(np.mean(rates))

    # ---- per-instruction returns + with/without-instruction split ----------
    eval_ep_instr = getattr(envs_runner, "eval_episode_instructions", None)
    if eval_ep_instr:
        with_instr = [r for r, inst in eval_ep_instr if inst is not None]
        no_instr = [r for r, inst in eval_ep_instr if inst is None]
        if with_instr:
            log_dict["Returns_With_Instruction"] = float(np.mean(with_instr))
        if no_instr:
            log_dict["Returns_Without_Instruction"] = float(np.mean(no_instr))

        per_instr_returns: Dict[str, List[float]] = {}
        for r, inst in eval_ep_instr:
            if inst is not None:
                per_instr_returns.setdefault(inst, []).append(r)
        for inst_text, returns in per_instr_returns.items():
            log_dict[f"Returns_Instruction/{_safe_key(inst_text)}"] = float(np.mean(returns))

    # ---- shaped (penalty-included) returns ---------------------------------
    # Returns_With_Instruction_Shaped lets you see the return the critic
    # actually saw — i.e. raw env reward + INSTRUCTION_PENALTY for any
    # non-compliant step. Compare against Returns_With_Instruction (raw)
    # to quantify how much penalty the agent "ate" per episode. The gap
    # between the two is the cumulative shaping signal.
    eval_ep_instr_shaped = getattr(envs_runner, "eval_episode_instructions_shaped", None)
    if eval_ep_instr_shaped:
        with_instr_s = [r for r, inst in eval_ep_instr_shaped if inst is not None]
        no_instr_s = [r for r, inst in eval_ep_instr_shaped if inst is None]
        if with_instr_s:
            log_dict["Returns_With_Instruction_Shaped"] = float(np.mean(with_instr_s))
        if no_instr_s:
            log_dict["Returns_Without_Instruction_Shaped"] = float(np.mean(no_instr_s))
        per_instr_shaped: Dict[str, List[float]] = {}
        for r, inst in eval_ep_instr_shaped:
            if inst is not None:
                per_instr_shaped.setdefault(inst, []).append(r)
        for inst_text, returns in per_instr_shaped.items():
            log_dict[f"Returns_Instruction_Shaped/{_safe_key(inst_text)}"] = float(np.mean(returns))

    # ---- completion diagnostics (why low returns? truncation vs failure) ---
    eval_diags = getattr(envs_runner, "eval_episode_diagnostics", None)
    if eval_diags:
        with_instr = [d for d in eval_diags if d.get("instruction") is not None]
        no_instr = [d for d in eval_diags if d.get("instruction") is None]
        if with_instr:
            log_dict["Completion_With_Instruction"] = float(
                np.mean([1.0 if d.get("completed", False) else 0.0 for d in with_instr])
            )
            log_dict["HorizonTrunc_With_Instruction"] = float(
                np.mean([1.0 if d.get("horizon_truncated", False) else 0.0 for d in with_instr])
            )
            log_dict["EpisodeLen_With_Instruction"] = float(
                np.mean([d.get("episode_len", 0) for d in with_instr])
            )
        if no_instr:
            log_dict["Completion_Without_Instruction"] = float(
                np.mean([1.0 if d.get("completed", False) else 0.0 for d in no_instr])
            )
            log_dict["HorizonTrunc_Without_Instruction"] = float(
                np.mean([1.0 if d.get("horizon_truncated", False) else 0.0 for d in no_instr])
            )
            log_dict["EpisodeLen_Without_Instruction"] = float(
                np.mean([d.get("episode_len", 0) for d in no_instr])
            )

    # ---- BERT embedding geometry (skipped for algs without a text encoder) -
    if encoder_agent is not None and instruction_texts and len(instruction_texts) >= 2:
        try:
            actor = encoder_agent.actor_net
            with torch.no_grad():
                embs = [actor.encode_instruction(t).squeeze(0) for t in instruction_texts]
                emb_stack = torch.stack(embs)
                norms = emb_stack.norm(dim=1, keepdim=True).clamp(min=1e-8)
                emb_normed = emb_stack / norms
                cos_sim = emb_normed @ emb_normed.t()
                n = len(instruction_texts)
                for i in range(n):
                    for j in range(i + 1, n):
                        si = _safe_key(instruction_texts[i], 25)
                        sj = _safe_key(instruction_texts[j], 25)
                        log_dict[f"Embedding_Cosine/{si}_vs_{sj}"] = float(cos_sim[i, j])
                        log_dict[f"Embedding_L2_Dist/{si}_vs_{sj}"] = float(
                            torch.norm(emb_stack[i] - emb_stack[j], p=2)
                        )

                # Scatter plot of the full cosine matrix (optional richer view).
                sim_data = []
                for i in range(n):
                    for j in range(n):
                        sim_data.append([
                            instruction_texts[i][:30],
                            instruction_texts[j][:30],
                            float(cos_sim[i, j]),
                        ])
                sim_table = wandb.Table(
                    columns=["instruction_a", "instruction_b", "cosine_similarity"],
                    data=sim_data,
                )
                log_dict["Embedding_Similarity_Matrix"] = wandb.plot.scatter(
                    sim_table,
                    "instruction_a",
                    "instruction_b",
                    title="Instruction Embedding Cosine Similarity",
                )
        except Exception as e:
            # Don't let a wandb plotting hiccup kill the training run.
            print(f"[wandb_logging] embedding metrics skipped: {e}")

    # ---- instruction occurrence distribution --------------------------------
    counter = getattr(envs_runner, "instruction_counter", None)
    if counter:
        for inst_text, count in counter.items():
            key = "no_instruction" if inst_text == "__no_instruction__" else _safe_key(inst_text)
            log_dict[f"Instruction_Count/{key}"] = count
        count_data = [
            [(k if k != "__no_instruction__" else "no_instruction"), v]
            for k, v in counter.items()
        ]
        if count_data:
            table = wandb.Table(columns=["Instruction", "Count"], data=count_data)
            log_dict["Instruction_Count_Distribution"] = wandb.plot.bar(
                table, "Instruction", "Count", title="Instruction Occurrence Count"
            )

    return log_dict


def clear_eval_buffers(envs_runner) -> None:
    """Reset the per-eval envs_runner buffers so next eval starts empty.

    Kept here so every alg clears the same set, preventing a buffer added in
    one alg's envs_runner from leaking into the next eval cycle of another.
    """
    for attr in (
        "eval_compliance",
        "eval_compliance_per_instruction",
        "eval_compliance_per_agent_instruction",
        "eval_episode_instructions",
        "eval_episode_diagnostics",
    ):
        if hasattr(envs_runner, attr):
            setattr(envs_runner, attr, [])
    # `instruction_counter` is a dict, not a list.
    if hasattr(envs_runner, "instruction_counter") and isinstance(
        getattr(envs_runner, "instruction_counter"), dict
    ):
        envs_runner.instruction_counter = {}
