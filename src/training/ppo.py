"""
PPO loss computation for the constrained clarification agent.

The combined Lagrangian objective is:
    L_ppo = -E[ min(r·A_lag, clip(r, 1-ε, 1+ε)·A_lag) ]
            - entropy_coeff * H(π)
            + kl_coeff * KL(π || π_ref)

where:
    A_lag = A_reward - λ₁ · A_q_cost - λ₂ · A_t_cost
    r     = π_θ(a|s) / π_θ_old(a|s)   (probability ratio)
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


def compute_lagrangian_advantages(
    adv_r: List[float],
    adv_q: List[float],
    adv_t: List[float],
    lambda1: float,
    lambda2: float,
    device: torch.device,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Combine three advantage streams into a single Lagrangian advantage.

    A_lag = A_reward - λ₁ · A_q_cost - λ₂ · A_t_cost

    Args:
        adv_r:     reward advantages  (list of floats, one per transition)
        adv_q:     q-cost advantages
        adv_t:     t-cost advantages
        lambda1:   Lagrange multiplier for question budget
        lambda2:   Lagrange multiplier for turn budget
        normalize: whiten the combined advantages (recommended for stability)

    Returns:
        A_lag: (N,) tensor on device
    """
    A_r = torch.tensor(adv_r, dtype=torch.float32, device=device)
    A_q = torch.tensor(adv_q, dtype=torch.float32, device=device)
    A_t = torch.tensor(adv_t, dtype=torch.float32, device=device)

    A_lag = A_r - lambda1 * A_q - lambda2 * A_t

    if normalize:
        A_lag = (A_lag - A_lag.mean()) / (A_lag.std() + 1e-8)

    return A_lag


def compute_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> Tuple[torch.Tensor, dict]:
    """
    Clipped PPO surrogate loss.

    Args:
        new_log_probs: (N,) log-probs under current policy
        old_log_probs: (N,) log-probs under the policy that collected the rollout
        advantages:    (N,) Lagrangian advantages
        clip_epsilon:  PPO clip range

    Returns:
        loss:  scalar tensor (minimise this)
        info:  dict with diagnostic values
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    loss_unclipped = ratio * advantages
    loss_clipped   = clipped * advantages
    ppo_loss = -torch.min(loss_unclipped, loss_clipped).mean()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean()
        clip_frac = (torch.abs(ratio - 1.0) > clip_epsilon).float().mean()

    info = {
        "ppo_loss":    ppo_loss.item(),
        "approx_kl":   approx_kl.item(),
        "clip_frac":   clip_frac.item(),
        "ratio_mean":  ratio.mean().item(),
    }
    return ppo_loss, info


def compute_value_loss(
    v_r_pred: torch.Tensor,
    v_q_pred: torch.Tensor,
    v_t_pred: torch.Tensor,
    returns_r: List[float],
    returns_q: List[float],
    returns_t: List[float],
    device: torch.device,
) -> Tuple[torch.Tensor, dict]:
    """
    MSE loss for all three value heads.

    Args:
        v_r_pred, v_q_pred, v_t_pred: (N,) predicted values from value heads
        returns_r, returns_q, returns_t: target returns (from GAE)

    Returns:
        value_loss: scalar tensor
        info: dict with per-head losses
    """
    R_r = torch.tensor(returns_r, dtype=torch.float32, device=device)
    R_q = torch.tensor(returns_q, dtype=torch.float32, device=device)
    R_t = torch.tensor(returns_t, dtype=torch.float32, device=device)

    loss_r = F.mse_loss(v_r_pred, R_r)
    loss_q = F.mse_loss(v_q_pred, R_q)
    loss_t = F.mse_loss(v_t_pred, R_t)

    total = loss_r + loss_q + loss_t
    info = {
        "value_loss_r": loss_r.item(),
        "value_loss_q": loss_q.item(),
        "value_loss_t": loss_t.item(),
        "value_loss":   total.item(),
    }
    return total, info


def compute_kl_penalty(
    new_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample KL divergence: KL(π || π_ref) ≈ new_logp - ref_logp.
    Returns mean over the batch.
    """
    return (new_log_probs - ref_log_probs).mean()


def compute_entropy_bonus(
    new_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate entropy as -mean(log_prob).
    This is a lower bound on entropy for token sequences.
    """
    return -new_log_probs.mean()
