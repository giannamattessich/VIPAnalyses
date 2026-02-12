import numpy as np
from utils.stats import zscore_robust


def robust_z(x, eps=1e-8):
    x = np.asarray(x)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + eps
    return (x - med) / (1.4826 * mad)

def softmax(logits, axis=-1):
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def expected_transition_counts(gamma):
    """
    gamma: (T, K) posteriors P(z_t=k)
    returns: (K, K) expected transition counts
    """
    gamma = np.asarray(gamma)
    return gamma[:-1].T @ gamma[1:]

def weak_state_posteriors(td, gamma, delta, motion,
                          q_hi=0.80, q_lo=0.20,
                          strength=3.0):
    """
    Returns gamma_post: (T,4) soft posteriors over [NREM,QW,AW,REM].

    td     : log10(theta/delta) (1D)
    gamma  : gamma power (1D)
    delta  : delta power (1D)
    motion : face motion energy (1D)
    q_hi/q_lo: quantile cutoffs for "high/low" anchors (data-driven)
    strength: larger => more confident anchors
    """
    td_z = robust_z(td)
    ga_z = robust_z(gamma)
    de_z = robust_z(delta)
    mo_z = robust_z(motion)

    # data-driven anchors
    td_hi, td_lo = np.nanquantile(td_z, q_hi), np.nanquantile(td_z, q_lo)
    ga_hi, ga_lo = np.nanquantile(ga_z, q_hi), np.nanquantile(ga_z, q_lo)
    de_hi, de_lo = np.nanquantile(de_z, q_hi), np.nanquantile(de_z, q_lo)
    mo_hi, mo_lo = np.nanquantile(mo_z, q_hi), np.nanquantile(mo_z, q_lo)

    # Score each state with simple interpretable “logit” functions
    # NREM: high delta, low gamma, low motion, low td
    logit_N = (
        + strength * (de_z - de_hi)      # wants delta high
        - 0.8 * strength * (ga_z - ga_lo)
        - 0.6 * strength * (mo_z - mo_lo)
        - 0.4 * strength * (td_z - td_lo)
    )

    # Active Wake: high motion + high gamma + higher td, low delta
    logit_A = (
        + strength * (mo_z - mo_hi)
        + 0.8 * strength * (ga_z - ga_hi)
        + 0.4 * strength * (td_z - td_hi)
        - 0.5 * strength * (de_z - de_lo)
    )

    # REM: high td but LOW motion and LOW gamma and LOW delta
    logit_R = (
        + strength * (td_z - td_hi)
        - 0.9 * strength * (mo_z - mo_lo)
        - 0.6 * strength * (ga_z - ga_lo)
        - 0.6 * strength * (de_z - de_lo)
    )

    # Quiet Wake: “in between” (moderate motion, moderate gamma, low-ish delta)
    # We can define it as a residual: prefer not-extreme scores.
    logit_Q = (
        - 0.3 * strength * np.abs(mo_z)
        - 0.3 * strength * np.abs(ga_z)
        - 0.3 * strength * np.abs(td_z)
        - 0.2 * strength * np.abs(de_z)
    )

    logits = np.vstack([logit_N, logit_Q, logit_A, logit_R]).T
    gamma_post = softmax(logits, axis=1)

    return gamma_post

def estimate_transmat_from_posteriors(gamma_post, alpha_prior=None):
    """
    gamma_post: (T,K)
    alpha_prior: (K,K) Dirichlet pseudocounts (optional)
    returns: transmat (K,K), counts (K,K)
    """
    counts = expected_transition_counts(gamma_post)

    if alpha_prior is None:
        alpha_prior = np.ones_like(counts) * 0.5  # mild smoothing

    trans = counts + alpha_prior
    trans = trans / (trans.sum(axis=1, keepdims=True) + 1e-12)
    return trans, counts

def default_structural_prior(K=4, hub=1, diag=5.0, offdiag=0.2, hub_bonus=0.8):
    """
    K=4 states [NREM,QW,AW,REM]; hub=1 (QW).
    Creates a Dirichlet pseudocount matrix.
    """
    A = np.ones((K, K)) * offdiag
    np.fill_diagonal(A, diag)
    A[:, hub] += hub_bonus        # encourage transitions into hub
    A[hub, :] += hub_bonus * 0.5  # encourage transitions out of hub too
    return A
