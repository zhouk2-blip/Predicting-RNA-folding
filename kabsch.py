# kabsch.py

import torch


def kabsch_align_single(P, Q, mask):
    """
    Align P to Q using Kabsch algorithm.

    P: (L, 3) predicted coordinates
    Q: (L, 3) true coordinates
    mask: (L,) 1 = valid residue, 0 = ignore

    Returns:
        P_aligned: (L, 3)
    """

    valid = mask.bool()

    if valid.sum() < 3:
        return P

    P_valid = P[valid]
    Q_valid = Q[valid]

    # center
    P_mean = P_valid.mean(dim=0, keepdim=True)
    Q_mean = Q_valid.mean(dim=0, keepdim=True)

    P_centered = P_valid - P_mean
    Q_centered = Q_valid - Q_mean

    # covariance
    H = P_centered.T @ Q_centered

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    # rotation
    R = Vt.T @ U.T

    # reflection correction
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # apply alignment to all P, not just valid residues
    P_aligned = (P - P_mean) @ R + Q_mean

    return P_aligned


def kabsch_align_batch(pred, target, mask):
    """
    Batch version of Kabsch alignment.

    pred: (B, L, 3)
    target: (B, L, 3)
    mask: (B, L)

    Returns:
        aligned_pred: (B, L, 3)
    """

    aligned = []

    for b in range(pred.shape[0]):
        aligned_b = kabsch_align_single(pred[b], target[b], mask[b])
        aligned.append(aligned_b)

    return torch.stack(aligned, dim=0)