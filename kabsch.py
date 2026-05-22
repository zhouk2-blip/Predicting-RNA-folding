# kabsch.py

import torch


def kabsch_align_single(P, Q, mask):
    """Purpose: Align one predicted structure to one target structure.

    Input:
        P: Predicted coordinates shaped (L, 3).
        Q: Target coordinates shaped (L, 3).
        mask: Valid-residue mask shaped (L,).
    Output:
        Kabsch-aligned predicted coordinates shaped (L, 3).
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

    # rotation with reflection correction, avoiding in-place edits on SVD outputs
    det = torch.det(Vt.T @ U.T)
    correction = torch.ones(3, dtype=H.dtype, device=H.device)
    correction[-1] = torch.sign(det)
    D = torch.diag(correction)
    R = Vt.T @ D @ U.T

    # apply alignment to all P, not just valid residues
    P_aligned = (P - P_mean) @ R + Q_mean

    return P_aligned


def kabsch_align_batch(pred, target, mask):
    """Purpose: Apply Kabsch alignment independently to each batch item.

    Input:
        pred: Predicted coordinates shaped (B, L, 3).
        target: Target coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
    Output:
        Batch of aligned predicted coordinates shaped (B, L, 3).
    """

    aligned = []

    for b in range(pred.shape[0]):
        aligned_b = kabsch_align_single(pred[b], target[b], mask[b])
        aligned.append(aligned_b)

    return torch.stack(aligned, dim=0)
