from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def MSE_loss_score(scores_0, scores_1, use_softmax: bool = False):
    if use_softmax:
        scores_0 = F.softmax(scores_0, dim=1)
        scores_1 = F.softmax(scores_1, dim=1)
    return F.mse_loss(scores_0, scores_1)


def KL_loss(scores_0, scores_1):
    scores_0 = scores_0.view(-1, scores_0.shape[-1])
    scores_1 = scores_1.view(-1, scores_1.shape[-1])
    return F.kl_div(
        F.log_softmax(scores_0, 1),
        F.log_softmax(scores_1, 1),
        log_target=True,
        reduction="batchmean",
    )


def cosine_similarity_loss(scanner_0_embeddings, scanner_1_embeddings):
    scanner_0_embeddings = scanner_0_embeddings.view(-1, scanner_0_embeddings.shape[-1])
    scanner_1_embeddings = scanner_1_embeddings.view(-1, scanner_1_embeddings.shape[-1])
    return 1 - F.cosine_similarity(scanner_0_embeddings, scanner_1_embeddings).mean()


def asymmetric_contrastive_loss(
    scanner_0_embeddings: torch.Tensor,
    scanner_1_embeddings: torch.Tensor,
    temperature: float,
    num_negative_samples: int,
    contrastive_diff_patient_negative: bool,
    max_tiles_per_scan: Optional[int] = None,
):
    batch_size, imgs_per_patient, emb_dim = scanner_0_embeddings.shape
    device = scanner_0_embeddings.device

    # Subsample tiles if max_tiles_per_scan is set
    if max_tiles_per_scan is not None and imgs_per_patient > max_tiles_per_scan:
        indices = torch.randperm(imgs_per_patient, device=device)[:max_tiles_per_scan]
        scanner_0_embeddings = scanner_0_embeddings[:, indices, :]
        scanner_1_embeddings = scanner_1_embeddings[:, indices, :]
        imgs_per_patient = max_tiles_per_scan

    total_images = batch_size * imgs_per_patient

    # 1. Flatten embeddings
    embedding_a = scanner_0_embeddings.view(total_images, emb_dim)
    embedding_p = scanner_1_embeddings.view(total_images, emb_dim)

    # Detect valid (non-zero) embeddings - these are real tiles, not padding
    valid_a = embedding_a.abs().sum(dim=1) > 0  # (total_images,)
    valid_p = embedding_p.abs().sum(dim=1) > 0  # (total_images,)
    valid_tiles = valid_a & valid_p  # Both must be valid for a valid pair

    # If no valid tiles, return zero loss
    if valid_tiles.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Filter to only valid tiles
    valid_indices = valid_tiles.nonzero(as_tuple=True)[0]
    embedding_a = embedding_a[valid_indices]
    embedding_p = embedding_p[valid_indices]
    num_valid = len(valid_indices)

    # Normalize only valid embeddings (avoids NaN from zero vectors)
    embedding_a = F.normalize(embedding_a, p=2.0, dim=1)
    embedding_p = F.normalize(embedding_p, p=2.0, dim=1)

    # 2. Define the Masks
    # Recompute patient indices for filtered tiles
    patient_indices_full = torch.arange(batch_size, device=device).repeat_interleave(imgs_per_patient)
    patient_indices = patient_indices_full[valid_indices]

    # identity_mask[i, j] is True if i == j (the exact same tile/index)
    identity_mask = torch.eye(num_valid, device=device).bool()
    same_patient_mask = patient_indices.unsqueeze(1) == patient_indices.unsqueeze(0)

    if contrastive_diff_patient_negative:
        # Negatives are images from DIFFERENT patients
        valid_negative_mask = ~same_patient_mask
    else:
        # Negatives are DIFFERENT images from the SAME patient
        # We take everyone in the same patient group but exclude the self-identity
        valid_negative_mask = same_patient_mask & (~identity_mask)

    # 3. Handle edge cases where no negatives exist
    actual_possible_negatives = valid_negative_mask.sum(dim=1).min().item()
    if actual_possible_negatives == 0:
        # This happens if batch_size=1 and imgs_per_patient=1 in "same patient" mode
        return torch.tensor(0.0, device=device, requires_grad=True)

    k = min(num_negative_samples, int(actual_possible_negatives))

    # 4. Select Negatives
    # We use a large negative value to ignore invalid pairs during topk
    random_scores = torch.rand(num_valid, num_valid, device=device)
    random_scores = random_scores.masked_fill(~valid_negative_mask, float("-inf"))

    negative_indices = torch.topk(random_scores, k=k, dim=1).indices

    # 5. Calculate Similarities
    # Positive pair: Image i from scanner_0 vs Image i from scanner_1
    pos_sim = torch.sum(embedding_a * embedding_p, dim=1, keepdim=True)  # (num_valid, 1)

    # Negative pairs: Image i from scanner_0 vs k selected images from scanner_0
    negative_embeddings = embedding_a[negative_indices]  # (num_valid, k, emb_dim)
    embedding_a_expanded = embedding_a.unsqueeze(1)

    neg_sim = torch.bmm(
        embedding_a_expanded, negative_embeddings.transpose(1, 2)
    ).squeeze(1)  # (num_valid, k)

    # 6. Loss Calculation
    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
    labels = torch.zeros(num_valid, dtype=torch.long, device=device)

    return F.cross_entropy(logits, labels)


def contrastive_loss(
    scanner_0_embeddings: torch.Tensor,
    scanner_1_embeddings: torch.Tensor,
    temperature: float,
    num_negative_samples: int,
    contrastive_diff_patient_negative: bool,
    max_tiles_per_scan: Optional[int] = None,
):
    # scanner_0 → scanner_1
    loss_01 = asymmetric_contrastive_loss(
        scanner_0_embeddings,
        scanner_1_embeddings,
        temperature=temperature,
        num_negative_samples=num_negative_samples,
        contrastive_diff_patient_negative=contrastive_diff_patient_negative,
        max_tiles_per_scan=max_tiles_per_scan,
    )

    # scanner_1 → scanner_0
    loss_10 = asymmetric_contrastive_loss(
        scanner_1_embeddings,
        scanner_0_embeddings,
        temperature=temperature,
        num_negative_samples=num_negative_samples,
        contrastive_diff_patient_negative=contrastive_diff_patient_negative,
        max_tiles_per_scan=max_tiles_per_scan,
    )

    # Average
    return 0.5 * (loss_01 + loss_10)