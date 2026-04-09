from __future__ import annotations

import random
from collections import defaultdict

import torch


def cluster_embeddings(
    embeddings: torch.Tensor,
    num_clusters: int,
    iterations: int = 25,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D tensor")
    if embeddings.size(0) < num_clusters:
        raise ValueError("Number of embeddings must be >= num_clusters")

    generator = torch.Generator(device=embeddings.device)
    generator.manual_seed(seed)
    initial_indices = torch.randperm(embeddings.size(0), generator=generator, device=embeddings.device)[:num_clusters]
    centroids = embeddings[initial_indices].clone()

    for _ in range(iterations):
        distances = torch.cdist(embeddings, centroids)
        assignments = distances.argmin(dim=1)
        updated_centroids = []
        for cluster_id in range(num_clusters):
            cluster_points = embeddings[assignments == cluster_id]
            if cluster_points.numel() == 0:
                replacement_index = random.randrange(embeddings.size(0))
                updated_centroids.append(embeddings[replacement_index])
            else:
                updated_centroids.append(cluster_points.mean(dim=0))
        centroids = torch.stack(updated_centroids, dim=0)

    final_distances = torch.cdist(embeddings, centroids)
    final_assignments = final_distances.argmin(dim=1)
    return final_assignments, centroids


def find_cluster_prototypes(
    embeddings: torch.Tensor, assignments: torch.Tensor, centroids: torch.Tensor
) -> dict[int, int]:
    prototypes: dict[int, int] = {}
    for cluster_id in range(centroids.size(0)):
        cluster_indices = torch.where(assignments == cluster_id)[0]
        if cluster_indices.numel() == 0:
            continue
        cluster_embeddings = embeddings[cluster_indices]
        centroid = centroids[cluster_id].unsqueeze(0)
        distances = torch.cdist(cluster_embeddings, centroid).squeeze(1)
        prototype_offset = int(distances.argmin().item())
        prototypes[cluster_id] = int(cluster_indices[prototype_offset].item())
    return prototypes


def sample_silhouette_score(
    embeddings: torch.Tensor,
    assignments: torch.Tensor,
    sample_size: int = 512,
    seed: int = 42,
) -> float:
    if embeddings.size(0) <= 1:
        return 0.0

    rng = random.Random(seed)
    indices = list(range(embeddings.size(0)))
    if len(indices) > sample_size:
        indices = rng.sample(indices, sample_size)

    sampled_embeddings = embeddings[indices]
    sampled_assignments = assignments[indices]
    distances = torch.cdist(sampled_embeddings, sampled_embeddings)

    grouped: dict[int, list[int]] = defaultdict(list)
    for offset, cluster_id in enumerate(sampled_assignments.tolist()):
        grouped[int(cluster_id)].append(offset)

    scores: list[float] = []
    for offset, cluster_id in enumerate(sampled_assignments.tolist()):
        same_cluster = [index for index in grouped[int(cluster_id)] if index != offset]
        if same_cluster:
            a_score = float(distances[offset, same_cluster].mean().item())
        else:
            a_score = 0.0

        b_score = float("inf")
        for other_cluster_id, members in grouped.items():
            if other_cluster_id == int(cluster_id):
                continue
            candidate = float(distances[offset, members].mean().item())
            if candidate < b_score:
                b_score = candidate

        if b_score == float("inf") and a_score == 0.0:
            scores.append(0.0)
            continue

        denom = max(a_score, b_score)
        scores.append(0.0 if denom == 0.0 else (b_score - a_score) / denom)

    return sum(scores) / len(scores) if scores else 0.0
