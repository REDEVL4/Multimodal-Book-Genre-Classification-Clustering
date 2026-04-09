from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def _build_backbone(backbone_name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, in_features

    if backbone_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return backbone, in_features

    raise ValueError(f"Unsupported backbone: {backbone_name}")


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text_ids)
        encoded, _ = self.encoder(embedded)
        mask = (text_ids != 0).unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.projection(self.dropout(pooled))


class MultimodalGenreModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        backbone_name: str = "efficientnet_b0",
        pretrained_backbone: bool = False,
        text_hidden_dim: int = 128,
        projection_dim: int = 256,
        dropout: float = 0.3,
        use_text: bool = True,
    ) -> None:
        super().__init__()
        self.use_text = use_text
        self.backbone_name = backbone_name
        self.pretrained_backbone = pretrained_backbone
        self.text_hidden_dim = text_hidden_dim
        self.projection_dim = projection_dim
        self.dropout_rate = dropout

        self.image_backbone, image_feature_dim = _build_backbone(backbone_name, pretrained_backbone)
        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        text_feature_dim = text_hidden_dim * 2
        if use_text:
            self.text_encoder = TextEncoder(vocab_size=vocab_size, hidden_dim=text_hidden_dim, dropout=dropout)
            self.text_projection = nn.Sequential(
                nn.Linear(text_feature_dim, projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.modality_gate = nn.Linear(projection_dim * 2, projection_dim)
        else:
            self.text_encoder = None
            self.text_projection = None
            self.modality_gate = None

        classifier_input_dim = projection_dim if not use_text else projection_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, num_classes),
        )

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        image_features = self.image_projection(self.image_backbone(images))
        if not self.use_text:
            logits = self.classifier(image_features)
            gate = torch.ones((images.size(0), 1), device=image_features.device, dtype=image_features.dtype)
            return {"logits": logits, "embeddings": image_features, "gate": gate}

        if text_ids is None:
            raise ValueError("text_ids must be provided when use_text=True")

        text_features = self.text_projection(self.text_encoder(text_ids))
        gate = torch.sigmoid(self.modality_gate(torch.cat([image_features, text_features], dim=1)))
        fused_features = gate * image_features + (1.0 - gate) * text_features
        classifier_input = torch.cat([image_features, text_features, fused_features], dim=1)
        logits = self.classifier(classifier_input)
        return {"logits": logits, "embeddings": fused_features, "gate": gate.mean(dim=1, keepdim=True)}
