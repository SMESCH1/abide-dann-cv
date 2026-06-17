"""
Modelo CNN para clasificación binaria ASD vs TC.

Recibe un tensor de 6 canales (6 imágenes del sujeto apiladas)
y produce una probabilidad de ser ASD (clase 1).
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha):
    return GradientReversalLayer.apply(x, alpha)

class ASD_DANN(nn.Module):
    """
    CNN liviana con Domain Adversarial Neural Network (DANN).

    Arquitectura:
        Feature Extractor compartida.
        Rama 1 (Clasificador): Predice ASD vs TC.
        Rama 2 (Dominio): GRL -> Predice sitio de origen (resonador).
    """

    def __init__(self, in_channels: int = 6, num_sites: int = 2, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Bloque 1: 6 → 16
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Bloque 2: 16 → 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Bloque 3: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Global Average Pooling (drástica reducción de parámetros)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Rama 1: Clasificador de Tarea (ASD vs TC)
        self.class_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Rama 2: Clasificador de Dominio (Resonadores)
        self.domain_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_sites),
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor de shape (batch, 6, 64, 64)
            alpha: Peso para el Gradient Reversal Layer

        Returns:
            class_logits: Logits (batch, 1) para clasificación binaria
            domain_logits: Logits (batch, num_sites) para clasificación de dominios
        """
        f = self.features(x)
        
        # Rama de clase
        class_logits = self.class_classifier(f)
        
        # Rama de dominio con GRL
        f_reversed = grad_reverse(f, alpha)
        domain_logits = self.domain_classifier(f_reversed)
        
        return class_logits, domain_logits


def count_parameters(model: nn.Module) -> int:
    """Cuenta los parámetros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test rápido
    model = ASD_DANN(num_sites=10)
    print(f"Parámetros entrenables: {count_parameters(model):,}")
    dummy = torch.randn(4, 6, 64, 64)
    class_out, domain_out = model(dummy, alpha=0.5)
    print(f"Input shape: {dummy.shape}")
    print(f"Class Output shape: {class_out.shape}")
    print(f"Domain Output shape: {domain_out.shape}")
