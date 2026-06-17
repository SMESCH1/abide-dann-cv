"""
Dataset para clasificación ASD vs TC a nivel sujeto.

Cada sujeto tiene 6 imágenes (2 derivadas × 3 vistas) que se apilan
como un tensor de 6 canales: (6, H, W).
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Orden fijo de los 6 canales
CHANNELS = [
    "falff_axial",
    "falff_coronal",
    "falff_sagital",
    "reho_axial",
    "reho_coronal",
    "reho_sagital",
]

IMG_SIZE = 64  # Redimensionar todas las imágenes a 64×64


def discover_subjects(data_dir: str, meta_csv: str = "abide_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv") -> list[dict]:
    """
    Recorre dataset/ASD y dataset/TC y agrupa las imágenes por sujeto.

    Returns:
        Lista de dicts con keys: 'subject_id', 'label', 'images' (dict canal→path), 'site'
    """
    subjects = []
    class_map = {"TC": 0, "ASD": 1}

    site_map = {}
    if meta_csv and os.path.exists(meta_csv):
        try:
            df = pd.read_csv(meta_csv, low_memory=False)
            if 'SUB_ID' in df.columns and 'SITE_ID' in df.columns:
                site_map = {str(row['SUB_ID']): str(row['SITE_ID']) for _, row in df.iterrows()}
        except Exception as e:
            print(f"Warning: Could not read metadata CSV: {e}")

    for class_name, label in class_map.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Agrupar archivos por subject_id
        grouped: dict[str, dict[str, str]] = defaultdict(dict)
        for fname in os.listdir(class_dir):
            if not fname.endswith(".png"):
                continue
            parts = fname.replace(".png", "").split("_", 1)
            if len(parts) != 2:
                continue
            sub_id, channel = parts[0], parts[1]
            grouped[sub_id][channel] = os.path.join(class_dir, fname)

        # Solo incluir sujetos con las 6 imágenes completas
        for sub_id, images in grouped.items():
            if all(ch in images for ch in CHANNELS):
                subjects.append({
                    "subject_id": sub_id,
                    "label": label,
                    "images": images,
                    "site": site_map.get(sub_id, "UNKNOWN")
                })

    return subjects


class SubjectDataset(Dataset):
    """
    Dataset que retorna las 6 imágenes de un sujeto apiladas como canales.

    Args:
        subjects: Lista de dicts generada por discover_subjects()
        augment: Si True, aplica data augmentation (para training)
    """

    def __init__(self, subjects: list[dict], augment: bool = False):
        self.subjects = subjects
        self.augment = augment

        # Transformaciones base: resize + to tensor + normalize
        self.base_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),  # Convierte a [0,1] y shape (1, H, W)
        ])

        # Augmentation: se aplica al tensor apilado (6, H, W)
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomRotation(degrees=2),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
            ])
        else:
            self.aug_transform = None

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        subject = self.subjects[idx]

        # Cargar y apilar las 6 imágenes como canales
        channel_tensors = []
        for ch_name in CHANNELS:
            img_path = subject["images"][ch_name]
            img = Image.open(img_path).convert("L")  # Escala de grises
            tensor = self.base_transform(img)  # Shape: (1, 64, 64)
            channel_tensors.append(tensor)

        # Apilar: (6, 64, 64). Los PNG ya vienen normalizados por percentiles [1,99]
        # en el downloader (rango [0,1]); ToTensor preserva ese rango. No se aplica
        # z-score por canal para no romper la comparabilidad entre canales ni
        # amplificar canales de baja varianza.
        x = torch.cat(channel_tensors, dim=0)

        # Augmentation (se aplica al tensor completo para mantener coherencia espacial)
        if self.aug_transform is not None:
            x = self.aug_transform(x)

        label = torch.tensor(subject["label"], dtype=torch.float32)
        return x, label, subject["site"]
