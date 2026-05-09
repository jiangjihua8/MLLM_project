"""
DINO model configuration registry.

Each entry defines the model specs needed for vision tower setup.
The correct config is selected automatically based on the vision_tower path:
  - 'dinov2-large' → DINOv2-L
  - 'dinov3-vitb16' → DINOv3-B
  - 'dinov3-vitl16' → DINOv3-L

To add a new variant, just add an entry to DINO_CONFIGS.
"""
from dataclasses import dataclass, field
from typing import Optional, List, ClassVar, Dict


@dataclass
class DinoConfig:
    """Configuration for a DINO vision encoder variant."""
    encoder_type: str           # "dinov2" or "dinov3"
    num_layers: int             # number of ViT transformer layers
    hidden_size: int            # feature dimension
    patch_size: int             # patch size (14 for DINOv2, 16 for DINOv3)
    image_size: int             # default input image size
    num_register_tokens: int    # 0 for DINOv2, 4 for DINOv3
    deepstack_visual_indexes: List[int]  # which layers to use for DeepStack

    @property
    def skip_tokens(self):
        """CLS + register tokens to skip when extracting patch features."""
        return 1 + self.num_register_tokens

    @property
    def num_patches_per_side(self):
        return self.image_size // self.patch_size

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2


# ====================== Registry ======================

DINO_CONFIGS: Dict[str, DinoConfig] = {
    # ------------- DINOv2 -------------
    "dinov2-large": DinoConfig(
        encoder_type="dinov2",
        num_layers=24,
        hidden_size=1024,
        patch_size=14,
        image_size=518,
        num_register_tokens=0,
        deepstack_visual_indexes=[6, 12, 18, 23],
    ),

    # ------------- DINOv3 ViT -------------
    "dinov3-vits16": DinoConfig(
        encoder_type="dinov3",
        num_layers=12,
        hidden_size=384,
        patch_size=16,
        image_size=224,
        num_register_tokens=4,
        deepstack_visual_indexes=[3, 6, 9, 11],
    ),
    "dinov3-vitb16": DinoConfig(
        encoder_type="dinov3",
        num_layers=12,
        hidden_size=768,
        patch_size=16,
        image_size=224,
        num_register_tokens=4,
        deepstack_visual_indexes=[3, 6, 9, 11],
    ),
    "dinov3-vitl16": DinoConfig(
        encoder_type="dinov3",
        num_layers=24,
        hidden_size=1024,
        patch_size=16,
        image_size=224,
        num_register_tokens=4,
        deepstack_visual_indexes=[6, 12, 18, 23],
    ),
    "dinov3-vith16plus": DinoConfig(
        encoder_type="dinov3",
        num_layers=24,
        hidden_size=1280,
        patch_size=16,
        image_size=224,
        num_register_tokens=4,
        deepstack_visual_indexes=[6, 12, 18, 23],
    ),
}


def get_dino_config(vision_tower_path: str, input_image_size: Optional[int] = None) -> DinoConfig:
    """
    Auto-detect DINO variant from vision_tower path and return its config.
    If input_image_size is given, it overrides the default image_size.

    Detection order:
      1. Exact key match in DINO_CONFIGS
      2. Substring match (e.g. 'dinov3-vitl16' contained in path)
    """
    path_lower = vision_tower_path.lower()

    for key, cfg in DINO_CONFIGS.items():
        if key in path_lower:
            cfg = DinoConfig(**{**cfg.__dict__, "image_size": input_image_size or cfg.image_size})
            return cfg

    raise KeyError(f"Cannot determine DINO variant from path: {vision_tower_path}. "
                   f"Known keys: {list(DINO_CONFIGS.keys())}")
