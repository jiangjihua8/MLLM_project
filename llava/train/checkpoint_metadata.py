import json
import os


def _unwrap_model(model):
    return getattr(model, "module", model)


def _get_config(model):
    model = _unwrap_model(model)
    return getattr(model, "config", None)


def _get_vision_tower(model):
    model = _unwrap_model(model)
    if hasattr(model, "get_vision_tower"):
        return model.get_vision_tower()
    if hasattr(model, "get_model") and hasattr(model.get_model(), "get_vision_tower"):
        return model.get_model().get_vision_tower()
    if hasattr(model, "model") and hasattr(model.model, "get_vision_tower"):
        return model.model.get_vision_tower()
    return getattr(model, "vision_tower", None)


def _as_plain_list(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return value


def sync_qwen_multimodal_config(model):
    """Persist the actual runtime multimodal setup into model.config."""
    config = _get_config(model)
    if config is None:
        return None

    vision_tower = _get_vision_tower(model)
    if isinstance(vision_tower, list):
        vision_tower = vision_tower[0] if vision_tower else None

    if vision_tower is not None:
        vision_tower_name = getattr(vision_tower, "vision_tower_name", None)
        if vision_tower_name:
            config.mm_vision_tower = vision_tower_name
            config.vision_tower = vision_tower_name

        vision_tower_type = getattr(vision_tower, "mm_vision_tower_type", None)
        if not vision_tower_type:
            class_name = vision_tower.__class__.__name__.lower()
            tower_name = str(getattr(vision_tower, "vision_tower_name", "")).lower()
            if "dinov3" in class_name or "dinov3" in tower_name:
                vision_tower_type = "dinov3"
            elif "dinov2" in class_name or "dinov2" in tower_name:
                vision_tower_type = "dinov2"
            elif "mobileclip" in class_name or "mobileclip" in tower_name:
                vision_tower_type = "mobileclip"
            elif "clip" in class_name or "clip" in tower_name:
                vision_tower_type = "clip"
        if vision_tower_type:
            config.mm_vision_tower_type = vision_tower_type

        input_image_size = getattr(vision_tower, "_target_size", None)
        if input_image_size is None:
            input_image_size = getattr(vision_tower, "input_image_size", None)
        if input_image_size is not None:
            config.input_image_size = input_image_size

        deepstack_indexes = _as_plain_list(getattr(vision_tower, "deepstack_visual_indexes", None))
        config.deepstack_visual_indexes = deepstack_indexes
        config.disable_deepstack = deepstack_indexes is None
    else:
        config.deepstack_visual_indexes = _as_plain_list(getattr(config, "deepstack_visual_indexes", None))
        if getattr(config, "deepstack_visual_indexes", None) is not None:
            config.disable_deepstack = False
        elif not hasattr(config, "disable_deepstack"):
            config.disable_deepstack = True

    return config


def write_qwen_multimodal_checkpoint_metadata(model, output_dir: str, trainer=None):
    if trainer is not None and not trainer.is_world_process_zero():
        return

    config = sync_qwen_multimodal_config(model)
    if config is None:
        return
    if not (getattr(config, "mm_vision_tower", None) or getattr(config, "vision_tower", None)):
        return

    payload = {
        "format": "qwen_multimodal_checkpoint",
        "model_type": getattr(config, "model_type", None),
        "mm_vision_tower": getattr(config, "mm_vision_tower", None),
        "vision_tower": getattr(config, "vision_tower", None),
        "mm_vision_tower_type": getattr(config, "mm_vision_tower_type", None),
        "input_image_size": getattr(config, "input_image_size", None),
        "deepstack_visual_indexes": getattr(config, "deepstack_visual_indexes", None),
        "disable_deepstack": getattr(config, "disable_deepstack", None),
        "bundled_vision_tower": True,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "qwen_multimodal_checkpoint.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
