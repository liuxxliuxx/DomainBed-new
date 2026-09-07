"""Backbone names shared by the runner and training code (no torch import)."""


def normalize_backbone(value):
    name = str(value).strip().lower()
    if name == "vit_b_16":
        name = "vit"
    if name not in ("resnet", "vit"):
        raise ValueError(f"Unknown backbone {value!r}; choose resnet or vit")
    return name


def is_vit(hparams):
    return normalize_backbone(hparams.get("backbone", "resnet")) == "vit"
