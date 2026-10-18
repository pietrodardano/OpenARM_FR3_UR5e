"""Open-vocabulary localization: GroundingDINO boxes -> SAM2 masks -> top-face centroid (pixels).

Needs the ``vision`` dependency group and ``weights/`` (NFR3: imported only when built). Weights:
GroundingDINO SwinT-OGC and SAM2.1 hiera small / large (see README).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from ..config import REPO_ROOT, VisionConfig
from .top_face import extract_top_face, face_centroid, rectify_homography

WEIGHTS = REPO_ROOT / "weights"
GD_CONFIG = WEIGHTS / "GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = WEIGHTS / "groundingdino_swint_ogc.pth"
SAM2 = {"small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
        "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml")}


@dataclass
class Hit:
    """One localized object in one frame."""

    label: str
    score: float
    box_xyxy: NDArray  # (4,) px
    centroid: Optional[Tuple[float, float]]  # top-face centre, px (None: segmentation failed)
    mask: Optional[NDArray] = None


def _patch_deformable_attention() -> None:
    """Routes GroundingDINO's deformable attention to its pure-PyTorch op.

    Its CUDA extension is not built here (no nvcc), and the op never falls back on its own.
    """
    import groundingdino.models.GroundingDINO.ms_deform_attn as msda

    class _PureTorch:
        @staticmethod
        def apply(value, spatial_shapes, _level_start_index, sampling_locations, attention_weights, _im2col_step):
            return msda.multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations,
                                                            attention_weights)

    msda.MultiScaleDeformableAttnFunction = _PureTorch


class Localizer:
    def __init__(self, cfg: VisionConfig, focal_px: Optional[float] = None) -> None:
        import torch
        import groundingdino.datasets.transforms as T
        from groundingdino.util.inference import load_model
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        _patch_deformable_attention()
        self.torch = torch
        self.cfg = cfg
        self.focal_px = focal_px
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        self.half = self.device == "cuda"
        ckpt, sam_cfg = SAM2[cfg.sam2_size]
        for f in (GD_CONFIG, GD_WEIGHTS, WEIGHTS / ckpt):
            if not f.exists():
                raise FileNotFoundError(f"missing model weights {f} (see README)")
        self.gd = load_model(str(GD_CONFIG), str(GD_WEIGHTS)).to(self.device)
        self.transform = T.Compose([T.RandomResize([cfg.infer_size], max_size=int(cfg.infer_size * 4 / 3)),
                                    T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.sam2 = SAM2ImagePredictor(build_sam2(sam_cfg, str(WEIGHTS / ckpt), device=self.device))

    def _boxes(self, rgb: NDArray, prompt: str) -> Tuple[NDArray, List[Tuple[str, float]]]:
        from groundingdino.util.inference import predict
        from PIL import Image

        tensor, _ = self.transform(Image.fromarray(rgb), None)
        ctx = self.torch.autocast("cuda", dtype=self.torch.float16) if self.half else contextlib.nullcontext()
        with ctx:
            boxes, logits, phrases = predict(model=self.gd, image=tensor, caption=prompt,
                                             box_threshold=self.cfg.box_threshold,
                                             text_threshold=self.cfg.text_threshold, device=self.device)
        h, w = rgb.shape[:2]
        b = boxes.numpy() * np.array([w, h, w, h])  # normalized cxcywh -> px xyxy
        xyxy = np.column_stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2])
        return xyxy, [(p, float(s)) for p, s in zip(phrases, logits)]

    def _masks(self, rgb: NDArray, xyxy: NDArray) -> List[NDArray]:
        h, w = rgb.shape[:2]
        ctx = self.torch.autocast("cuda", dtype=self.torch.bfloat16) if self.half else contextlib.nullcontext()
        with ctx, self.torch.inference_mode():
            self.sam2.set_image(rgb)
            masks, _, _ = self.sam2.predict(box=xyxy, multimask_output=False)
        return list(np.asarray(masks).reshape(-1, h, w).astype(bool))

    def locate(self, frame_bgr: NDArray, prompt: str) -> List[Hit]:
        """Objects matching ``prompt``: box, score, top-face centroid (px)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        xyxy, labels = self._boxes(rgb, prompt)
        if len(xyxy) == 0:
            return []
        h, w = rgb.shape[:2]
        H = rectify_homography(self.cfg.tilt_deg, w, h, self.focal_px or float(np.hypot(w, h)) / 2.0)
        hits = []
        for (label, score), box, mask in zip(labels, xyxy, self._masks(rgb, xyxy)):
            top = extract_top_face(mask, H)
            hits.append(Hit(label=label, score=score, box_xyxy=box, centroid=face_centroid(top, H), mask=top))
        return hits
