"""Open-vocabulary object localization: GroundingDINO boxes -> SAM2 masks -> PCA.

GroundingDINO finds objects from a text prompt, SAM2 turns each box into a
pixel-accurate mask, and the mask is fitted with PCA to recover the top face's
centroid, orientation and side lengths.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from src.vision.rect_face_pca import (
    RectFace,
    extract_top_face,
    fit_rect_face,
    rectify_homography,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GD_CONFIG = REPO_ROOT / "weights" / "GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = REPO_ROOT / "weights" / "groundingdino_swint_ogc.pth"
SAM2_CHECKPOINTS = {
    "small": (REPO_ROOT / "weights" / "sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "large": (REPO_ROOT / "weights" / "sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}

_deform_attn_patched = False
_bert_attention_mask_patched = False


def patch_deformable_attention() -> None:
    """Route GroundingDINO's deformable attention through its PyTorch fallback.

    The op always calls its compiled CUDA extension when tensors are on GPU,
    even when that extension failed to build (no CUDA toolkit to compile it) --
    it never falls back on its own.
    """
    global _deform_attn_patched
    if _deform_attn_patched:
        return
    import groundingdino.models.GroundingDINO.ms_deform_attn as msda

    class _PureTorchDeformAttn:
        @staticmethod
        def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
            del level_start_index, im2col_step
            return msda.multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )

    msda.MultiScaleDeformableAttnFunction = _PureTorchDeformAttn
    _deform_attn_patched = True


def patch_bert_attention_mask() -> None:
    """Restores three ``ModuleUtilsMixin`` methods ``transformers`` dropped by 5.x.

    GroundingDINO's ``BertModelWarper`` (``groundingdino/models/GroundingDINO/bertwarper.py``)
    grabs ``get_extended_attention_mask``, ``invert_attention_mask`` and ``get_head_mask`` off the
    BERT model at construction time. GroundingDINO was written against an older ``transformers``
    API that had these on every model; building the model otherwise raises ``AttributeError``.
    The implementations below are the standard ones from older ``transformers`` releases.
    """
    global _bert_attention_mask_patched
    if _bert_attention_mask_patched:
        return
    from transformers import PreTrainedModel

    if all(hasattr(PreTrainedModel, n) for n in ("get_extended_attention_mask", "invert_attention_mask", "get_head_mask")):
        _bert_attention_mask_patched = True
        return

    def _model_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
        del input_shape, device
        dtype = dtype if dtype is not None else _model_dtype(self)
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"attention_mask with shape {attention_mask.shape} is not 2D or 3D")
        extended = extended.to(dtype=dtype)
        return (1.0 - extended) * torch.finfo(dtype).min

    def invert_attention_mask(self, encoder_attention_mask):
        dtype = _model_dtype(self)
        if encoder_attention_mask.dim() == 3:
            extended = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            extended = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(f"encoder_attention_mask with shape {encoder_attention_mask.shape} is not 2D or 3D")
        extended = extended.to(dtype=dtype)
        return (1.0 - extended) * torch.finfo(dtype).min

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is None:
            return [None] * num_hidden_layers
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.to(dtype=_model_dtype(self))
        return head_mask.unsqueeze(-1) if is_attention_chunked else head_mask

    if not hasattr(PreTrainedModel, "get_extended_attention_mask"):
        PreTrainedModel.get_extended_attention_mask = get_extended_attention_mask
    if not hasattr(PreTrainedModel, "invert_attention_mask"):
        PreTrainedModel.invert_attention_mask = invert_attention_mask
    if not hasattr(PreTrainedModel, "get_head_mask"):
        PreTrainedModel.get_head_mask = get_head_mask
    _bert_attention_mask_patched = True


@dataclass
class Detection:
    """One localized object."""

    label: str
    score: float
    box_xyxy: NDArray[np.float64]  # (4,) pixel coords
    mask: NDArray[np.bool_] | None  # (H, W) top face if isolated, else whole object
    face: RectFace | None  # PCA fit of the mask, None if segmentation failed
    height_px: float = 0.0  # projected box height recovered from the silhouette


class ObjectLocalizer:
    def __init__(
        self,
        sam2_size: str = "small",
        device: str = "cuda",
        infer_size: int = 480,
        half: bool = True,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        tilt_deg: float = 20.0,
        focal_px: float | None = None,
        detect_every: int = 1,
    ) -> None:
        patch_deformable_attention()
        patch_bert_attention_mask()
        from groundingdino.util.inference import load_model
        import groundingdino.datasets.transforms as T
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device if torch.cuda.is_available() else "cpu"
        self.half = half and self.device == "cuda"
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.tilt_deg = tilt_deg
        self.focal_px = focal_px
        self.detect_every = max(1, detect_every)
        self._homography: NDArray[np.float64] | None = None
        self._homography_shape: tuple[int, int] | None = None
        self._frame_count = 0
        self._cached_boxes: NDArray[np.float64] | None = None
        self._cached_labels: list[tuple[str, float]] = []

        # load_model() leaves the model on CPU and predict() would otherwise
        # move all 172M params across on its first call mid-loop.
        self.gd = load_model(str(GD_CONFIG), str(GD_WEIGHTS)).to(self.device)
        self.gd_transform = T.Compose([
            T.RandomResize([infer_size], max_size=int(infer_size * 4 / 3)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        ckpt, cfg = SAM2_CHECKPOINTS[sam2_size]
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing SAM2 checkpoint: {ckpt}")
        self.sam2 = SAM2ImagePredictor(build_sam2(cfg, str(ckpt), device=self.device))

    def _detect(self, image_rgb: NDArray[np.uint8], prompt: str):
        from groundingdino.util.inference import predict

        tensor, _ = self.gd_transform(Image.fromarray(image_rgb), None)
        kwargs = dict(
            model=self.gd, image=tensor, caption=prompt,
            box_threshold=self.box_threshold, text_threshold=self.text_threshold,
        )
        if self.half:
            with torch.autocast("cuda", dtype=torch.float16):
                return predict(**kwargs)
        return predict(**kwargs)

    def homography_for(self, width: int, height: int) -> NDArray[np.float64] | None:
        """Cached rectifying homography for this frame size, or None when tilt is 0."""
        if self.tilt_deg == 0.0:
            return None
        if self._homography_shape != (width, height):
            self._homography = rectify_homography(self.tilt_deg, width, height, self.focal_px)
            self._homography_shape = (width, height)
        return self._homography

    def _boxes_for(self, image_rgb: NDArray[np.uint8], prompt: str, w: int, h: int):
        """GroundingDINO boxes, re-detecting only every `detect_every` frames."""
        due = self._frame_count % self.detect_every == 0 or self._cached_boxes is None
        self._frame_count += 1
        if not due:
            return self._cached_boxes, self._cached_labels

        boxes_cxcywh, logits, phrases = self._detect(image_rgb, prompt)
        if len(boxes_cxcywh) == 0:
            self._cached_boxes, self._cached_labels = np.empty((0, 4)), []
            return self._cached_boxes, self._cached_labels

        # GroundingDINO returns normalized cxcywh; SAM2 wants absolute xyxy.
        boxes = boxes_cxcywh.numpy() * np.array([w, h, w, h])
        xyxy = np.column_stack([
            boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2,
        ])
        self._cached_boxes = xyxy
        self._cached_labels = [(p, float(s)) for p, s in zip(phrases, logits)]
        return self._cached_boxes, self._cached_labels

    def _segment(self, image_rgb: NDArray[np.uint8], xyxy: NDArray[np.float64]):
        """Whole-object SAM2 masks, one per box.

        SAM2 is asked for the whole object, which is what it is trained to do;
        isolating the top face is left to geometry in `locate`.
        """
        h, w = image_rgb.shape[:2]
        # SAM2's image encoder dominates this pipeline; bf16 autocast cuts it
        # by ~40% with no visible change to the masks.
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if self.half
                    else contextlib.nullcontext())
        with autocast, torch.inference_mode():
            self.sam2.set_image(image_rgb)
            masks, _scores, _ = self.sam2.predict(box=xyxy, multimask_output=False)
        return list(np.asarray(masks).reshape(-1, h, w).astype(bool))

    def locate(
        self,
        frame_bgr: NDArray[np.uint8],
        prompt: str,
        segment: bool = True,
        top_face_only: bool = True,
    ) -> list[Detection]:
        """Detect objects matching `prompt`, segment them, and fit their top faces."""
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        xyxy, labels = self._boxes_for(image_rgb, prompt, w, h)
        if len(xyxy) == 0:
            return []

        masks = self._segment(image_rgb, xyxy) if segment else None
        homography = self.homography_for(w, h)

        detections = []
        for i, (phrase, score) in enumerate(labels):
            mask = masks[i] if masks is not None and i < len(masks) else None
            height_px = 0.0
            if mask is not None and top_face_only:
                mask, height_px = extract_top_face(mask, homography)
            detections.append(Detection(
                label=phrase,
                score=score,
                box_xyxy=xyxy[i],
                mask=mask,
                face=fit_rect_face(mask, homography) if mask is not None else None,
                height_px=height_px,
            ))
        return detections
