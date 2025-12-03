"""
LaMa (Large Mask Inpainting) ONNX 추론 모듈 - 단순화 버전

모델 다운로드:
    https://huggingface.co/Carve/LaMa-ONNX
    models/lama/lama_fp32.onnx
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class LamaInpainter:
    """LaMa ONNX 인페인터 (단순화 버전)"""

    MODEL_SIZE = 512

    def __init__(self, model_path: Path):
        if ort is None:
            raise ImportError("onnxruntime 설치 필요: pip install onnxruntime")

        providers = ort.get_available_providers()
        self.session = ort.InferenceSession(str(model_path), providers=providers)

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """이미지 인페인팅"""
        orig_h, orig_w = image.shape[:2]

        # 전처리
        img_input, mask_input, pad_info = self._preprocess(image, mask)

        # 추론
        output = self.session.run(None, {
            self.session.get_inputs()[0].name: img_input,
            self.session.get_inputs()[1].name: mask_input
        })[0]

        # 후처리
        result = self._postprocess(output, image, mask, (orig_h, orig_w), pad_info)
        return result

    def _preprocess(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple]:
        """전처리: 리사이즈, 패딩, 정규화"""
        h, w = image.shape[:2]
        scale = self.MODEL_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # 리사이즈
        img_resized = cv2.resize(image, (new_w, new_h))
        mask_resized = cv2.resize(mask, (new_w, new_h))

        # 패딩
        pad_h, pad_w = self.MODEL_SIZE - new_h, self.MODEL_SIZE - new_w
        top, left = pad_h // 2, pad_w // 2

        img_padded = np.pad(img_resized, ((top, pad_h - top), (left, pad_w - left), (0, 0)), mode='reflect')
        mask_padded = np.pad(mask_resized, ((top, pad_h - top), (left, pad_w - left)), mode='reflect')

        # BGR -> RGB, 정규화, NCHW
        img_input = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))[np.newaxis, ...]

        # 마스크 정규화, N1HW (1 = 인페인팅할 영역)
        mask_input = (mask_padded.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, ...]
        # 마스크 이진화 (0 또는 1로 명확하게)
        mask_input = (mask_input > 0.5).astype(np.float32)

        return img_input, mask_input, (top, left, new_h, new_w)

    def _postprocess(self, output: np.ndarray, orig_image: np.ndarray,
                     mask: np.ndarray, orig_size: Tuple, pad_info: Tuple) -> np.ndarray:
        """후처리: 패딩 제거, 리사이즈, 블렌딩"""
        top, left, new_h, new_w = pad_info
        orig_h, orig_w = orig_size

        # NCHW -> HWC
        result = output[0].transpose(1, 2, 0)

        # 출력 범위 자동 감지: [0,1] 범위면 255 곱하기, 이미 [0,255] 범위면 그대로
        if result.max() <= 1.0:
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
        else:
            result = np.clip(result, 0, 255).astype(np.uint8)

        # 패딩 제거 및 원본 크기로
        result = result[top:top + new_h, left:left + new_w]
        result = cv2.resize(result, (orig_w, orig_h))

        # RGB -> BGR
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # 마스크 영역만 블렌딩
        blend = mask.astype(np.float32) / 255.0
        blend = cv2.GaussianBlur(blend, (5, 5), 0)
        blend = np.clip(blend, 0, 1)
        blend = blend[:, :, np.newaxis]  # (H, W) -> (H, W, 1)
        output_img = orig_image * (1 - blend) + result * blend

        return output_img.astype(np.uint8)


def load_lama_model(model_dir: Path) -> Optional[LamaInpainter]:
    """LaMa 모델 로더"""
    paths = [
        model_dir / "lama" / "lama_fp32.onnx",
        model_dir / "lama" / "lama.onnx",
        model_dir / "lama_fp32.onnx",
    ]
    for path in paths:
        if path.exists():
            return LamaInpainter(path)
    return None
