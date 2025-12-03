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

        # 입력 이름 확인 및 저장
        self.input_names = {inp.name: inp for inp in self.session.get_inputs()}
        print(f"[LaMa] 모델 입력: {list(self.input_names.keys())}")

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """이미지 인페인팅 - 마스크 영역만 크롭하여 고품질 처리"""
        orig_h, orig_w = image.shape[:2]

        # 마스크 영역의 바운딩 박스 찾기
        crop_box = self._get_mask_bbox(mask, padding=50)

        if crop_box is None:
            return image  # 마스크가 비어있으면 원본 반환

        x1, y1, x2, y2 = crop_box
        crop_w, crop_h = x2 - x1, y2 - y1

        # 마스크 영역 크롭
        cropped_image = image[y1:y2, x1:x2].copy()
        cropped_mask = mask[y1:y2, x1:x2].copy()

        print(f"[LaMa] 원본: {orig_w}x{orig_h}, 크롭 영역: {crop_w}x{crop_h}")

        # 전처리 (크롭된 영역만)
        img_input, mask_input, pad_info = self._preprocess(cropped_image, cropped_mask)

        # 추론
        output = self.session.run(None, {
            "image": img_input,
            "mask": mask_input
        })[0]

        # 후처리 (크롭된 크기로 복원)
        inpainted_crop = self._postprocess(output, crop_h, crop_w, pad_info)

        # 원본 이미지에 결과 붙이기
        result = image.copy()
        result[y1:y2, x1:x2] = inpainted_crop

        return result

    def _get_mask_bbox(self, mask: np.ndarray, padding: int = 50) -> Optional[Tuple[int, int, int, int]]:
        """마스크의 바운딩 박스 찾기 (패딩 포함)"""
        h, w = mask.shape[:2]

        # 마스크가 있는 픽셀 좌표 찾기
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return None

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # 패딩 추가 (경계 체크)
        x1 = max(0, x_min - padding)
        y1 = max(0, y_min - padding)
        x2 = min(w, x_max + padding)
        y2 = min(h, y_max + padding)

        # 최소 크기 보장 (너무 작으면 컨텍스트 부족)
        min_size = 256
        if x2 - x1 < min_size:
            center_x = (x1 + x2) // 2
            x1 = max(0, center_x - min_size // 2)
            x2 = min(w, x1 + min_size)
        if y2 - y1 < min_size:
            center_y = (y1 + y2) // 2
            y1 = max(0, center_y - min_size // 2)
            y2 = min(h, y1 + min_size)

        return (x1, y1, x2, y2)

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

    def _postprocess(self, output: np.ndarray, orig_h: int, orig_w: int, pad_info: Tuple) -> np.ndarray:
        """후처리: 패딩 제거, 리사이즈 (LaMa는 완성된 이미지를 출력)"""
        top, left, new_h, new_w = pad_info

        # NCHW -> HWC
        result = output[0].transpose(1, 2, 0)

        # ONNX 모델은 0-255 범위로 출력 (PyTorch와 다름)
        # 안전하게 클리핑
        result = np.clip(result, 0, 255).astype(np.uint8)

        # 패딩 제거 및 원본 크기로
        result = result[top:top + new_h, left:left + new_w]
        result = cv2.resize(result, (orig_w, orig_h))

        # RGB -> BGR
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        return result


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
