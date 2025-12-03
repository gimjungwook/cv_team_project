# 코드 상세 설명서 (Detailed Code Explanation)

객체 탐지 및 제거 애플리케이션의 모든 코드를 상세하게 설명합니다.

---

# 1. app.py - 메인 애플리케이션

## 1.1 Import 및 의존성

```python
import sys
from pathlib import Path
from typing import List, Tuple, Optional
```
- `sys`: 명령줄 인자 처리 (`sys.argv`)
- `pathlib.Path`: 파일 경로를 객체지향적으로 다룸
- `typing`: 타입 힌트 (코드 가독성 향상)

```python
import cv2
import numpy as np
```
- `cv2`: OpenCV 라이브러리 (이미지 처리, GUI)
- `numpy`: 수치 연산, 배열 처리

```python
from core.detect import load_default_detector, YOLODetector, DetectionResult
from core.lama import load_lama_model, LamaInpainter
from core.inpaint import telea_inpaint
```
- 커스텀 모듈에서 필요한 클래스/함수 import

---

## 1.2 ObjectEraser 클래스 - 초기화

```python
class ObjectEraser:
    """OpenCV 기반 객체 제거 애플리케이션"""

    MAX_WINDOW_WIDTH = 1600   # 최대 윈도우 너비
    MAX_WINDOW_HEIGHT = 900   # 최대 윈도우 높이
```
- **클래스 변수**: 모든 인스턴스가 공유하는 상수
- 윈도우 크기를 제한하여 큰 이미지도 화면에 맞게 표시

```python
    def __init__(self, image_path: str):
        self.original = cv2.imread(image_path)
```
- `cv2.imread()`: 이미지 파일을 BGR 형식의 numpy 배열로 로드
- `self.original`: 원본 이미지 (절대 변경하지 않음)

```python
        if self.original is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
```
- 파일이 없거나 읽을 수 없으면 예외 발생

```python
        self.result = self.original.copy()
```
- `copy()`: 원본의 복사본 생성 (독립적인 메모리)
- `self.result`: 인페인팅 결과가 저장될 이미지

```python
        self.mode = "telea"  # "telea", "telea_cv", "lama"
```
- 현재 선택된 인페인팅 모드
- `telea`: 직접 구현한 Telea 알고리즘
- `telea_cv`: OpenCV 내장 함수
- `lama`: 딥러닝 기반 LaMa 모델

```python
        self.detections: List[DetectionResult] = []
        self.rois: List[Tuple[int, int, int, int]] = []
```
- `detections`: YOLO가 탐지한 객체 목록
- `rois`: 사용자가 수동으로 선택한 영역 (x, y, width, height)

```python
        self.detector: Optional[YOLODetector] = None
        self.lama: Optional[LamaInpainter] = None
```
- **지연 로딩 (Lazy Loading)**: 처음에는 None
- 실제로 필요할 때만 모델을 로드 (메모리 절약)

```python
        self.window_name = "Object Eraser"
```
- OpenCV 창의 제목

---

## 1.3 메인 루프 (run 메서드)

```python
    def run(self) -> None:
        """메인 루프"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
```
- `namedWindow()`: 이름이 있는 창 생성
- `WINDOW_NORMAL`: 창 크기 조절 가능

```python
        cv2.resizeWindow(self.window_name, self.MAX_WINDOW_WIDTH, self.MAX_WINDOW_HEIGHT)
```
- 창 크기를 지정된 최대 크기로 설정

```python
        while True:
            self._show_image()
            key = cv2.waitKey(0) & 0xFF
```
- **무한 루프**: 사용자가 종료할 때까지 실행
- `_show_image()`: 현재 상태를 화면에 표시
- `waitKey(0)`: 키 입력을 무한 대기 (0 = 무한)
- `& 0xFF`: 하위 8비트만 추출 (키 코드)

```python
            if key == ord('q') or key == 27:  # q or ESC
                break
```
- `ord('q')`: 문자 'q'의 ASCII 코드 (113)
- `27`: ESC 키의 ASCII 코드
- `break`: 루프 종료

```python
            elif key == ord('d'):
                self._detect()
```
- 'd' 키: YOLO 객체 탐지 실행

```python
            elif key == ord('r'):
                self._add_roi()
```
- 'r' 키: 수동 ROI 추가

```python
            elif key == ord('e'):
                self._erase()
```
- 'e' 키: 선택된 영역 제거 (인페인팅)

```python
            elif key == ord('t'):
                self.mode = "telea"
                print("모드 변경: Telea 직접 구현 → 'e' 키로 제거 실행")
```
- 't' 키: Telea 직접 구현 모드로 변경

```python
            elif key == ord('c'):
                self.mode = "telea_cv"
                print("모드 변경: cv2.inpaint() → 'e' 키로 제거 실행")
```
- 'c' 키: OpenCV 내장 인페인팅 모드

```python
            elif key == ord('l'):
                self.mode = "lama"
                print("모드 변경: LaMa → 'e' 키로 제거 실행")
```
- 'l' 키: LaMa 딥러닝 모드

```python
            elif key == ord('s'):
                self._save()
```
- 's' 키: 결과 이미지 저장

```python
            elif key == ord('z'):
                self.result = self.original.copy()
                self.detections = []
                self.rois = []
                print("원본으로 복원됨")
```
- 'z' 키: 모든 변경 취소, 원본으로 복원

```python
        cv2.destroyAllWindows()
```
- 프로그램 종료 시 모든 OpenCV 창 닫기

---

## 1.4 이미지 표시 (_show_image 메서드)

```python
    def _show_image(self) -> None:
        """Before | After 나란히 표시 + 조작법"""
        h, w = self.original.shape[:2]
```
- `shape[:2]`: (높이, 너비, 채널) 중 높이와 너비만
- BGR 이미지는 (H, W, 3) 형태

```python
        display_original = self.original.copy()
        self._draw_overlays(display_original)
```
- 원본 이미지 복사본에 탐지 결과/ROI 박스 그리기
- 원본 자체는 수정하지 않음

```python
        combined = np.hstack([display_original, self.result])
```
- `np.hstack()`: 두 이미지를 가로로 나란히 붙임
- 왼쪽: 원본 (박스 표시), 오른쪽: 결과

```python
        panel_x = w + 30
        line_h = 55
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thick = 3
```
- `panel_x`: 텍스트 시작 X 좌표 (오른쪽 이미지 위)
- `line_h`: 줄 간격
- `font`: OpenCV 기본 폰트
- `font_scale`: 폰트 크기 배율
- `thick`: 텍스트 두께

```python
        mode_text = {"telea": "Telea", "telea_cv": "cv2.inpaint", "lama": "LaMa"}
        cv2.putText(combined, f"[Mode] {mode_text.get(self.mode, self.mode)}",
                    (panel_x, 80), font, 5.0, (0, 255, 0), 8)
```
- 현재 모드를 초록색으로 크게 표시
- `putText(이미지, 텍스트, 위치, 폰트, 크기, 색상, 두께)`
- 색상은 BGR 순서: (0, 255, 0) = 초록

```python
        cv2.putText(combined, f"Detections: {len(self.detections)} | ROIs: {len(self.rois)}",
                    (panel_x, 160), font, 1.5, (0, 255, 255), 3)
```
- 탐지된 객체 수와 ROI 수를 노란색으로 표시

```python
        controls = [
            ("=== Controls ===", (255, 255, 255)),
            ("d: YOLO Detect", (100, 255, 100)),
            # ... 기타 조작법
        ]
```
- 조작법 목록: (텍스트, BGR 색상) 튜플

```python
        y = 230
        for text, color in controls:
            cv2.putText(combined, text, (panel_x, y), font, font_scale, color, thick)
            y += line_h
```
- 각 조작법을 순서대로 표시
- 매번 y 좌표를 증가시켜 다음 줄로 이동

```python
        cv2.putText(combined, "[ Original ]", (w // 2 - 100, h - 30),
                    font, 1.3, (255, 255, 255), 3)
        cv2.putText(combined, "[ Result ]", (w + w // 2 - 80, h - 30),
                    font, 1.3, (255, 255, 255), 3)
```
- 이미지 하단에 "Original", "Result" 라벨 표시

```python
        cv2.imshow(self.window_name, combined)
```
- 합쳐진 이미지를 창에 표시

---

## 1.5 오버레이 그리기 (_draw_overlays 메서드)

```python
    def _draw_overlays(self, image: np.ndarray) -> None:
        """탐지 결과와 ROI 박스 그리기"""
        for det in self.detections:
            x, y, w, h = det.bbox
```
- 각 탐지 결과에서 바운딩 박스 좌표 추출
- `bbox`: (x, y, width, height) 형식

```python
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
```
- `rectangle(이미지, 좌상단, 우하단, 색상, 두께)`
- 초록색 사각형으로 탐지 영역 표시

```python
            label = f"{det.label} {det.confidence * 100:.0f}%"
            cv2.putText(image, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```
- 박스 위에 클래스명과 신뢰도 표시
- `:.0f%`: 소수점 없이 퍼센트로 표시

```python
        for i, roi in enumerate(self.rois):
            x, y, w, h = roi
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)
```
- 수동 ROI는 주황색으로 표시
- `enumerate()`: 인덱스와 값을 동시에 가져옴

```python
            cv2.putText(image, f"ROI {i + 1}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
```
- ROI 번호 표시 (1부터 시작)

---

## 1.6 YOLO 탐지 (_detect 메서드)

```python
    def _detect(self) -> None:
        """YOLO 탐지"""
        try:
            if self.detector is None:
                print("YOLO 모델 로딩 중...")
                self.detector = load_default_detector(Path("models"))
```
- **지연 로딩**: 처음 탐지할 때만 모델 로드
- 이후에는 이미 로드된 모델 재사용

```python
            print("객체 탐지 중...")
            self.detections = self.detector.detect(self.original)
            print(f"탐지 완료: {len(self.detections)}개 객체 발견")
```
- 원본 이미지에서 객체 탐지 수행
- 결과를 `self.detections`에 저장

```python
        except FileNotFoundError as e:
            print(f"오류: {e}")
        except Exception as e:
            print(f"탐지 실패: {e}")
```
- 모델 파일이 없거나 에러 발생 시 처리

---

## 1.7 ROI 추가 (_add_roi 메서드)

```python
    def _add_roi(self) -> None:
        """cv2.selectROI로 ROI 추가"""
        print("ROI 선택: 마우스로 드래그 후 Enter/Space, 취소는 c")
        roi = cv2.selectROI("Select ROI", self.original, fromCenter=False, showCrosshair=True)
```
- `selectROI()`: 마우스로 영역 선택하는 대화상자
- `fromCenter=False`: 좌상단에서 드래그
- `showCrosshair=True`: 십자선 표시

```python
        cv2.destroyWindow("Select ROI")
```
- 선택 완료 후 ROI 선택 창 닫기

```python
        x, y, w, h = roi
        if w > 0 and h > 0:
            self.rois.append((x, y, w, h))
            print(f"ROI 추가됨: ({x}, {y}, {w}, {h})")
        else:
            print("ROI 선택 취소됨")
```
- 유효한 영역이면 목록에 추가
- 취소하면 (0, 0, 0, 0) 반환됨

---

## 1.8 마스크 생성 (_create_mask 메서드)

```python
    def _create_mask(self) -> np.ndarray:
        """GrabCut으로 마스크 생성"""
        h, w = self.original.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
```
- 이미지와 같은 크기의 검은색 마스크 생성
- `dtype=np.uint8`: 0-255 값 저장

```python
        boxes = [(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]) for d in self.detections]
        boxes.extend(self.rois)
```
- 탐지 결과와 수동 ROI를 하나의 리스트로 합침

```python
        for bx, by, bw, bh in boxes:
            # 경계 보정
            x1 = max(bx, 0)
            y1 = max(by, 0)
            x2 = min(bx + bw, w - 1)
            y2 = min(by + bh, h - 1)
```
- 박스 좌표가 이미지 범위를 벗어나지 않도록 보정
- `max(bx, 0)`: 음수면 0으로
- `min(bx + bw, w - 1)`: 이미지 너비를 초과하면 제한

```python
            if x2 - x1 < 10 or y2 - y1 < 10:
                mask[y1:y2, x1:x2] = 255
                continue
```
- 너무 작은 영역은 GrabCut 없이 직접 마스크 처리
- `mask[y1:y2, x1:x2] = 255`: 해당 영역을 흰색(255)으로

```python
            try:
                gc_mask = np.zeros((h, w), dtype=np.uint8)
                bgd = np.zeros((1, 65), np.float64)
                fgd = np.zeros((1, 65), np.float64)
```
- **GrabCut 초기화**
- `gc_mask`: GrabCut 결과 저장용 마스크
- `bgd`, `fgd`: 배경/전경 모델 (GMM 파라미터, 65개)

```python
                cv2.grabCut(self.original, gc_mask, (x1, y1, x2 - x1, y2 - y1),
                            bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
```
- **GrabCut 알고리즘 실행**
- `3`: 반복 횟수
- `GC_INIT_WITH_RECT`: 사각형 영역으로 초기화

```python
                fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
```
- `GC_FGD`: 확실한 전경
- `GC_PR_FGD`: 아마도 전경 (Probable Foreground)
- 둘 다 255(흰색)로, 나머지는 0(검정)으로

```python
                if np.any(fg):
                    mask = cv2.bitwise_or(mask, fg)
                else:
                    mask[y1:y2, x1:x2] = 255
```
- 전경이 있으면 기존 마스크와 OR 연산
- 없으면 전체 영역을 마스크로

```python
            except cv2.error:
                mask[y1:y2, x1:x2] = 255
```
- GrabCut 실패 시 전체 영역을 마스크로

```python
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask
```
- **팽창 연산 (Dilation)**
- 5x5 커널로 2회 팽창
- 마스크 경계를 확장하여 인페인팅 품질 향상

---

## 1.9 인페인팅 실행 (_erase 메서드)

```python
    def _erase(self) -> None:
        """선택된 영역 제거"""
        if not self.detections and not self.rois:
            print("제거할 영역이 없습니다. 'd'로 탐지하거나 'r'로 ROI를 추가하세요.")
            return
```
- 제거할 영역이 없으면 안내 메시지 출력

```python
        print(f"마스크 생성 중... ({len(self.detections)}개 탐지 + {len(self.rois)}개 ROI)")
        mask = self._create_mask()
```
- 마스크 생성

```python
        if not np.any(mask):
            print("마스크가 비어있습니다.")
            return
```
- `np.any()`: 배열에 0이 아닌 값이 있는지 확인

```python
        try:
            if self.mode == "telea":
                print("Telea 인페인팅 처리 중 (직접 구현)...")
                self.result = telea_inpaint(self.result, mask, inpaint_radius=5)
```
- **Telea 직접 구현** 호출
- `inpaint_radius=5`: 복원에 사용할 이웃 픽셀 반경

```python
            elif self.mode == "telea_cv":
                print("cv2.inpaint() 처리 중...")
                self.result = cv2.inpaint(self.result, mask, 5, cv2.INPAINT_TELEA)
```
- **OpenCV 내장 함수** 사용
- `cv2.INPAINT_TELEA`: Telea 알고리즘 지정

```python
            elif self.mode == "lama":
                if self.lama is None:
                    print("LaMa 모델 로딩 중...")
                    self.lama = load_lama_model(Path("models"))
                    if self.lama is None:
                        print("LaMa 모델을 찾을 수 없습니다. models/lama/lama_fp32.onnx 필요")
                        return
```
- **LaMa 지연 로딩**
- 모델 파일이 없으면 에러 메시지

```python
                print("LaMa 인페인팅 처리 중...")
                self.result = self.lama.inpaint(self.result, mask)
```
- LaMa 모델로 인페인팅

```python
            print("제거 완료!")
            self.detections = []
            self.rois = []
```
- 처리 후 탐지/ROI 목록 초기화

```python
        except Exception as e:
            print(f"제거 실패: {e}")
```
- 에러 처리

---

## 1.10 결과 저장 (_save 메서드)

```python
    def _save(self) -> None:
        """결과 저장"""
        output_path = "result.png"
        cv2.imwrite(output_path, self.result)
        print(f"저장됨: {output_path}")
```
- `imwrite()`: 이미지를 파일로 저장
- PNG 형식으로 현재 디렉토리에 저장

---

## 1.11 메인 함수

```python
def main():
    if len(sys.argv) < 2:
        print("사용법: python app.py <이미지 경로>")
        print("예: python app.py image.jpg")
        sys.exit(1)
```
- `sys.argv`: 명령줄 인자 리스트
- `sys.argv[0]`: 스크립트 이름
- `sys.argv[1]`: 첫 번째 인자 (이미지 경로)

```python
    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"파일을 찾을 수 없습니다: {image_path}")
        sys.exit(1)
```
- 파일 존재 여부 확인

```python
    print(f"이미지 로드: {image_path}")
    print("\n=== 키보드 단축키 ===")
    # ... 단축키 안내 출력
```
- 사용자에게 조작법 안내

```python
    app = ObjectEraser(image_path)
    app.run()
```
- 애플리케이션 인스턴스 생성 및 실행

```python
if __name__ == "__main__":
    main()
```
- 스크립트로 직접 실행할 때만 `main()` 호출
- 모듈로 import할 때는 실행되지 않음

---

# 2. core/detect.py - YOLO 객체 탐지

## 2.1 DetectionResult 데이터클래스

```python
from dataclasses import dataclass

@dataclass
class DetectionResult:
    """YOLO 탐지 결과"""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
```
- `@dataclass`: 자동으로 `__init__`, `__repr__` 등 생성
- `label`: 클래스 이름 (예: "person", "car")
- `confidence`: 신뢰도 (0.0 ~ 1.0)
- `bbox`: 바운딩 박스 좌표

---

## 2.2 클래스 이름 로드

```python
def _load_names(names_path: Path) -> List[str]:
    with names_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
```
- `.names` 파일에서 클래스 이름 목록 로드
- `strip()`: 양쪽 공백/줄바꿈 제거
- 빈 줄은 제외

---

## 2.3 YOLODetector 초기화

```python
class YOLODetector:
    def __init__(
        self,
        cfg_path: Path,
        weights_path: Path,
        names_path: Path,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
```
- `cfg_path`: 네트워크 구조 파일 (.cfg)
- `weights_path`: 학습된 가중치 파일 (.weights)
- `names_path`: 클래스 이름 파일 (.names)
- `conf_threshold`: 신뢰도 임계값 (이하는 무시)
- `nms_threshold`: NMS 임계값 (중복 제거)

```python
        if not cfg_path.exists() or not weights_path.exists():
            raise FileNotFoundError("YOLO config/weights not found...")
```
- 필수 파일 존재 확인

```python
        self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
```
- **Darknet 모델 로드**
- OpenCV DNN 모듈로 YOLO 네트워크 생성

```python
        self.names = _load_names(names_path) if names_path.exists() else []
```
- 클래스 이름 로드 (없으면 빈 리스트)

```python
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
```
- 임계값 저장

```python
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
```
- **출력 레이어 이름 가져오기**
- `getUnconnectedOutLayers()`: 최종 출력 레이어 인덱스
- YOLO는 여러 스케일에서 탐지하므로 여러 출력 레이어 존재

---

## 2.4 탐지 수행 (detect 메서드)

```python
    def detect(self, image_bgr: np.ndarray) -> List[DetectionResult]:
        h, w = image_bgr.shape[:2]
```
- 이미지 크기 가져오기

```python
        blob = cv2.dnn.blobFromImage(image_bgr, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
```
- **Blob 생성** (신경망 입력 형식)
- `scalefactor=1/255.0`: 0-255 → 0-1 정규화
- `size=(416, 416)`: YOLO 입력 크기
- `swapRB=True`: BGR → RGB 변환
- `crop=False`: 비율 유지 (크롭 안함)

```python
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
```
- blob을 네트워크 입력으로 설정
- 순전파 실행, 출력 레이어 결과 반환

```python
        boxes: List[tuple[int, int, int, int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []
```
- 탐지 결과 저장용 리스트

```python
        for output in outputs:
            for detection in output:
```
- 각 출력 레이어, 각 탐지 결과 순회

```python
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
```
- `detection[0:4]`: 박스 좌표
- `detection[4]`: objectness (객체 존재 확률)
- `detection[5:]`: 각 클래스별 점수
- `argmax()`: 가장 높은 점수의 클래스 ID

```python
                if confidence < self.conf_threshold:
                    continue
```
- 신뢰도가 임계값 미만이면 무시

```python
                center_x, center_y, bw, bh = (
                    int(detection[0] * w),
                    int(detection[1] * h),
                    int(detection[2] * w),
                    int(detection[3] * h),
                )
```
- YOLO 출력은 **정규화된 좌표** (0-1)
- 원본 이미지 크기로 변환

```python
                x = max(int(center_x - bw / 2), 0)
                y = max(int(center_y - bh / 2), 0)
```
- 중심 좌표 → 좌상단 좌표 변환
- 음수 방지

```python
                boxes.append((x, y, bw, bh))
                confidences.append(confidence)
                class_ids.append(class_id)
```
- 결과 저장

```python
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
```
- **Non-Maximum Suppression (NMS)**
- 겹치는 박스 중 가장 좋은 것만 선택
- 반환값: 선택된 박스의 인덱스

```python
        results: List[DetectionResult] = []
        for i in indices.flatten() if len(indices) > 0 else []:
```
- `flatten()`: 2D 배열 → 1D 배열
- 빈 경우 빈 리스트 반환

```python
            label = self.names[class_ids[i]] if class_ids[i] < len(self.names) else f"id{class_ids[i]}"
            results.append(DetectionResult(label=label, confidence=confidences[i], bbox=boxes[i]))
```
- 클래스 ID를 이름으로 변환
- `DetectionResult` 객체 생성

```python
        return results
```
- 탐지 결과 리스트 반환

---

## 2.5 기본 탐지기 로더

```python
def load_default_detector(model_dir: Path) -> YOLODetector:
    cfg_candidates = sorted(model_dir.rglob("*.cfg"))
    weight_candidates = sorted(model_dir.rglob("*.weights"))
    name_candidates = sorted(model_dir.rglob("*.names"))
```
- `rglob()`: 재귀적으로 파일 검색
- 모든 .cfg, .weights, .names 파일 찾기

```python
    if not cfg_candidates or not weight_candidates:
        raise FileNotFoundError("YOLO config/weights not found...")
```
- 필수 파일이 없으면 에러

```python
    cfg_path = cfg_candidates[0]
    weights_path = weight_candidates[0]
    for cfg in cfg_candidates:
        stem = cfg.stem
        match = cfg.with_suffix(".weights")
        if match.exists():
            cfg_path = cfg
            weights_path = match
            break
```
- **매칭 쌍 찾기**
- 같은 이름의 .cfg와 .weights 파일 우선

```python
    names_path = cfg_path.with_suffix(".names")
    if not names_path.exists():
        names_path = model_dir / "coco.names"
        if not names_path.exists() and name_candidates:
            names_path = name_candidates[0]
```
- .names 파일 찾기 (여러 위치 시도)

```python
    return YOLODetector(cfg_path=cfg_path, weights_path=weights_path, names_path=names_path)
```
- YOLODetector 인스턴스 생성 및 반환

---

# 3. core/inpaint.py - Telea 알고리즘 직접 구현

## 3.1 픽셀 상태 상수

```python
KNOWN = 0      # 알려진 픽셀 (원본 또는 이미 복원됨)
BAND = 1       # 경계 픽셀 (처리 대기 중)
UNKNOWN = 2    # 알려지지 않은 픽셀 (마스크 내부)
```
- **Fast Marching Method**의 세 가지 상태
- BAND: 경계선을 따라 확장되는 "파도"

---

## 3.2 TeleaInpainter 초기화

```python
class TeleaInpainter:
    def __init__(self, inpaint_radius: int = 5):
        self.radius = inpaint_radius
        self.epsilon = 1e-6  # 0으로 나누기 방지
```
- `radius`: 복원 시 참조할 이웃 픽셀 범위
- `epsilon`: 수치 안정성을 위한 작은 값

---

## 3.3 인페인팅 메인 함수

```python
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        result = image.copy().astype(np.float32)
```
- 결과 이미지를 float32로 변환 (정밀한 연산)

```python
        binary_mask = (mask > 0).astype(np.uint8)
```
- 마스크 이진화: 0이 아닌 값 → 1

```python
        flag = np.zeros((h, w), dtype=np.uint8)
        flag[binary_mask > 0] = UNKNOWN
```
- **상태 맵 초기화**
- 마스크 영역: UNKNOWN
- 나머지: KNOWN (기본값 0)

```python
        dist = np.full((h, w), np.inf, dtype=np.float32)
        dist[flag == KNOWN] = 0
```
- **거리 맵 초기화**
- 모든 픽셀: 무한대
- KNOWN 픽셀: 0 (경계까지 거리 없음)

```python
        heap: List[Tuple[float, int, int]] = []
        self._init_boundary(flag, dist, heap, h, w)
```
- **우선순위 큐 (힙) 초기화**
- 경계 픽셀들을 큐에 추가

```python
        self._fast_marching(result, flag, dist, heap, h, w)
```
- **Fast Marching Method 실행**
- 경계에서 안쪽으로 픽셀 복원

```python
        return np.clip(result, 0, 255).astype(np.uint8)
```
- 값 범위 0-255로 제한
- uint8로 변환하여 반환

---

## 3.4 경계 초기화 (_init_boundary)

```python
    def _init_boundary(self, flag, dist, heap, h, w):
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
```
- **4방향 이웃**: 상, 하, 좌, 우

```python
        for y in range(h):
            for x in range(w):
                if flag[y, x] == UNKNOWN:
```
- 모든 UNKNOWN 픽셀 순회

```python
                    for dy, dx in neighbors:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if flag[ny, nx] == KNOWN:
                                flag[y, x] = BAND
                                dist[y, x] = 1.0
                                heapq.heappush(heap, (1.0, y, x))
                                break
```
- 이웃 중 KNOWN 픽셀이 있으면
- → 현재 픽셀을 BAND로 변경
- → 거리를 1로 설정
- → 우선순위 큐에 추가

---

## 3.5 Fast Marching Method (_fast_marching)

```python
    def _fast_marching(self, image, flag, dist, heap, h, w):
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while heap:
            d, y, x = heapq.heappop(heap)
```
- **최소 거리 픽셀 추출**
- 힙에서 가장 작은 거리의 픽셀

```python
            if flag[y, x] == KNOWN:
                continue
```
- 이미 처리된 픽셀은 건너뛰기

```python
            self._inpaint_pixel(image, flag, dist, y, x, h, w)
```
- **현재 픽셀 복원**

```python
            flag[y, x] = KNOWN
```
- 상태를 KNOWN으로 변경

```python
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if flag[ny, nx] == UNKNOWN:
                        flag[ny, nx] = BAND
                        new_dist = self._compute_distance(dist, ny, nx, h, w)
                        dist[ny, nx] = new_dist
                        heapq.heappush(heap, (new_dist, ny, nx))
```
- 이웃 UNKNOWN 픽셀들을 BAND로 변경
- 거리 계산 후 큐에 추가

---

## 3.6 Eikonal 방정식으로 거리 계산 (_compute_distance)

```python
    def _compute_distance(self, dist, y, x, h, w):
        dx_min = np.inf
        if x > 0:
            dx_min = min(dx_min, dist[y, x - 1])
        if x < w - 1:
            dx_min = min(dx_min, dist[y, x + 1])
```
- **x 방향 최소 거리**
- 좌우 이웃 중 작은 거리 선택

```python
        dy_min = np.inf
        if y > 0:
            dy_min = min(dy_min, dist[y - 1, x])
        if y < h - 1:
            dy_min = min(dy_min, dist[y + 1, x])
```
- **y 방향 최소 거리**

```python
        if np.isinf(dx_min) and np.isinf(dy_min):
            return np.inf
```
- 둘 다 무한대면 무한대 반환

```python
        if np.isinf(dx_min):
            return dy_min + 1.0
        if np.isinf(dy_min):
            return dx_min + 1.0
```
- 하나만 유효하면 그 값 + 1

```python
        # Eikonal 방정식: (d - dx)^2 + (d - dy)^2 = 1
        a = 2.0
        b = -2.0 * (dx_min + dy_min)
        c = dx_min ** 2 + dy_min ** 2 - 1.0
```
- **이차방정식 계수**
- Eikonal 방정식을 이차방정식으로 변환

```python
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return min(dx_min, dy_min) + 1.0
```
- 판별식이 음수면 단순 근사

```python
        return (-b + np.sqrt(discriminant)) / (2 * a)
```
- **근의 공식**으로 거리 계산
- 더 큰 근 선택 (+ 부호)

---

## 3.7 픽셀 복원 (_inpaint_pixel)

```python
    def _inpaint_pixel(self, image, flag, dist, y, x, h, w):
        r = self.radius
        total_weight = 0.0
        weighted_sum = np.zeros(3, dtype=np.float32)
```
- `r`: 참조할 이웃 범위
- `weighted_sum`: RGB 채널별 가중합

```python
        grad_y, grad_x = self._estimate_gradient(dist, y, x, h, w)
```
- **그래디언트 추정**
- 경계 방향 벡터

```python
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny, nx = y + dy, x + dx
```
- r x r 범위의 이웃 픽셀 순회

```python
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if flag[ny, nx] != KNOWN:
                    continue
```
- 범위 밖이거나 KNOWN이 아니면 무시

```python
                pixel_dist = np.sqrt(dy ** 2 + dx ** 2)
                if pixel_dist > r or pixel_dist < self.epsilon:
                    continue
```
- 반경 초과하거나 너무 가까우면 무시

```python
                # 1. 거리 가중치
                w_dist = 1.0 / (pixel_dist ** 2 + self.epsilon)
```
- **거리 가중치**: 가까울수록 큼
- 역제곱 법칙

```python
                # 2. 방향 가중치
                if pixel_dist > self.epsilon:
                    dir_y = dy / pixel_dist
                    dir_x = dx / pixel_dist
                    dot = grad_y * dir_y + grad_x * dir_x
                    w_dir = max(0.0, -dot + 1.0)
                else:
                    w_dir = 1.0
```
- **방향 가중치**: 그래디언트와 반대 방향일수록 큼
- 등고선(경계)을 따라가는 방향 선호

```python
                # 3. 레벨셋 가중치
                level_diff = abs(dist[ny, nx] - dist[y, x])
                w_level = 1.0 / (1.0 + level_diff)
```
- **레벨셋 가중치**: 같은 거리의 픽셀 선호
- 등고선상의 픽셀이 더 관련 있음

```python
                weight = w_dist * w_dir * w_level
                total_weight += weight
                weighted_sum += weight * image[ny, nx]
```
- **최종 가중치**: 세 가중치의 곱
- 가중합 계산

```python
        if total_weight > self.epsilon:
            image[y, x] = weighted_sum / total_weight
        else:
            self._fallback_average(image, flag, y, x, h, w, r)
```
- 가중 평균으로 픽셀 값 결정
- 가중치 합이 너무 작으면 단순 평균 사용

---

## 3.8 그래디언트 추정 (_estimate_gradient)

```python
    def _estimate_gradient(self, dist, y, x, h, w):
        grad_y = 0.0
        grad_x = 0.0
```
- 거리 맵의 그래디언트 (경계 방향)

```python
        # x 방향 그래디언트
        if x > 0 and x < w - 1:
            d_left = dist[y, x - 1]
            d_right = dist[y, x + 1]
            if not (np.isinf(d_left) or np.isinf(d_right)):
                grad_x = (d_right - d_left) / 2.0
```
- **중앙 차분**: (오른쪽 - 왼쪽) / 2
- 무한대 값은 제외

```python
        # y 방향도 동일하게
        # ...
```

```python
        mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        if mag < self.epsilon:
            return 0.0, 0.0
        return grad_y / mag, grad_x / mag
```
- **정규화**: 크기를 1로
- 방향만 필요하므로

---

## 3.9 폴백 평균 (_fallback_average)

```python
    def _fallback_average(self, image, flag, y, x, h, w, r):
        count = 0
        total = np.zeros(3, dtype=np.float32)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and flag[ny, nx] == KNOWN:
                    total += image[ny, nx]
                    count += 1

        if count > 0:
            image[y, x] = total / count
```
- **단순 평균**: 가중치 없이 평균
- Telea 가중치 계산이 실패했을 때 사용

---

## 3.10 편의 함수

```python
def telea_inpaint(image, mask, inpaint_radius=5):
    inpainter = TeleaInpainter(inpaint_radius)
    return inpainter.inpaint(image, mask)
```
- 함수형 인터페이스 제공

```python
def cv_inpaint(image, mask, inpaint_radius=5):
    import cv2
    return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
```
- OpenCV 내장 함수 래퍼

---

# 4. core/lama.py - LaMa 딥러닝 인페인팅

## 4.1 ONNX Runtime Import

```python
try:
    import onnxruntime as ort
except ImportError:
    ort = None
```
- 선택적 의존성 처리
- onnxruntime이 없어도 다른 모드 사용 가능

---

## 4.2 LamaInpainter 초기화

```python
class LamaInpainter:
    MODEL_SIZE = 512
```
- LaMa ONNX 모델은 512x512 고정 입력

```python
    def __init__(self, model_path: Path):
        if ort is None:
            raise ImportError("onnxruntime 설치 필요: pip install onnxruntime")
```
- onnxruntime 없으면 에러

```python
        providers = ort.get_available_providers()
        self.session = ort.InferenceSession(str(model_path), providers=providers)
```
- **사용 가능한 실행 제공자** 자동 선택
- CUDA, DirectML, CPU 등
- **추론 세션** 생성

---

## 4.3 인페인팅 메인 함수

```python
    def inpaint(self, image, mask):
        orig_h, orig_w = image.shape[:2]
```
- 원본 크기 저장 (복원 시 필요)

```python
        img_input, mask_input, pad_info = self._preprocess(image, mask)
```
- **전처리**: 512x512로 변환

```python
        output = self.session.run(None, {
            "image": img_input,
            "mask": mask_input
        })[0]
```
- **ONNX 추론 실행**
- 입력: 딕셔너리 (이름 → 텐서)
- 출력: 리스트 (첫 번째 출력 사용)

```python
        result = self._postprocess(output, orig_h, orig_w, pad_info)
        return result
```
- **후처리**: 원본 크기로 복원

---

## 4.4 전처리 (_preprocess)

```python
    def _preprocess(self, image, mask):
        h, w = image.shape[:2]
        scale = self.MODEL_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
```
- **스케일 계산**: 긴 쪽이 512가 되도록
- 비율 유지

```python
        img_resized = cv2.resize(image, (new_w, new_h))
        mask_resized = cv2.resize(mask, (new_w, new_h))
```
- 이미지와 마스크 리사이즈

```python
        pad_h, pad_w = self.MODEL_SIZE - new_h, self.MODEL_SIZE - new_w
        top, left = pad_h // 2, pad_w // 2
```
- **패딩 크기 계산**
- 512에서 부족한 만큼
- 중앙 정렬을 위해 양쪽에 분배

```python
        img_padded = np.pad(img_resized, ((top, pad_h - top), (left, pad_w - left), (0, 0)), mode='reflect')
        mask_padded = np.pad(mask_resized, ((top, pad_h - top), (left, pad_w - left)), mode='reflect')
```
- **Reflect 패딩**: 경계 픽셀을 반사
- `((상, 하), (좌, 우), (0, 0))`: 채널 축은 패딩 없음

```python
        img_input = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
```
- **BGR → RGB** 변환 (모델 학습 형식)
- **정규화**: 0-255 → 0-1

```python
        img_input = np.transpose(img_input, (2, 0, 1))[np.newaxis, ...]
```
- **HWC → CHW** 변환 (PyTorch 형식)
- (H, W, 3) → (3, H, W)
- **배치 차원 추가**: (1, 3, H, W)

```python
        mask_input = (mask_padded.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, ...]
        mask_input = (mask_input > 0.5).astype(np.float32)
```
- 마스크 정규화 및 이진화
- (1, 1, H, W) 형태

```python
        return img_input, mask_input, (top, left, new_h, new_w)
```
- 패딩 정보도 반환 (후처리에 필요)

---

## 4.5 후처리 (_postprocess)

```python
    def _postprocess(self, output, orig_h, orig_w, pad_info):
        top, left, new_h, new_w = pad_info
```
- 패딩 정보 언팩

```python
        result = output[0].transpose(1, 2, 0)
```
- **CHW → HWC** 변환
- (3, H, W) → (H, W, 3)

```python
        result = np.clip(result, 0, 255).astype(np.uint8)
```
- **ONNX 모델은 0-255 범위로 출력**
- 클리핑 후 uint8 변환

```python
        result = result[top:top + new_h, left:left + new_w]
```
- **패딩 제거**
- 패딩 전 크기로 잘라내기

```python
        result = cv2.resize(result, (orig_w, orig_h))
```
- **원본 크기로 복원**

```python
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
```
- **RGB → BGR** 변환 (OpenCV 형식)

```python
        return result
```

---

## 4.6 모델 로더

```python
def load_lama_model(model_dir: Path) -> Optional[LamaInpainter]:
    paths = [
        model_dir / "lama" / "lama_fp32.onnx",
        model_dir / "lama" / "lama.onnx",
        model_dir / "lama_fp32.onnx",
    ]
```
- 여러 가능한 경로 시도

```python
    for path in paths:
        if path.exists():
            return LamaInpainter(path)
    return None
```
- 첫 번째로 존재하는 파일 사용
- 없으면 None 반환

---

# 5. 알고리즘 요약

## 5.1 Telea 알고리즘 (Fast Marching Method)

1. **초기화**: 마스크 경계 픽셀을 BAND로 설정
2. **반복**:
   - 경계에서 가장 가까운 픽셀 선택
   - 이웃 KNOWN 픽셀들의 가중 평균으로 복원
   - 새로운 경계 픽셀 추가
3. **가중치 요소**:
   - 거리 가중치: 가까울수록 큼
   - 방향 가중치: 등고선 방향 선호
   - 레벨셋 가중치: 같은 거리 선호

## 5.2 LaMa (Large Mask Inpainting)

1. **전처리**: 512x512로 리사이즈 + 패딩
2. **추론**: ONNX 모델로 딥러닝 인페인팅
3. **후처리**: 패딩 제거 + 원본 크기로 복원

## 5.3 GrabCut

1. **초기화**: 사각형 영역으로 전경/배경 추정
2. **반복**: GMM으로 전경/배경 모델 업데이트
3. **결과**: 전경 마스크 추출

---

# 6. 참고 자료

- **Telea 논문**: A. Telea, "An Image Inpainting Technique Based on the Fast Marching Method", Journal of Graphics Tools, 2004
- **LaMa 모델**: [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX)
- **YOLO**: You Only Look Once (Darknet)
- **GrabCut**: C. Rother, V. Kolmogorov, A. Blake, "GrabCut: Interactive Foreground Extraction", SIGGRAPH 2004
