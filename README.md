# 객체 탐지 및 제거 프로그램 (Object Detector & Eraser)

## 1. 프로젝트 소개
YOLO로 객체를 탐지하고 GrabCut + Inpainting으로 제거하는 OpenCV 기반 애플리케이션입니다.

## 2. 주요 기능

### 2.1 인페인팅 모드
| 모드 | 설명 | 키 |
|------|------|---|
| Telea 직접 구현 | Fast Marching Method 알고리즘 직접 구현 | `t` |
| cv2.inpaint() | OpenCV 내장 Telea 인페인팅 | `c` |
| LaMa | 딥러닝 기반 Large Mask Inpainting | `l` |

### 2.2 워크플로우
1. 이미지 로드
2. `d` 키로 YOLO 객체 탐지 또는 `r` 키로 수동 ROI 선택
3. 원하는 모드 선택 (`t`/`c`/`l`)
4. `e` 키로 제거 실행
5. `s` 키로 결과 저장

## 3. 제약 사항
- 입력 이미지: FHD (1920x1080) 이하 권장
- 탐지 대상: COCO 80개 클래스
- 제거 객체 크기: 이미지 면적의 30% 이하 권장

## 4. 설치

```bash
# 가상 환경 생성
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# 의존성 설치
pip install -r requirements.txt
```

## 5. 모델 파일 준비

### YOLOv2 (필수)
`models/` 폴더에 이미 포함:
- `yolov2-tiny.cfg`
- `yolov2-tiny.weights`
- `coco.names`

### LaMa (선택)
```bash
# 방법 1: 직접 다운로드
# https://huggingface.co/Carve/LaMa-ONNX 에서 lama_fp32.onnx 다운로드
# models/lama/ 폴더에 저장

# 방법 2: huggingface-hub 사용
pip install huggingface-hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Carve/LaMa-ONNX', filename='lama_fp32.onnx', local_dir='models/lama')"
```

## 6. 사용법

```bash
python app.py <이미지 경로>
# 예: python app.py image.jpg
```

## 7. 키보드 단축키

| 키 | 기능 |
|----|------|
| `d` | YOLO 탐지 실행 |
| `r` | ROI 수동 추가 (드래그) |
| `e` | 선택 영역 제거 |
| `t` | Telea 직접 구현 모드 |
| `c` | cv2.inpaint() 모드 |
| `l` | LaMa 모드 |
| `s` | 결과 저장 (result.png) |
| `z` | 원본 복원 |
| `q` | 종료 |

## 8. 파일 구조

```
ComputerVisionTeamProject/
├── app.py              # 메인 애플리케이션
├── core/
│   ├── detect.py       # YOLOv2 객체 탐지
│   ├── lama.py         # LaMa ONNX 인페인팅
│   └── inpaint.py      # Telea 알고리즘 직접 구현
├── models/
│   ├── yolov2-tiny.cfg
│   ├── yolov2-tiny.weights
│   ├── coco.names
│   └── lama/           # LaMa 모델 (별도 다운로드)
└── requirements.txt
```

## 9. 화면 구성
- 왼쪽: 원본 이미지 (탐지 결과/ROI 표시)
- 오른쪽: 결과 이미지
- 상단: 현재 모드 및 탐지/ROI 개수 표시
