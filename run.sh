#!/bin/bash

# 가상환경 활성화
source .venv/bin/activate

# 사용법 안내
if [ -z "$1" ]; then
    echo "사용법: ./run.sh <이미지 경로>"
    echo "예: ./run.sh image.jpg"
    exit 1
fi

# 앱 실행
python app.py "$1"
