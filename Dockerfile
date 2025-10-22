# Python 슬림 베이스
FROM python:3.12-slim

# 시스템 패키지 (OpenCV가 필요로 하는 런타임 라이브러리)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 작업 폴더
WORKDIR /src

# 파이썬 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 복사
COPY src ./src/

# Gradio 외부접속 세팅
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

# (선택) 헬스체크 기다림 없이 바로 실행
CMD ["python", "src/app.py"]
