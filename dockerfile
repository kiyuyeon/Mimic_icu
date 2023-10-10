# 베이스 이미지로 Python 3.12을 사용합니다.
FROM python:3.12

# 작업 디렉토리를 설정합니다.
WORKDIR /app

# 현재 디렉토리의 모든 파일을 컨테이너의 /app 디렉토리에 복사합니다.
COPY . /app

# 필요한 패키지를 설치합니다.
RUN pip install -r requirements.txt

# FastAPI 애플리케이션을 실행합니다.
CMD ["python", "model.py"]

