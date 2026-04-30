FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    pydantic \
    numpy \
    pandas \
    scikit-learn \
    pillow \
    tensorflow-cpu==2.15.1 \
    keras==2.15.0 \
    joblib

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]