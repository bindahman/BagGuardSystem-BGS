FROM python:3.10-slim

# optimize Python environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# set working directory
WORKDIR /app

# ✅ OpenCV runtime deps (fix: ImportError: libGL.so.1)
# libgl1 provides libGL.so.1
# libglib2.0-0 is a common OpenCV runtime dependency
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# copy dependency file
COPY requirements.txt .

# install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy project files
COPY . .

# start the application
CMD ["python", "Daohan_test_image/test.py", "Daohan_test_video/0.py"]