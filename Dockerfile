# Geotag web app + JSON API on one NVIDIA GPU (a 16 GB T4 is enough, e.g. AWS g4dn.xlarge).
#   docker build -t geotag .
#   docker run --gpus all -p 80:7860 -v $PWD/runs_3b:/data -v $PWD/hf:/hf geotag
# /data holds the files demo.py reads from $NV_OUT (see its docstring) plus ckpt/lora_mixed/final;
# /hf caches the CLIP and Qwen2.5-VL weights so they download only once.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY nv nv
COPY demo.py .

ENV NV_OUT=/data HF_HOME=/hf
EXPOSE 7860
CMD ["python", "demo.py", "--adapter", "ckpt/lora_mixed/final", "--port", "7860"]
