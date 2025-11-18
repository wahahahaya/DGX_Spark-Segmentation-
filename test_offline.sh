""
pip install -U transformers accelerate torch triton==3.4 kernels
pip install "huggingface_hub>=0.34.0,<1.0.0"
hf --help  # 確認有這個指令
hf download openai/gpt-oss-20b \
  --local-dir gpt-oss-20b
""

export HF_HUB_OFFLINE=1 # 設定離線模式

python gpt-oss.py