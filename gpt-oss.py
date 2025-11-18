from transformers import pipeline
import torch
import time

start_time = time.time()

model_id = "/home/nvidia/DGX_Spark-Segmentation-/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "請用繁體中文簡短說明量子力學是什麼。"},
]

out = pipe(
    messages,
    max_new_tokens=256,
)
print(out[0]["generated_text"][-1])

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")