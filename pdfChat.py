import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen3-8B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda"
)

print("Model loaded.")


def handler(job):
    job_input = job["input"]

    text = job_input.get("text", "")
    question = job_input.get("question", "")

    prompt = f"""
You are a helpful assistant.

Answer the question ONLY using the provided text.
If the answer is not present in the text, say:
"The answer is not available in the provided text."

Text:
{text}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response}


runpod.serverless.start({"handler": handler})
