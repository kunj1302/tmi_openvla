# server.py
import os
import shutil
import logging
from datetime import datetime
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CONFIGURATION
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PORT = 6000 # Using 6000 to avoid conflicts with defaults

# Hugging Face cache location (avoid filling up $HOME on cluster nodes)
# This ensures model/tokenizer weights are downloaded to and re-used from scratch.
HF_CACHE_DIR = "/scratch1/kgolwala/hf_cache"
HF_TMP_DIR = os.path.join(HF_CACHE_DIR, "tmp")
HF_XET_DIR = os.path.join(HF_CACHE_DIR, "xet")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(HF_TMP_DIR, exist_ok=True)
os.makedirs(HF_XET_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_CACHE_DIR, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_CACHE_DIR, "transformers"))
# Hugging Face may use temp files during download; keep those off the root disk/home.
os.environ.setdefault("TMPDIR", HF_TMP_DIR)
# You hit `xet_get(...)` in the traceback; keep Xet cache off $HOME as well.
os.environ.setdefault("HF_XET_CACHE", HF_XET_DIR)
# If Xet causes issues on your cluster filesystem, disable it (falls back to regular downloads).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

try:
    free_gb = shutil.disk_usage(HF_CACHE_DIR).free / (1024 ** 3)
    if free_gb < 25:
        print(
            f"WARNING: Only {free_gb:.1f} GB free in {HF_CACHE_DIR}. "
            "Llama-3.1-8B downloads are ~20+ GB on disk (4-bit affects VRAM, not download size). "
            "You may need to free scratch space or switch to a smaller/quantized repo."
        )
except Exception as e:
    print(f"NOTE: Could not check free space for {HF_CACHE_DIR}: {e}")

print(f"Loading {MODEL_ID} in 4-bit mode... (This takes a minute)")

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_DIR)

# 2. Load Model (Quantized to fit in your A5000 alongside OpenVLA)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=HF_CACHE_DIR,
    device_map="auto",          # Finds your GPU
    load_in_4bit=True,          # Uses ~6GB VRAM
    torch_dtype=torch.float16,
)

# 3. Create Pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id
)

print(">>> Model Loaded Successfully. Waiting for requests...")

@app.route('/filter', methods=['POST'])
def filter_cmd():
    request_start = datetime.now()
    data = request.json
    noisy_text = data.get("instruction", "")
    
    # Log incoming request
    logger.info("=" * 60)
    logger.info(f"[REQUEST] Received filter request")
    logger.info(f"[INPUT] Noisy instruction: '{noisy_text}'")
    print(f"\n{'='*60}")
    print(f"[LLAMA3 SERVER] Received filter request at {request_start.strftime('%H:%M:%S')}")
    print(f"[LLAMA3 SERVER] INPUT: '{noisy_text}'")
    
    # Llama 3 Chat Template
    # Strict prompt with 3 Few-Shot Examples
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a robotic command simplifier. "
                "Convert the input into a single strict command: 'pick up [object] and place it in [target]'. "
                "Remove all fluff, politeness, and urgency. "
                "Standardize verbs to 'pick up' and 'place'. "
                "Output ONLY the clean command string."
            )
        },
        
        {"role": "user", "content": "Hey there, I need you to grab the alphabet soup and then carefully place it in the basket on the counter, could you manage that?"},
        {"role": "assistant", "content": "pick up the alphabet soup and place it in the basket"},
        # The Actual Input
        {"role": "user", "content": noisy_text},
    ]

    # Format the prompt correctly for Llama 3
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Run Inference
    inference_start = datetime.now()
    outputs = pipe(
        prompt, 
        max_new_tokens=30, 
        do_sample=False, # Deterministic (Greedy decoding)
        temperature=None,
        top_p=None
    )
    inference_time = (datetime.now() - inference_start).total_seconds()
    
    # Parse output: Remove the prompt and system tokens
    generated_text = outputs[0]["generated_text"]
    # The model repeats the prompt, so we cut it off
    clean_cmd = generated_text[len(prompt):].strip()
    
    # Log the output
    total_time = (datetime.now() - request_start).total_seconds()
    logger.info(f"[OUTPUT] Clean command: '{clean_cmd}'")
    logger.info(f"[TIMING] Inference: {inference_time:.2f}s, Total: {total_time:.2f}s")
    print(f"[LLAMA3 SERVER] OUTPUT: '{clean_cmd}'")
    print(f"[LLAMA3 SERVER] Inference time: {inference_time:.2f}s, Total time: {total_time:.2f}s")
    print(f"{'='*60}\n")
    
    return jsonify({"clean_command": clean_cmd})

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(f"[LLAMA3 SERVER] Server starting on http://0.0.0.0:{PORT}")
    print(f"[LLAMA3 SERVER] Filter endpoint: http://localhost:{PORT}/filter")
    print(f"[LLAMA3 SERVER] Ready to receive instructions for sanitization")
    print("=" * 60 + "\n")
    logger.info(f"Server starting on port {PORT}")
    # Threaded=False ensures simple sequential processing
    app.run(host='0.0.0.0', port=PORT, threaded=False)