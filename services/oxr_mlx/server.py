"""OXR model service on Apple silicon: OXR-1.0 in MLX (model.py), answering the OpenAI-compatible
chat requests the document parsing API sends to the vLLM service (port 8080).

    MODEL_PATH="$PWD/models/OXR-1.0" python services/oxr_mlx/server.py --port 8080

POST /v1/chat/completions with one user message (an image_url data URL and a text prompt): greedy
decoding, with repetition_penalty and max_tokens as sent. One request at a time; set
oxr_model.max_concurrency to 1 (services/oxr_mlx/config.mlx.yaml).

Memory: MLX may use at most OXR_MLX_MEMORY_GB (default 5.5) and keeps at most 1 GB of freed
buffers. Generation stops, and the request fails, when the machine has less than
OXR_MLX_MIN_FREE_GB (default 1.5) available, rather than pushing it into swap. Every request logs
its tokens, time and peak memory.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import mlx.core as mx
import psutil
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model import OXR  # noqa: E402

GB = 2**30
parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8080)
args = parser.parse_args()
model_path = Path(os.environ.get("MODEL_PATH", "models/OXR-1.0"))
memory_gb = float(os.environ.get("OXR_MLX_MEMORY_GB", "5.5"))
min_free_gb = float(os.environ.get("OXR_MLX_MIN_FREE_GB", "1.5"))

if not (model_path / "model.safetensors").is_file():
    sys.exit(f"no model.safetensors in {model_path}; set MODEL_PATH to the OXR-1.0 model folder")
if psutil.virtual_memory().available < (min_free_gb + 3.5) * GB:
    sys.exit(f"only {psutil.virtual_memory().available / GB:.1f} GB available; the model needs about 3.5 GB "
             f"and {min_free_gb} GB must stay free (OXR_MLX_MIN_FREE_GB)")
mx.set_memory_limit(int(memory_gb * GB))
mx.set_cache_limit(GB)
model = OXR(model_path)


def complete(body: dict) -> dict:
    (message,) = body["messages"]
    image, prompt = None, ""
    for item in message["content"]:
        if item["type"] == "image_url":
            image = Image.open(io.BytesIO(base64.b64decode(item["image_url"]["url"].split(",", 1)[1]))).convert("RGB")
        elif item["type"] == "text":
            prompt += item["text"]
    if image is None:
        raise ValueError("OXR needs an image in the message")
    max_tokens = int(body.get("max_tokens") or 8192)
    penalty = float(body.get("repetition_penalty") or 1.0)
    low = lambda n: psutil.virtual_memory().available < min_free_gb * GB  # noqa: E731
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    ids, prompt_tokens, finish = model.generate(image, prompt, max_tokens=max_tokens, repetition_penalty=penalty, stop=low)
    print(f"{prompt_tokens:5} + {len(ids):5} tokens  {time.perf_counter() - t0:6.1f}s  peak {mx.get_peak_memory() / GB:.2f}, "
          f"after {mx.get_active_memory() / GB:.2f} GB  available {psutil.virtual_memory().available / GB:.1f} GB"
          f"{'  STOPPED: low memory' if finish == 'stopped' else ''}", flush=True)
    if finish == "stopped":
        raise MemoryError(f"less than {min_free_gb} GB available; stopped after {len(ids)} tokens")
    text = model.tokenizer.decode(ids, skip_special_tokens=True)
    return {
        "id": f"oxr-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()), "model": body.get("model", "oxr"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(ids), "total_tokens": prompt_tokens + len(ids)},
    }


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            if self.path.rstrip("/") != "/v1/chat/completions":
                raise LookupError(self.path)
            payload, status = complete(json.loads(self.rfile.read(int(self.headers["content-length"])))), 200
        except Exception as error:  # noqa: BLE001 - report any failure to the client
            payload, status = {"error": {"message": f"{type(error).__name__}: {error}"}}, 404 if isinstance(error, LookupError) else 400
            print(f"failed: {payload['error']['message'][:300]}", flush=True)
            mx.clear_cache()
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


# One request at a time: the model is not shared between threads
print(f"OXR-1.0 in MLX, {mx.get_active_memory() / GB:.1f} GB (limit {memory_gb} GB), "
      f"listening on http://{args.host}:{args.port}/v1", flush=True)
HTTPServer((args.host, args.port), Handler).serve_forever()
