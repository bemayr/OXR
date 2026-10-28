"""OXR-1.0 in MLX, for Apple silicon. Used by server.py; checked against the PyTorch model by check.py.

The language model is mlx-lm's Qwen3. The vision encoder, visual head and visual embedding table are
written here after modeling_oxr.py in the model folder (Hugging Face revision c708407), which also
has the image preprocessing and prompt layout reproduced below. Nothing from the model folder is
executed: weights are read from safetensors, the tokenizer is Transformers' own class.
"""

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.generate import generate_step
from mlx_lm.models import qwen3
from mlx_lm.models.cache import make_prompt_cache
from PIL import Image
from transformers import AutoTokenizer

MIN_PIXELS, MAX_PIXELS = 200704, 921600  # as the vLLM plugin (plugins/vllm/oxr_vllm/oxr_processor.py)
EOS = (151643, 151645)
IMAGE_START, IMAGE_END, ATOM = -301, -302, -300  # modeling_oxr.py
PATCH, STRIDE = 16, 2


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """OXRVisualTokenizer.smart_resize."""
    if height < factor or width < factor:
        if height < width:
            width, height = round(factor / height * width), factor
        else:
            height, width = round(factor / width * height), factor
    elif max(height, width) / min(height, width) > 200:
        if height > width:
            height = 200 * width
        else:
            width = 200 * height
    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar, w_bar = math.floor(height / beta / factor) * factor, math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar, w_bar = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def bicubic_matrix(n_out: int, n_in: int, a: float = -0.75) -> np.ndarray:
    """PyTorch's bicubic interpolation (align_corners=False) along one axis, as an n_out × n_in matrix."""
    m = np.zeros((n_out, n_in), np.float64)
    for i in range(n_out):
        x = (i + 0.5) * n_in / n_out - 0.5
        x0 = math.floor(x)
        t = x - x0
        weights = [
            ((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a,
            ((a + 2) * t - (a + 3)) * t * t + 1,
            ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1,
            ((a * (2 - t) - 5 * a) * (2 - t) + 8 * a) * (2 - t) - 4 * a,
        ]
        for k, w in zip(range(-1, 3), weights):
            m[i, min(max(x0 + k, 0), n_in - 1)] += w
    return m


def blocks(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Row-major (h, w, ...) to the order OXR keeps patches in: 2 × 2 blocks, row by row."""
    rest = a.shape[2:]
    return a.reshape(h // STRIDE, STRIDE, w // STRIDE, STRIDE, *rest).swapaxes(1, 2).reshape(h * w, *rest)


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(dim, dim) for _ in range(4))

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        n = x.shape[0]
        q, k, v = (p(x).reshape(n, self.heads, self.head_dim) for p in (self.q_proj, self.k_proj, self.v_proj))
        # apply_rotary_pos_emb_vision: in float32, back to the input type
        q = (q.astype(mx.float32) * cos + rotate_half(q.astype(mx.float32)) * sin).astype(x.dtype)
        k = (k.astype(mx.float32) * cos + rotate_half(k.astype(mx.float32)) * sin).astype(x.dtype)
        q, k, v = (t.transpose(1, 0, 2)[None] for t in (q, k, v))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)
        return self.out_proj(o[0].transpose(1, 0, 2).reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))  # gelu_pytorch_tanh


class EncoderLayer(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(c["hidden_size"], eps=c["layer_norm_eps"])
        self.self_attn = Attention(c["hidden_size"], c["num_attention_heads"])
        self.layer_norm2 = nn.LayerNorm(c["hidden_size"], eps=c["layer_norm_eps"])
        self.mlp = MLP(c["hidden_size"], c["intermediate_size"])

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), cos, sin)
        return x + self.mlp(self.layer_norm2(x))


class Vision(nn.Module):
    """Siglip2NavitModel for one image. Every layer attends to the whole image (fullatt_block_indexes
    is None), so OXR's reordering into windows and back cancels out and is left out here."""

    def __init__(self, c: dict):
        super().__init__()
        self.c = c
        dim = c["hidden_size"]
        self.patch_embedding = nn.Linear(c["num_channels"] * PATCH * PATCH, dim)  # the 16 × 16 convolution
        self.position_embedding = mx.zeros(((c["image_size"] // PATCH) ** 2, dim))
        self.layers = [EncoderLayer(c) for _ in range(c["num_hidden_layers"])]
        head_dim = dim // c["num_attention_heads"]
        self.inv_freq = 1.0 / (10000 ** (np.arange(0, head_dim // 2, 2, dtype=np.float32) / (head_dim // 2)))

    def __call__(self, patches: mx.array, h: int, w: int, keep: tuple[int, ...] = ()) -> tuple[mx.array, list]:
        """Patch rows in OXR's block order → the last layer's output (no post_layernorm, as
        OXRVisualTokenizer._encode takes hidden_states[-1]); also the outputs of the layers in keep."""
        x = self.patch_embedding(patches)
        side = self.c["image_size"] // PATCH
        grid = np.array(self.position_embedding.astype(mx.float32)).reshape(side, side, -1)
        pe = np.einsum("hi,ijc,wj->hwc", bicubic_matrix(h, side), grid, bicubic_matrix(w, side))
        x = x + mx.array(blocks(pe, h, w)).astype(x.dtype)
        rows = blocks(np.repeat(np.arange(h)[:, None], w, 1), h, w)
        cols = blocks(np.repeat(np.arange(w)[None, :], h, 0), h, w)
        rope = np.concatenate([rows[:, None] * self.inv_freq, cols[:, None] * self.inv_freq], axis=-1)
        rope = np.concatenate([rope, rope], axis=-1)[:, None, :]  # tokens, 1 (heads), head_dim
        cos, sin = mx.array(np.cos(rope)), mx.array(np.sin(rope))
        kept = []
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin)
            if i in keep:
                kept.append(x)
        return x, kept


class OXR:
    def __init__(self, path: Path):
        """path: the OXR-1.0 model folder (hf download SoMarkAI/OXR-1.0 --local-dir models/OXR-1.0)."""
        self.path = Path(path)
        config = json.loads((self.path / "config.json").read_text())
        self.vc = config["vit_config"]
        # config.json names the model's own code; the tokenizer does not need it
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), trust_remote_code=False)
        self.pixels = json.loads((self.path / "preprocessor_config.json").read_text())
        weights = mx.load(str(self.path / "model.safetensors"))

        self.llm = qwen3.Model(qwen3.ModelArgs.from_dict(config["llm_config"]))
        llm = self.llm.sanitize({k[4:]: v for k, v in weights.items() if k.startswith("llm.")})
        self.llm.load_weights(list(llm.items()))

        self.vision = Vision(self.vc)
        prefix = "visual_tokenizer.vit.vision_model."
        vision = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
        vision = {k.replace("embeddings.", "").replace("encoder.", "").replace("position_embedding.weight", "position_embedding"): v
                  for k, v in vision.items() if not k.startswith("post_layernorm")}
        vision["patch_embedding.weight"] = vision["patch_embedding.weight"].reshape(self.vc["hidden_size"], -1)
        self.vision.load_weights(list(vision.items()))

        self.head = weights["visual_tokenizer.head.0.weight"]
        self.head_norm = nn.LayerNorm(self.head.shape[0], eps=1e-5)  # torch.nn.LayerNorm's default
        self.head_norm.load_weights([("weight", weights["visual_tokenizer.head.1.weight"]), ("bias", weights["visual_tokenizer.head.1.bias"])])
        self.vte = weights["vte.weight"]
        mx.eval(self.llm.parameters(), self.vision.parameters(), self.head, self.head_norm.parameters(), self.vte)
        del weights
        mx.clear_cache()

    def preprocess(self, image: Image.Image) -> tuple[np.ndarray, int, int]:
        """OXRVisualTokenizer.preprocess for one image: patch rows in block order, and the grid."""
        image = image.convert("RGB")
        width, height = image.size
        rh, rw = smart_resize(height, width, PATCH * STRIDE, MIN_PIXELS, MAX_PIXELS)
        # SiglipImageProcessor with the PIL backend (which Transformers 5 loads only with PyTorch
        # installed): resize, rescale, normalise; the same pixel values (check.py)
        p = self.pixels
        pixels = np.asarray(image.resize((rw, rh), resample=Image.Resampling(p["resample"])), dtype=np.float32)
        pixels = (pixels * np.float32(p["rescale_factor"]) - np.float32(p["image_mean"])) / np.float32(p["image_std"])
        pixels = pixels.transpose(2, 0, 1)
        c, h, w = pixels.shape[0], rh // PATCH, rw // PATCH
        patches = pixels.reshape(1, 1, c, h // STRIDE, STRIDE, PATCH, w // STRIDE, STRIDE, PATCH)
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(h * w, c * PATCH * PATCH)
        return patches, h, w

    def input_ids(self, prompt: str, atoms: int) -> list[int]:
        """The chat prompt with the image's placeholders, as OXR.preprocess_inputs builds it."""
        messages = [{"role": "user", "content": [{"type": "image", "image": None}, {"type": "text", "text": prompt}]}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in text.split("<image>")]
        ids = chunks[0]
        for chunk in chunks[1:]:
            ids = ids + [IMAGE_START] + [ATOM] * atoms + [IMAGE_END] + chunk
        return ids

    def visual_embeddings(self, patches: np.ndarray, h: int, w: int) -> mx.array:
        """OXRVisualTokenizer.forward and the vte lookup: soft visual tokens times the embedding table."""
        x, _ = self.vision(mx.array(patches).astype(mx.bfloat16), h, w)
        features = x.reshape(x.shape[0] // (STRIDE * STRIDE), -1)
        logits = self.head_norm(features @ self.head.T)
        tokens = mx.softmax(logits.astype(mx.float32), axis=-1).astype(logits.dtype)
        return tokens @ self.vte[: tokens.shape[-1]]  # the four indicator columns are zero

    def embeddings(self, ids: list[int], visual: mx.array) -> mx.array:
        """OXR.merge_multimodal: text embeddings, the image's start and end indicators, the visual ones."""
        start, end = ids.index(IMAGE_START), ids.index(IMAGE_END)
        text = self.llm.model.embed_tokens
        parts = [text(mx.array(ids[:start])) if start else None, self.vte[self.vte.shape[0] - 4][None], visual,
                 self.vte[self.vte.shape[0] - 3][None], text(mx.array(ids[end + 1 :])) if end + 1 < len(ids) else None]
        return mx.concatenate([p for p in parts if p is not None], axis=0)

    def generate(self, image: Image.Image, prompt: str, **kwargs) -> tuple[list[int], int, str]:
        return self.generate_from(*self.preprocess(image), prompt, **kwargs)

    def generate_from(self, patches: np.ndarray, h: int, w: int, prompt: str, max_tokens: int = 8192,
                      repetition_penalty: float = 1.0, stop=lambda n: False) -> tuple[list[int], int, str]:
        """Greedy decoding as OXR.generate with do_sample=False: the new token ids, the prompt length
        and the finish reason. stop(n) is asked every 32 tokens and ends generation when True."""
        ids = self.input_ids(prompt, h * w // (STRIDE * STRIDE))
        embeds = self.embeddings(ids, self.visual_embeddings(patches, h, w))
        processors = []
        if repetition_penalty != 1.0:
            # Transformers' RepetitionPenaltyLogitsProcessor over every generated token (mlx-lm's own
            # looks back only 20); generate_step passes only generated tokens when the prompt is empty
            def penalty(tokens: mx.array, logits: mx.array) -> mx.array:
                chosen = logits[:, tokens]
                logits[:, tokens] = mx.where(chosen < 0, chosen * repetition_penalty, chosen / repetition_penalty)
                return logits
            processors.append(penalty)
        out, finish = [], "length"
        for token, _ in generate_step(mx.array([], dtype=mx.int32), self.llm, max_tokens=max_tokens, input_embeddings=embeds,
                                      logits_processors=processors or None, prompt_cache=make_prompt_cache(self.llm)):
            out.append(int(token))
            if out[-1] in EOS:
                finish = "stop"
                break
            if len(out) % 32 == 0 and stop(len(out)):
                finish = "stopped"
                break
        mx.clear_cache()
        return out, len(ids), finish
