"""Checks the MLX port (model.py) against the PyTorch model in the model folder, step by step.

    MODEL_PATH="$PWD/models/OXR-1.0" python services/oxr_mlx/check.py reference <image> <out.npz> [tokens]
    MODEL_PATH="$PWD/models/OXR-1.0" python services/oxr_mlx/check.py compare <out.npz>...
    MODEL_PATH="$PWD/models/OXR-1.0" python services/oxr_mlx/check.py vision32 <out.npz>

reference runs modeling_oxr.py from the model folder, on the CPU by default (OXR_REFERENCE_DEVICE=mps
uses the GPU; its fused attention needs ~7 GB there for a full-size region), and saves the prompt ids,
the pixel rows, the outputs of vision layers 0, 13 and 26, the visual embeddings, the logits of the
first answer token and the greedy answer. compare prints how close the port comes at each step.
vision32 runs the vision encoder alone in float32 on both sides: the port should match to rounding.

reference and vision32 need torch and torchvision as well (pip install torch==2.11.0 torchvision==0.26.0).
Names containing "table" get the table prompt, others the text prompt.
"""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/OXR-1.0"))
PROMPTS = {"table": "Recognize the table of the image into html format.", "text": "Recognize the content of the image."}
KEEP = (0, 13, 26)


def torch_model(dtype_name: str = "bfloat16"):
    import torch
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    OXR = get_class_from_dynamic_module("modeling_oxr.OXR", str(MODEL_PATH))
    config = sys.modules[OXR.__module__.rsplit(".", 1)[0] + ".configuration_oxr"].OXRConfig.from_pretrained(str(MODEL_PATH))
    return OXR, config, getattr(torch, dtype_name)


def reference(image_path: str, out: str, tokens: int) -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    from model import MAX_PIXELS, MIN_PIXELS

    OXR, config, dtype = torch_model()
    device = os.environ.get("OXR_REFERENCE_DEVICE", "cpu")
    model = OXR.from_pretrained(str(MODEL_PATH), config=config, dtype=dtype).to(device).eval()
    prompt = PROMPTS["table" if "table" in Path(image_path).stem else "text"]
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    ids, pv, grid = model.preprocess_inputs(messages, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    f32 = lambda t: t.detach().float().cpu().numpy()  # noqa: E731
    with torch.inference_mode():
        pvd, gd = pv.to(device, dtype), grid.to(device)
        hidden = model.visual_tokenizer.vit(pvd, gd, output_hidden_states=True, return_dict=True).hidden_states
        visual = model.vte(model.visual_tokenizer(pvd, gd))
        embeds = model.merge_multimodal(input_ids=ids.to(device), pixel_values=pvd, grid_thws=gd)
        logits = model.llm(inputs_embeds=embeds).logits[0, -1]
        answer = model.generate(ids.to(device), pixel_values=pvd, grid_thws=gd, max_new_tokens=tokens, do_sample=False,
                                eos_token_id=[151643, 151645], pad_token_id=model.text_tokenizer.pad_token_id, use_cache=True)
    np.savez(out, image=str(Path(image_path).resolve()), ids=ids[0].numpy(), pixels=pv.numpy(), grid=grid[0].numpy(), prompt=prompt,
             **{f"layer{i}": f32(hidden[i]) for i in KEEP}, visual=f32(visual), logits=f32(logits), answer=answer[0].cpu().numpy())
    print(f"{out}: {ids.shape[1]} prompt tokens, grid {grid[0].tolist()}, {answer.shape[1]} answer tokens")


def close(a, b) -> str:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return f"cos {cos:.7f}, max |diff| {np.abs(a - b).max():.4g} (max |ref| {np.abs(a).max():.4g})"


def compare(files: list[str]) -> None:
    import mlx.core as mx

    from model import OXR, STRIDE

    model = OXR(MODEL_PATH)
    for f in files:
        ref = np.load(f)
        print(f"== {Path(f).stem}")
        prompt = str(ref["prompt"])
        patches, h, w = model.preprocess(Image.open(str(ref["image"])))
        grid_same = [h, w] == ref["grid"][1:].tolist()
        print(f"  preprocessing: grid {'same' if grid_same else 'DIFFERENT'} {[h, w]}, pixels {close(ref['pixels'], patches) if grid_same else '-'}")
        patches = ref["pixels"]  # from here on the reference's own inputs, so each step is checked alone
        h, w = (int(x) for x in ref["grid"][1:])
        ids = model.input_ids(prompt, h * w // (STRIDE * STRIDE))
        print(f"  prompt ids: {'same' if ids == ref['ids'].tolist() else 'DIFFERENT'} ({len(ids)})")
        _, kept = model.vision(mx.array(patches).astype(mx.bfloat16), h, w, keep=KEEP)
        for i, k in zip(KEEP, kept):
            print(f"  vision layer {i:2}: {close(ref[f'layer{i}'], np.array(k.astype(mx.float32)))}")
        visual = model.visual_embeddings(patches, h, w)
        print(f"  visual embeddings: {close(ref['visual'], np.array(visual.astype(mx.float32)))}")
        embeds = model.embeddings(ids, visual)
        logits = np.array(model.llm(mx.zeros((1, len(ids)), dtype=mx.int32), input_embeddings=embeds[None])[0, -1].astype(mx.float32))
        print(f"  first logits: {close(ref['logits'], logits)}; top token {int(ref['logits'].argmax())} vs {int(logits.argmax())}")
        expected = ref["answer"].tolist()
        answer, _, _ = model.generate_from(patches, h, w, prompt, max_tokens=len(expected))
        same = next((i for i, (a, b) in enumerate(zip(expected, answer)) if a != b), min(len(expected), len(answer)))
        print(f"  greedy answer: {same} of {len(expected)} tokens the same"
              + ("" if same == len(expected) == len(answer) else
                 f"\n    PyTorch: {model.tokenizer.decode(expected[max(0, same - 8):same + 12])!r}"
                 f"\n    MLX:     {model.tokenizer.decode(answer[max(0, same - 8):same + 12])!r}"))


def vision32(file: str) -> None:
    """The vision encoder in float32 on both sides, in one process: the port should match to rounding."""
    import mlx.core as mx
    import torch
    from safetensors import safe_open

    from model import Vision

    ref = np.load(file)
    pixels, grid = ref["pixels"], ref["grid"]
    OXR, config, _ = torch_model()
    module = sys.modules[OXR.__module__]
    vit_config = config.vit_config
    vit_config._attn_implementation = "eager"
    prefix = "visual_tokenizer.vit."
    vit = module.Siglip2NavitModel(vit_config).float().eval()
    with safe_open(str(MODEL_PATH / "model.safetensors"), "pt") as f:
        vit.load_state_dict({k[len(prefix):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(prefix)}, strict=True)
    with torch.inference_mode():
        expected = vit(torch.tensor(pixels).float(), torch.tensor(grid)[None], output_hidden_states=True, return_dict=True).hidden_states[-1].numpy()
    del vit

    weights = mx.load(str(MODEL_PATH / "model.safetensors"))
    p = prefix + "vision_model."
    v = {k[len(p):]: weights[k].astype(mx.float32) for k in weights if k.startswith(p)}
    v = {k.replace("embeddings.", "").replace("encoder.", "").replace("position_embedding.weight", "position_embedding"): a
         for k, a in v.items() if not k.startswith("post_layernorm")}
    v["patch_embedding.weight"] = v["patch_embedding.weight"].reshape(vit_config.hidden_size, -1)
    vision = Vision(vit_config.to_dict())
    vision.load_weights(list(v.items()))
    x, _ = vision(mx.array(pixels).astype(mx.float32), int(grid[1]), int(grid[2]))
    print(f"float32 vision output: {close(expected, np.array(x))}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "reference":
        reference(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 300)
    elif command == "compare":
        compare(sys.argv[2:])
    elif command == "vision32":
        vision32(sys.argv[2])
    else:
        sys.exit(__doc__)
