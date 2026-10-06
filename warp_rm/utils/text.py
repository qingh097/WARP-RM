"""Frozen CLIP text embeddings for language conditioning (one per task string)."""
from __future__ import annotations
import numpy as np, torch

_CACHE: dict = {}

def _clip_text_embeds(clip, out):
    """transformers<5 returns a tensor; >=5 returns BaseModelOutputWithPooling."""
    import torch as _torch
    if _torch.is_tensor(out):
        return out
    te = getattr(out, "text_embeds", None)
    if te is not None:
        return te
    return clip.text_projection(out.pooler_output)


def embed_tasks(tasks: list[str], device, model_name: str = "openai/clip-vit-base-patch16") -> dict[str, np.ndarray]:
    """Return {task: unit-norm float32 (512,)} using the frozen CLIP text tower."""
    from transformers import CLIPModel, CLIPTokenizer
    todo = sorted({t or "" for t in tasks} - set(_CACHE))
    if todo:
        tok = CLIPTokenizer.from_pretrained(model_name)
        clip = CLIPModel.from_pretrained(model_name).to(device).eval()
        with torch.no_grad():
            for i in range(0, len(todo), 64):
                b = todo[i:i + 64]
                t = tok(b, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
                e = torch.nn.functional.normalize(_clip_text_embeds(clip, clip.get_text_features(**t)).float(), dim=-1).cpu().numpy()
                for s, v in zip(b, e):
                    _CACHE[s] = v.astype(np.float32)
        del clip
    return {t or "": _CACHE[t or ""] for t in tasks}
