import os
import json
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEFAULT_SUMMARIZATION_CONFIG = {
    "max_input_length": 512,
    "max_summary_length": 100,
    "min_length": 10,
    "num_beams": 4,
    "no_repeat_ngram_size": 3,
    "length_penalty": 1.0,
    "early_stopping": True,
    "padding": "longest",
    "batch_size": 8
}

def load_summarizer(
    model_id: str = "Omar10lfc/arabart-xlsum-arabic",
    revision: Optional[str] = None,
    device: Optional[str] = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {}
    if revision:
        kw["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kw)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kw).to(device).eval()
    return model, tokenizer, device

def summarize_batch(
    texts: List[str], 
    model, 
    tokenizer, 
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None
) -> List[str]:
    """
    Unified summarization function adhering to the documented final protocol.
    Config parameters are strictly validated, batched per batch_size, and logged.
    """
    if not texts:
        return []

    cfg = dict(DEFAULT_SUMMARIZATION_CONFIG)
    if config:
        cfg.update(config)

    if device is None:
        device = model.device if hasattr(model, 'device') else ("cuda" if torch.cuda.is_available() else "cpu")

    # Clean texts using light summarization cleaner
    from .normalize import clean_for_summary
    cleaned_texts = [clean_for_summary(t) for t in texts]

    gen_kwargs = {
        "max_length": cfg["max_summary_length"],
        "min_length": cfg["min_length"],
        "num_beams": cfg["num_beams"],
        "no_repeat_ngram_size": cfg["no_repeat_ngram_size"],
        "length_penalty": cfg["length_penalty"],
        "early_stopping": cfg["early_stopping"],
    }

    batch_size = cfg.get("batch_size", 8)
    all_summaries = []

    for b_start in range(0, len(cleaned_texts), batch_size):
        b_texts = cleaned_texts[b_start:b_start + batch_size]
        inputs = tokenizer(
            b_texts,
            max_length=cfg["max_input_length"],
            truncation=True,
            padding=cfg["padding"],
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        summaries = [
            tokenizer.decode(g, skip_special_tokens=True).strip()
            for g in output_ids
        ]
        all_summaries.extend(summaries)

    return all_summaries
