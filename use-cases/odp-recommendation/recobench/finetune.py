"""
recobench.finetune  --  Phase 5: domain-adapt the embedder on the reuse signal.

The pretrained sentence encoder (all-MiniLM-L6-v2) knows general English, not
which ODPs an ontology tends to reuse. This module fine-tunes it with a
contrastive objective built purely from the TRAINING ontologies: an ontology's
term bag (anchor) is pulled towards the document of each pattern it reused
(positive), with the other patterns in the batch acting as in-batch negatives
(MultipleNegativesRankingLoss). This is the supervised "learn from existing
implementations" idea the 2019 ODPReco paper proposed but never built, realised
as a cheap, CPU-only embedding adaptation.

The tuned model is saved to disk and used through the ordinary
``Embedder('st:<path>')`` spec, so every embedding backend (item retrieval and
CF) can be re-evaluated with it. Training ontologies only; the held-out test
ontologies are never seen during fine-tuning.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .core import Ontology
from .content import build_pattern_docs

log = logging.getLogger("recobench.finetune")


def make_pairs(train: list[Ontology]) -> list[dict]:
    """(anchor = ontology text, positive = reused-pattern document) pairs."""
    pdocs = build_pattern_docs(train)
    pairs = []
    for o in train:
        anchor = o.text or o.name
        for stem in o.gt_stems:
            doc = pdocs.get(stem)
            if doc and anchor:
                pairs.append({"anchor": anchor, "positive": doc})
    return pairs


def finetune_embedder(train: list[Ontology], out_dir: Path,
                      base: str = "all-MiniLM-L6-v2", epochs: int = 3,
                      batch_size: int = 16, seed: int = 1) -> Path:
    """Fine-tune `base` on the reuse pairs from `train`; save to `out_dir`."""
    from sentence_transformers import SentenceTransformer, losses
    try:
        from datasets import Dataset
        from sentence_transformers import (SentenceTransformerTrainer,
                                           SentenceTransformerTrainingArguments)
        have_trainer = True
    except Exception:
        have_trainer = False

    pairs = make_pairs(train)
    if not pairs:
        raise RuntimeError("no reuse pairs to train on")
    log.info("fine-tuning %s on %d (ontology, pattern) pairs", base, len(pairs))

    model = SentenceTransformer(base)
    loss = losses.MultipleNegativesRankingLoss(model)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if have_trainer:
        ds = Dataset.from_list(pairs)
        args = SentenceTransformerTrainingArguments(
            output_dir=str(out_dir / "_ckpt"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            warmup_ratio=0.1,
            learning_rate=2e-5,
            fp16=False, bf16=False,
            seed=seed,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            dataloader_drop_last=False,
        )
        trainer = SentenceTransformerTrainer(
            model=model, args=args, train_dataset=ds, loss=loss)
        trainer.train()
    else:  # older sentence-transformers fallback
        from torch.utils.data import DataLoader
        from sentence_transformers import InputExample
        examples = [InputExample(texts=[p["anchor"], p["positive"]]) for p in pairs]
        dl = DataLoader(examples, shuffle=True, batch_size=batch_size)
        model.fit(train_objectives=[(dl, loss)], epochs=epochs,
                  warmup_steps=max(1, len(dl) // 10), show_progress_bar=False)

    model.save(str(out_dir))
    log.info("saved fine-tuned embedder to %s", out_dir)
    return out_dir
