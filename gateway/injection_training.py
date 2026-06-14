#!/usr/bin/env python3
"""Fine-tune BERTimbau para detecção de prompt injection.

Uso:
    python -m gateway.injection_training

Carrega ``train/injection_dataset.jsonl``, fine-tuna
``neuralmind/bert-base-portuguese-cased`` com 2 labels (clean / injection)
e salva o modelo em ``models/injection_classifier/``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATASET_PATH = _PROJECT_ROOT / "train" / "injection_dataset.jsonl"
_OUTPUT_DIR = _PROJECT_ROOT / "models" / "injection_classifier"

# ---------------------------------------------------------------------------
# Hyperparams
# ---------------------------------------------------------------------------
_BASE_MODEL = "neuralmind/bert-base-portuguese-cased"
_EPOCHS = 3
_LR = 2e-5
_BATCH_SIZE = 16
_MAX_LEN = 256
_VAL_SPLIT = 0.2
_SEED = 42


def _load_dataset(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(int(obj["label"]))
    logger.info("Dataset carregado: %d exemplos (%d injection, %d clean)", len(texts), sum(labels), len(labels) - sum(labels))
    return texts, labels


def main() -> None:
    # Late imports — não pesa no gateway.
    try:
        import torch
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as e:
        logger.error("Dependência não instalada: %s. Instale com: pip install transformers torch scikit-learn", e)
        sys.exit(1)

    # -- dados ----------------------------------------------------------------
    texts, labels = _load_dataset(_DATASET_PATH)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=_VAL_SPLIT, random_state=_SEED, stratify=labels,
    )

    # -- tokenização ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)

    def encode(texts_: list[str]) -> tuple:
        enc = tokenizer(texts_, truncation=True, padding=True, max_length=_MAX_LEN, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    train_ids, train_masks = encode(train_texts)
    val_ids, val_masks = encode(val_texts)
    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long)

    train_ds = TensorDataset(train_ids, train_masks, train_labels_t)
    val_ds = TensorDataset(val_ids, val_masks, val_labels_t)
    train_dl = DataLoader(train_ds, batch_size=_BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=_BATCH_SIZE)

    # -- modelo ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(_BASE_MODEL, num_labels=2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=_LR)
    total_steps = len(train_dl) * _EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # -- treino ---------------------------------------------------------------
    for epoch in range(1, _EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            b_ids, b_masks, b_labels = (t.to(device) for t in batch)
            optimizer.zero_grad()
            outputs = model(b_ids, attention_mask=b_masks, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)
        logger.info("Epoch %d/%d — loss: %.4f", epoch, _EPOCHS, avg_loss)

    # -- validação ------------------------------------------------------------
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in val_dl:
            b_ids, b_masks, b_labels = (t.to(device) for t in batch)
            outputs = model(b_ids, attention_mask=b_masks)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(b_labels.cpu().tolist())

    acc = accuracy_score(all_true, all_preds)
    report = classification_report(all_true, all_preds, target_names=["clean", "injection"], digits=4)
    logger.info("Validation accuracy: %.4f", acc)
    print("\n" + report)

    # -- salva ----------------------------------------------------------------
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(_OUTPUT_DIR))
    tokenizer.save_pretrained(str(_OUTPUT_DIR))
    logger.info("Modelo salvo em %s", _OUTPUT_DIR)


if __name__ == "__main__":
    main()
