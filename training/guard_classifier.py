# %% [markdown]
# # NovaTech guard classifier: LoRA fine-tune of DeBERTa-v3-small
#
# Labels every user message as **blocked** (attack on the assistant), **smalltalk**, or **task**
# before any LLM call.
#
# **On Colab:** Runtime → Change runtime type → T4 GPU, then Run all. When asked, upload
# `train_pool.jsonl` and `test_handwritten.jsonl` from `backend/data/guard/`.
# At the end, `guard-lora.zip` downloads: unzip it into `backend/models/guard-lora/`.
#
# **Locally (smoke test):** `SMOKE=1 python training/guard_classifier.py` trains on a tiny
# subset for one epoch, just to prove the code runs end to end. `SMOKE_N=400 SMOKE_EPOCHS=3`
# additionally checks that it actually learns (about 10 minutes on CPU).

# %%
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    os.system('pip install -q "transformers>=5,<6" "peft>=0.21" sentencepiece')
    # Colab's preinstalled torchao is too old for peft and isn't needed.
    os.system("pip uninstall -y -q torchao")

SMOKE = os.environ.get("SMOKE") == "1"
SMOKE_N = int(os.environ.get("SMOKE_N", 64))          # training rows in a smoke test
SMOKE_EPOCHS = int(os.environ.get("SMOKE_EPOCHS", 1))
BASE_MODEL = "microsoft/deberta-v3-small"
LABELS = ["blocked", "smalltalk", "task"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
MAX_LEN = 128        # attack messages can be long, but the signal is almost always early
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| smoke test:", SMOKE)

# %% [markdown]
# ## 1. Data
# * `train_pool.jsonl`: synthetic examples that passed a second, independent labelling pass.
# * Real human-written injections from `deepset/prompt-injections` (Apache-2.0), added as
#   **blocked**. Its benign rows are general web questions, not NovaTech traffic, so they're skipped.
# * `test_handwritten.jsonl`: written by hand, never trained on or used for model selection.

# %%
def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

if IN_COLAB:
    from google.colab import files
    uploaded = files.upload()   # choose train_pool.jsonl and test_handwritten.jsonl
    data_dir = Path(".")
else:
    data_dir = Path(__file__).resolve().parent.parent / "backend" / "data" / "guard"

pool = read_jsonl(data_dir / "train_pool.jsonl")
test = read_jsonl(data_dir / "test_handwritten.jsonl")

from datasets import load_dataset

deepset = load_dataset("deepset/prompt-injections")
real_attacks = [{"text": r["text"], "label": "blocked", "source": "deepset"}
                for r in deepset["train"] if r["label"] == 1]
deepset_test_attacks = [r["text"] for r in deepset["test"] if r["label"] == 1]
print(f"synthetic pool {len(pool)} | deepset train attacks {len(real_attacks)} | "
      f"deepset test attacks {len(deepset_test_attacks)} | handwritten test {len(test)}")

# %%
from sklearn.model_selection import train_test_split

examples = pool + real_attacks
train_rows, val_rows = train_test_split(examples, test_size=0.15, random_state=SEED,
                                        stratify=[r["label"] for r in examples])
if SMOKE:
    train_rows, val_rows = train_rows[:SMOKE_N], val_rows[:64]

for name, rows in [("train", train_rows), ("val", val_rows), ("test", test)]:
    counts = {label: sum(r["label"] == label for r in rows) for label in LABELS}
    print(f"{name:5} {len(rows):5}  {counts}")

# %% [markdown]
# ## 2. Model: DeBERTa-v3-small + LoRA
# LoRA trains small low-rank updates to the attention and dense layers (about 1–2% of the
# weights) plus the new classification head. The saved adapter is a few MB, and the base
# model downloads like the embedding models do.

# %%
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
    dtype=torch.float32)   # the checkpoint is fp16; DeBERTa-v3 trains unstably in fp16

lora = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["query_proj", "key_proj", "value_proj", "dense"],
    modules_to_save=["classifier", "pooler"],   # new layers: trained fully, saved with the adapter
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# %%
from datasets import Dataset

def encode(rows):
    ds = Dataset.from_dict({"text": [r["text"] for r in rows],
                            "labels": [LABEL2ID[r["label"]] for r in rows]})
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN), batched=True)

train_ds, val_ds = encode(train_rows), encode(val_rows)

# %% [markdown]
# ## 3. Training
# Class-weighted loss: smalltalk has the fewest examples, and without weights the model can
# score well overall while mislabelling it. Model selection uses **macro-F1** on validation,
# which weighs every class equally.

# %%
from sklearn.metrics import f1_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

counts = np.bincount([LABEL2ID[r["label"]] for r in train_rows], minlength=len(LABELS))
class_weights = torch.tensor(len(train_rows) / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float)
print("class weights:", dict(zip(LABELS, class_weights.round(decimals=2).tolist())))

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels,
                                                 weight=class_weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    return {"macro_f1": f1_score(pred.label_ids, pred.predictions.argmax(-1), average="macro")}

args = TrainingArguments(
    output_dir="guard-out",
    num_train_epochs=SMOKE_EPOCHS if SMOKE else 6,
    learning_rate=2e-4,              # LoRA uses a higher LR than full fine-tuning
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    warmup_steps=0.1,                # transformers 5: a value < 1 is a ratio of total steps
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=20,
    fp16=False,                      # DeBERTa-v3 is unstable in fp16; fp32 is fast enough here
    report_to=[],
    seed=SEED,
)
trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                          data_collator=DataCollatorWithPadding(tokenizer),
                          compute_metrics=compute_metrics)
start = time.time()
trainer.train()
print(f"training took {time.time() - start:.0f}s")

# %% [markdown]
# ## 4. Evaluation on the handwritten test set
# The two errors don't cost the same:
# * **Missed attack** (blocked → task): other layers still hold (permission gate, ingest quarantine,
#   output guard), but one layer of defence is gone.
# * **False alarm** (task → blocked): a real customer is refused. Visible and annoying.

# %%
from sklearn.metrics import classification_report, confusion_matrix

model.eval()

@torch.no_grad()
def predict(texts, batch_size=32):
    preds, probs = [], []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(model.device)
        p = torch.softmax(model(**enc).logits, dim=-1).cpu()
        preds += p.argmax(-1).tolist()
        probs += p.max(-1).values.tolist()
    return [LABELS[i] for i in preds], probs

test_texts = [r["text"] for r in test]
y_true = [r["label"] for r in test]
y_pred, y_prob = predict(test_texts)

print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))
print("confusion matrix (rows = true, columns = predicted):", LABELS)
print(confusion_matrix(y_true, y_pred, labels=LABELS))

print("\nErrors:")
for r, p, conf in zip(test, y_pred, y_prob):
    if p != r["label"]:
        print(f"  {r['label']:9} → {p:9} ({conf:.2f})  [{r.get('note', '')}]  {r['text'][:90]}")

deepset_pred, _ = predict(deepset_test_attacks)
recall = sum(p == "blocked" for p in deepset_pred) / len(deepset_test_attacks)
print(f"\nRecall on deepset's held-out real attacks: {recall:.1%} ({len(deepset_test_attacks)} attacks)")

# %% [markdown]
# ## 5. CPU latency (what the guard node will see) and export

# %%
cpu_model = model.to("cpu")
sample = "Where is my order #1025? It was supposed to arrive yesterday."
with torch.no_grad():
    enc = tokenizer(sample, return_tensors="pt")
    cpu_model(**enc)   # warm-up
    t = time.perf_counter()
    for _ in range(20):
        cpu_model(**enc)
print(f"CPU latency: {(time.perf_counter() - t) / 20 * 1000:.1f} ms per message")

out_dir = Path("guard-lora")
model.save_pretrained(out_dir)       # adapter + classifier head only (a few MB)
tokenizer.save_pretrained(out_dir)
(out_dir / "guard_meta.json").write_text(json.dumps({
    "base_model": BASE_MODEL, "labels": LABELS, "max_len": MAX_LEN,
    "test_macro_f1": f1_score(y_true, y_pred, average="macro"),
    "deepset_attack_recall": recall,
}, indent=2))
print("saved:", sorted(p.name for p in out_dir.iterdir()))

if IN_COLAB:
    os.system("zip -qr guard-lora.zip guard-lora")
    files.download("guard-lora.zip")
