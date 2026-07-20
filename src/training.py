import json
import re
from pathlib import Path

import numpy as np
import evaluate
import nltk
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_NAME = "google/flan-t5-base"
INPUT_PATH = "/home/ap2194.UNT/micro/data/selected_subquestions.jsonl"
OUTPUT_DIR = "/home/ap2194.UNT/micro/data/flan_t5_micro_grouped"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256
TRAIN_TEST_SPLIT = 0.1
SEED = 42

LEARNING_RATE = 3e-5
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 7
LOGGING_STEPS = 20

GENERATION_MAX_LENGTH = 128
NUM_BEAMS = 4

nltk.download("punkt", quiet=True)

rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("sacrebleu")


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rows.append(row)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e
    return rows


def validate_rows(rows):
    if not rows:
        raise ValueError("No training examples found.")

    required_keys = {"input_text", "target_text"}
    for i, row in enumerate(rows):
        missing = required_keys - row.keys()
        if missing:
            raise ValueError(
                f"Row {i} is missing required keys: {missing}. "
                f"Each row must contain at least 'input_text' and 'target_text'."
            )

        if not isinstance(row["input_text"], str) or not row["input_text"].strip():
            raise ValueError(f"Row {i} has invalid or empty 'input_text'.")

        if not isinstance(row["target_text"], str) or not row["target_text"].strip():
            raise ValueError(f"Row {i} has invalid or empty 'target_text'.")


def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=256):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=max_target_length,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_numbered_lines(text: str):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned = []
    for line in lines:
        line = re.sub(r"^\d+\.\s*", "", line).strip()
        if line:
            cleaned.append(line)
    return cleaned


def is_numbered_list(text: str) -> int:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return 0
    matches = sum(bool(re.match(r"^\d+\.\s+", line)) for line in lines)
    return 1 if matches == len(lines) else 0


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [normalize_text(pred) for pred in decoded_preds]
    decoded_labels = [normalize_text(label) for label in decoded_labels]

    rouge_result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    bleu_refs = [[label] for label in decoded_labels]
    bleu_result = bleu_metric.compute(
        predictions=decoded_preds,
        references=bleu_refs,
    )

    pred_counts = [len(split_numbered_lines(pred)) for pred in decoded_preds]
    ref_counts = [len(split_numbered_lines(label)) for label in decoded_labels]

    avg_generated_count = float(np.mean(pred_counts)) if pred_counts else 0.0
    avg_reference_count = float(np.mean(ref_counts)) if ref_counts else 0.0

    count_match_rate = (
        float(np.mean([int(p == r) for p, r in zip(pred_counts, ref_counts)]))
        if pred_counts else 0.0
    )

    within_range_rate = (
        float(np.mean([int(3 <= p <= 5) for p in pred_counts]))
        if pred_counts else 0.0
    )

    format_rate = (
        float(np.mean([is_numbered_list(pred) for pred in decoded_preds]))
        if decoded_preds else 0.0
    )

    exact_match_rate = (
        float(np.mean([int(pred == label) for pred, label in zip(decoded_preds, decoded_labels)]))
        if decoded_preds else 0.0
    )

    return {
        "rouge1": round(rouge_result["rouge1"], 4),
        "rouge2": round(rouge_result["rouge2"], 4),
        "rougeL": round(rouge_result["rougeL"], 4),
        "sacrebleu": round(bleu_result["score"], 4),
        "avg_generated_count": round(avg_generated_count, 4),
        "avg_reference_count": round(avg_reference_count, 4),
        "count_match_rate": round(count_match_rate, 4),
        "within_range_rate": round(within_range_rate, 4),
        "format_rate": round(format_rate, 4),
        "exact_match_rate": round(exact_match_rate, 4),
    }



def print_sample_predictions(trainer, tokenizer, eval_dataset, raw_eval_rows, n=5):
    print("\n" + "=" * 80)
    print(f"SAMPLE PREDICTIONS (showing {min(n, len(raw_eval_rows))})")
    print("=" * 80)

    for i in range(min(n, len(raw_eval_rows))):
        row = raw_eval_rows[i]
        prompt = row["input_text"]
        reference = row["target_text"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )

        inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}

        outputs = trainer.model.generate(
            **inputs,
            max_new_tokens=GENERATION_MAX_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    rows = load_jsonl(INPUT_PATH)
    validate_rows(rows)

    print(f"Loaded {len(rows)} examples from: {INPUT_PATH}")

    dataset = Dataset.from_list(rows)

    split_dataset = dataset.train_test_split(
        test_size=TRAIN_TEST_SPLIT,
        seed=SEED,
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Train size: {len(train_dataset)}")
    print(f"Eval size : {len(eval_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(
            x,
            tokenizer,
            max_input_length=MAX_INPUT_LENGTH,
            max_target_length=MAX_TARGET_LENGTH,
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_eval = eval_dataset.map(
        lambda x: preprocess_function(
            x,
            tokenizer,
            max_input_length=MAX_INPUT_LENGTH,
            max_target_length=MAX_TARGET_LENGTH,
        ),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=2,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        generation_max_length=GENERATION_MAX_LENGTH,
        logging_dir=str(Path(OUTPUT_DIR) / "logs"),
        logging_steps=LOGGING_STEPS,
        fp16=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    trainer.train()

    print("\nRunning final evaluation on validation set...")
    metrics = trainer.evaluate()
    print("\nFINAL EVAL METRICS:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to: {OUTPUT_DIR}")

    raw_eval_rows = [eval_dataset[i] for i in range(len(eval_dataset))]
    print_sample_predictions(trainer, tokenizer, tokenized_eval, raw_eval_rows, n=5)


if __name__ == "__main__":
    main()