import json
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

MODEL_NAME = "google/flan-t5-base"
INPUT_PATH = "/home/ap2194.UNT/micro/data/selected_subquestions.jsonl"
PROMPT_PATH = "/home/ap2194.UNT/micro/prompts/prompt.txt"
OUTPUT_DIR = "/home/ap2194.UNT/micro/data/flan_t5_micro"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def preprocess_function(examples, tokenizer, prompt_template, max_input_length=512, max_target_length=256):
    inputs = [
        prompt_template.format(question=q)
        for q in examples["raw_text"]
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=max_target_length,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    rows = load_jsonl(INPUT_PATH)
    prompt_template = load_prompt(PROMPT_PATH)

    print(f"Loaded {len(rows)} examples")

    if len(rows) == 0:
        raise ValueError("No training examples found.")

    dataset = Dataset.from_list(rows)

    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, prompt_template),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    tokenized_eval = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, prompt_template),
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        save_strategy="epoch",
        #eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=10,
        predict_with_generate=True,
        logging_dir=str(Path(OUTPUT_DIR) / "logs"),
        logging_steps=20,
        fp16=False,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()