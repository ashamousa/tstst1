"""Training script for fine-tuning a Hugging Face text classification model on a JSONL dataset.

The dataset must contain two fields:
- ``text``: the raw text input.
- ``labels``: an integer that represents the class id.

Usage example:
    python train.py \
        --model-name-or-path bert-base-uncased \
        --train-file path/to/train.jsonl \
        --validation-file path/to/valid.jsonl \
        --output-dir ./model-output

This script relies on ``transformers`` and ``datasets`` packages. Install them via:
    pip install transformers datasets evaluate
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import evaluate
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """Command line arguments for the training script."""

    model_name_or_path: str
    train_file: Path
    validation_file: Optional[Path]
    output_dir: Path
    max_length: int = 512
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    num_train_epochs: int = 3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    logging_steps: int = 50
    eval_steps: Optional[int] = None
    seed: int = 42


def parse_args() -> ScriptArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True, help="Model checkpoint to fine-tune.")
    parser.add_argument(
        "--train-file",
        type=Path,
        required=True,
        help="Path to the training dataset in JSONL format.",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=None,
        help="Optional path to the validation dataset in JSONL format.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store checkpoints and logs.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for tokenization.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8,
        help="Batch size per device for training.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=8,
        help="Batch size per device for evaluation.",
    )
    parser.add_argument("--num-train-epochs", type=int, default=3, help="Number of epochs to train.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="Warmup ratio for the scheduler.")
    parser.add_argument("--logging-steps", type=int, default=50, help="Log metrics every N steps.")
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Evaluate every N steps. Defaults to epoch-based evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    return ScriptArguments(**vars(args))


def load_json_dataset(train_path: Path, validation_path: Optional[Path]) -> DatasetDict:
    data_files = {"train": str(train_path)}
    if validation_path is not None:
        data_files["validation"] = str(validation_path)

    dataset = load_dataset("json", data_files=data_files)
    return dataset


def prepare_label_mapping(dataset: DatasetDict) -> tuple[DatasetDict, dict[int, str], dict[str, int]]:
    """Ensure labels are consecutive integers and return mapping dictionaries."""

    unique_labels = sorted(set(dataset["train"]["labels"]))
    id2label = {idx: str(label) for idx, label in enumerate(unique_labels)}
    label2id = {label: idx for idx, label in id2label.items()}

    def _map_labels(example):
        example["labels"] = label2id[str(example["labels"])]
        return example

    dataset = dataset.map(_map_labels)
    return dataset, id2label, label2id


def tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer, max_length: int) -> DatasetDict:
    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = examples["labels"]
        return tokenized

    return dataset.map(preprocess_function, batched=True, remove_columns=["text"])


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Loading dataset...")
    raw_dataset = load_json_dataset(args.train_file, args.validation_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Prepare label mapping before tokenization.
    processed_dataset, id2label, label2id = prepare_label_mapping(raw_dataset)
    num_labels = len(id2label)

    tokenized_dataset = tokenize_dataset(processed_dataset, tokenizer, args.max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = predictions.argmax(axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    has_validation = args.validation_file is not None
    evaluation_strategy = "steps" if (has_validation and args.eval_steps) else ("epoch" if has_validation else "no")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        evaluation_strategy=evaluation_strategy,
        save_strategy="epoch",
        eval_steps=args.eval_steps if has_validation else None,
        load_best_model_at_end=has_validation,
        metric_for_best_model="accuracy" if has_validation else None,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
