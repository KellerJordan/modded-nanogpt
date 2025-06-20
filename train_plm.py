#! /usr/bin/env python3
# py -m train_plm
import argparse
import torch
import torch.nn.functional as F
from torchinfo import summary
from transformers import TrainingArguments, EvalPrediction
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from huggingface_hub import login, hf_hub_download
from datasets import load_dataset, Dataset

from model.model import PLM
from data.dataset_classes import SequenceDatasetFromList, SequenceCollator
from custom_trainer import CustomTrainer


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

### Check for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


def compute_dsm_metrics(eval_preds: EvalPrediction):
    ### NOTE the eval mask percentage is fixed at 15%
    metrics = {}
    logits = eval_preds.predictions[0] if isinstance(eval_preds.predictions, tuple) else eval_preds.predictions
    labels = eval_preds.label_ids[0] if isinstance(eval_preds.label_ids, tuple) else eval_preds.label_ids
    # labels are already -100 for non-masked tokens
    logits_torch = torch.tensor(logits)
    labels_torch = torch.tensor(labels)
    # We need ot do this because the eval loss is scaled by the mask rate
    cross_entropy_loss = F.cross_entropy(
        logits_torch.view(-1, logits_torch.shape[-1]), 
        labels_torch.view(-1),
        ignore_index=-100
    )
    metrics['cross_entropy_loss'] = cross_entropy_loss
    y_pred = logits.argmax(axis=-1).flatten()
    y_true = labels.flatten()
    valid_indices = y_true != -100
    y_pred = y_pred[valid_indices]
    y_true = y_true[valid_indices]
    f1 = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    metrics["f1"] = f1
    metrics["prec"] = prec
    metrics["rec"] = rec
    metrics["acc"] = acc
    metrics["mcc"] = mcc
    return metrics


def get_eval_data():
    local_file = hf_hub_download(
        repo_id="Synthyra/omg_prot50",
        filename=f"data/valid-00000-of-00001.parquet",
        repo_type="dataset"
    )
    data = Dataset.from_parquet(local_file).shuffle(seed=42).select(range(1000))
    print(data)
    valid_seqs = data['sequence']
    local_file = hf_hub_download(
        repo_id="Synthyra/omg_prot50",
        filename=f"data/test-00000-of-00001.parquet",
        repo_type="dataset"
    )
    data = Dataset.from_parquet(local_file).shuffle(seed=42).select(range(1000))
    print(data)
    test_seqs = data['sequence']
    return valid_seqs, test_seqs


def parse_args():
    parser = argparse.ArgumentParser(description="Synthyra Trainer")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--model_path", type=str, default="Synthyra/ESM2-650M", help="Path to the model to train")
    parser.add_argument("--save_path", type=str, default="GleghornLab/DSM_650", help="Path to save the model and report to wandb")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of steps to train for")
    parser.add_argument("--wandb_project", type=str, default="DSM", help="Wandb project name")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length of sequences fed to the model")
    parser.add_argument("--save_every", type=int, default=1000, help="Save the model every n steps and evaluate every n/2 steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision for training")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    args = parser.parse_args()
    return args


def main(args):
    ### Load model
    model = PLM.from_pretrained(args.model_path)
    tokenizer = model.tokenizer
    summary(model)

    ### Load Dataset
    train_dataset = load_dataset("Synthyra/omg_prot50", split="train", streaming=True).shuffle(seed=42)
    valid_seqs, test_seqs = get_eval_data()
    if args.bugfix:
        valid_seqs = valid_seqs[:10]
        test_seqs = test_seqs[:10]
    
    valid_dataset = SequenceDatasetFromList(valid_seqs)
    test_dataset = SequenceDatasetFromList(test_seqs)
    data_collator = SequenceCollator(tokenizer, args.max_length)

    ### Define Training Arguments
    training_args = TrainingArguments(
        output_dir=args.save_path.split('/')[-1],
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=100,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=args.save_every,
        eval_steps=args.save_every,
        warmup_steps=args.save_every,
        logging_dir="./logs",
        learning_rate=args.lr,
        fp16=args.fp16,
        dataloader_num_workers=4 if not args.bugfix else 0,
        dataloader_prefetch_factor=10,
        report_to="wandb" if WANDB_AVAILABLE else 'none',
        save_total_limit=3,
        max_grad_norm=10.0,
        label_names=['input_ids'],
    )

    ### Create a trainer
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        training_args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_dsm_metrics,
    )

    ### Train
    metrics = trainer.evaluate(test_dataset)
    print('Initial Metrics: \n', metrics)
    trainer.train()
    metrics = trainer.evaluate(test_dataset)
    print('Final Metrics: \n', metrics)
    trainer.model.push_to_hub(args.save_path, private=True)
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    # py -m train_dsm
    args = parse_args()

    if WANDB_AVAILABLE:
        run_name = args.save_path.split('/')[-1]
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    if args.token is not None:
        login(args.token)    

    if args.bugfix:
        args.batch_size = 2
        args.max_length = 32
        args.save_every = 1000

    main(args)
