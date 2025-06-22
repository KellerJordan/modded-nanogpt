#! /usr/bin/env python3
# py -m train_plm
import argparse
import torch
import torch.nn.functional as F
from torchinfo import summary
from transformers import TrainingArguments, EvalPrediction, Trainer
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from huggingface_hub import login, hf_hub_download
from datasets import load_dataset, Dataset

from model.model import PLM, PLMConfig
from data.dataset_classes import SequenceDatasetFromList, TokenBasedSequenceCollator, TokenBasedIterableDataset
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
    lm_logits = eval_preds.predictions[0] if isinstance(eval_preds.predictions, tuple) else eval_preds.predictions
    input_ids = eval_preds.label_ids[0] if isinstance(eval_preds.label_ids, tuple) else eval_preds.label_ids
    lm_logits, labels = lm_logits

    # labels are already -100 for non-masked tokens
    lm_logits_torch = torch.tensor(lm_logits)
    labels_torch = torch.tensor(labels)
    # We need ot do this because the eval loss is scaled by the mask rate
    cross_entropy_loss = F.cross_entropy(
        lm_logits_torch.view(-1, lm_logits_torch.shape[-1]), 
        labels_torch.view(-1),
        ignore_index=-100
    )

    metrics['cross_entropy_loss'] = cross_entropy_loss

    y_pred = lm_logits.argmax(axis=-1).flatten()
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
    parser.add_argument("--wandb_token", type=str, default=None, help="Wandb token")
    parser.add_argument("--save_path", type=str, default="Synthyra/speedrun_test", help="Path to save the model and report to wandb")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of steps to train for")
    parser.add_argument("--wandb_project", type=str, default="SpeedrunPLM", help="Wandb project name")
    parser.add_argument("--save_every", type=int, default=1000, help="Save the model every n steps and evaluate every n/2 steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision for training")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    parser.add_argument("--p_attention", action="store_true", help="Use PAttention")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--n_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--num_att_tokens", type=int, default=512, help="Number of attention tokens")
    parser.add_argument("--expansion_ratio", type=float, default=8/3, help="Expansion ratio for MLP")
    parser.add_argument("--soft_logit_cap", type=float, default=16.0, help="Soft logit cap")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="Number of hidden layers")
    parser.add_argument("--sliding_window_size", type=int, default=512, help="Sliding window size for PAttention")
    parser.add_argument("--target_token_count", type=int, default=64*1024, help="Target token count for training")
    parser.add_argument("--disable_muon", action="store_true", help="Disable Muon optimizer")
    parser.add_argument("--unet", action="store_true", help="Use UNet")
    args = parser.parse_args()
    return args


def main(args):
    ### Load model
    config = PLMConfig(
        p_attention=args.p_attention,
        hidden_size=args.hidden_size,
        num_attention_heads=args.n_heads,
        num_att_tokens=args.num_att_tokens,
        expansion_ratio=args.expansion_ratio,
        soft_logit_cap=args.soft_logit_cap,
        num_hidden_layers=args.num_hidden_layers,
        sliding_window_size=args.sliding_window_size,
        unet=args.unet,
    )
    model = PLM(config)
    tokenizer = model.tokenizer
    summary(model)

    ### Load Dataset
    train_dataset = load_dataset("Synthyra/omg_prot50", split="train", streaming=True).shuffle(seed=42)
    valid_seqs, test_seqs = get_eval_data()
    if args.bugfix:
        valid_seqs = valid_seqs[:10]
        test_seqs = test_seqs[:10]
    
    train_dataset = TokenBasedIterableDataset(train_dataset, target_token_count=args.target_token_count, col_name='sequence')
    valid_dataset = SequenceDatasetFromList(valid_seqs)
    test_dataset = SequenceDatasetFromList(test_seqs)
    data_collator = TokenBasedSequenceCollator(tokenizer)

    ### Define Training Arguments
    training_args = TrainingArguments(
        output_dir=args.save_path.split('/')[-1],
        overwrite_output_dir=True,
        per_device_train_batch_size=1,  # Each item from TokenBasedIterableDataset is already a complete batch
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
        dataloader_prefetch_factor=10 if not args.bugfix else None,
        report_to="wandb" if WANDB_AVAILABLE else 'none',
        save_total_limit=3,
        max_grad_norm=10.0,
        label_names=['input_ids'],
    )

    ### Create a trainer
    if args.disable_muon:
        trainer_cls = Trainer
    else:
        trainer_cls = CustomTrainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
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
        if args.wandb_token is not None:
            wandb.login(key=args.wandb_token)
            run_name = args.save_path.split('/')[-1]
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        else:
            print("Wandb token is not provided, skipping wandb")
            WANDB_AVAILABLE = False

    if args.token is not None:
        login(args.token)    

    if args.bugfix:
        args.batch_size = 2
        args.p_attention = False
        args.hidden_size = 32
        args.n_heads = 2
        args.num_att_tokens = 32
        args.expansion_ratio = 2.0
        args.soft_logit_cap = 16.0
        args.num_hidden_layers = 1
        args.sliding_window_size = 32
        args.save_every = 1000

    main(args)
