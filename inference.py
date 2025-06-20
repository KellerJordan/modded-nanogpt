import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
from datasets import Dataset
from functools import partial
from transformers import EsmTokenizer
from huggingface_hub import hf_hub_download
from model.model import ESM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Model path")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    return parser.parse_args()


def tokenize(seq, tokenizer):
    return tokenizer.encode(seq, add_special_tokens=True, truncation=True, max_length=1024)


def main(args):
    model = ESM.from_pretrained(args.model_path).eval().cuda()
    print(model)
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    tokenize_fn = partial(tokenize, tokenizer=tokenizer)
    batch_size = args.batch_size
    model_name = args.model_path.split('/')[-1]
    results = []

    for type in ['valid', 'test']:
        local_file = hf_hub_download(
            repo_id="Synthyra/omg_prot50",
            filename=f"data/{type}-00000-of-00001.parquet",
            repo_type="dataset"
        )
        data = Dataset.from_parquet(local_file)
        print(data)
        sequences = data['sequence']
        sequences = sorted(sequences, key=len, reverse=True)
        print(sequences[-1])
        total_tokens = sum(len(seq[:1022]) + 2 for seq in sequences)
        print(f"Total tokens: {total_tokens}")

        torch.manual_seed(42)
        total_loss, count = 0.0, 0
        all_true, all_pred = [], []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc=f'Evaluating {model_name}'):
                batch = sequences[i:i+batch_size]
                input_ids = []
                for seq in batch:
                    input_ids.extend(tokenize_fn(seq))
                input_ids = torch.tensor(input_ids).long().cuda()
                
                sliding_window_size = torch.tensor(1024, dtype=torch.int32, device='cuda')
                logits, loss, labels = model.inference(input_ids, sliding_window_size, 0.15)
                all_true.extend(labels.cpu().numpy().flatten())
                all_pred.extend(logits.argmax(dim=-1).cpu().numpy().flatten())
                total_loss += loss.item()
                count += 1

        average_loss = total_loss / count
        perplexity = torch.exp(torch.tensor(average_loss)).item()
        
        # Calculate metrics
        all_true = np.array(all_true).flatten()
        all_pred = np.array(all_pred).flatten()
        mask = (all_true != -100)
        all_true = all_true[mask]
        all_pred = all_pred[mask]
        precision = precision_score(all_true, all_pred, average='weighted')
        recall = recall_score(all_true, all_pred, average='weighted')
        f1 = f1_score(all_true, all_pred, average='weighted')
        accuracy = accuracy_score(all_true, all_pred)
        mcc = matthews_corrcoef(all_true, all_pred)
        
        print(f'Results for {model_name}:')
        print(f'Loss: {average_loss}')
        print(f'Perplexity: {perplexity:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1: {f1:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'MCC: {mcc:.4f}')
        
        results.append({
            'model': model_name,
            'loss': round(average_loss, 4),
            'perplexity': round(perplexity, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'accuracy': round(accuracy, 4),
            'mcc': round(mcc, 4)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'speedrun_{model_name}.csv', index=False)


if __name__ == '__main__':
    main(parse_args())
