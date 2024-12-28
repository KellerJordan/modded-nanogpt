import torch
import pandas as pd
import numpy as np
from transformers import EsmTokenizer, EsmForMaskedLM, AutoModelForMaskedLM
from datasets import Dataset
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
from utils import ProteinMasker


def main():
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
        total_tokens = sum(len(seq) for seq in sequences)
        print(f"Total tokens: {total_tokens}")

        # Load the ESM tokenizer and model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_names = {
            'facebook/esm2_t6_8M_UR50D': 'ESM2-8M',
            'facebook/esm2_t12_35M_UR50D': 'ESM2-35M',
            'Synthyra/ESMplusplus_small': 'ESMC-300M',
            'facebook/esm2_t30_150M_UR50D': 'ESM2-150M',
            'Synthyra/ESMplusplus_large': 'ESMC-600M',
            'facebook/esm2_t33_650M_UR50D': 'ESM2-650M'
        }

        mask_rate, batch_size = 0.15, 4
        results = []
        for model_name, nickname in model_names.items():
            torch.manual_seed(42)
            if 'synthyra' in model_name.lower():
                model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to(device).eval()
                tokenizer = model.tokenizer
            else:
                model = EsmForMaskedLM.from_pretrained(model_name).to(device).eval()
                tokenizer = EsmTokenizer.from_pretrained(model_name)
            masker = ProteinMasker(tokenizer, mask_rate)
            total_loss, count = 0.0, 0
            all_true, all_pred = [], []
            
            with torch.no_grad():
                for i in tqdm(range(0, len(sequences), batch_size), desc=f'Evaluating {model_name}'):
                    batch = sequences[i:i+batch_size]
                    encoding = tokenizer(batch, return_tensors='pt', padding=True, add_special_tokens=True, truncation=True, max_length=1024)
                    input_ids, attention_mask = encoding['input_ids'].to(device), encoding['attention_mask'].to(device)
                    masked_input_ids, labels = masker(input_ids)

                    out = model(input_ids=masked_input_ids, attention_mask=attention_mask, labels=labels)
                    logits, loss = out.logits, out.loss
                    
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
            
            print(f'Results for {nickname}:')
            print(f'Loss: {average_loss}')
            print(f'Perplexity: {perplexity:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1: {f1:.4f}')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'MCC: {mcc:.4f}')
            
            results.append({
                'model': nickname,
                'loss': round(average_loss, 4),
                'perplexity': round(perplexity, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'accuracy': round(accuracy, 4),
                'mcc': round(mcc, 4)
            })
            
            model.cpu()
            del model
            torch.cuda.empty_cache()

        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'esm2_benchmark_result_{type}.csv', index=False)


if __name__ == '__main__':
    main()
