import torch
import random
import pandas as pd
from transformers import EsmTokenizer, EsmForMaskedLM
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef


# Set a fixed seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


local_file = hf_hub_download(
    repo_id="Synthyra/omg_prot50",
    filename="data/test-00000-of-00001.parquet",
    repo_type="dataset"
)
local_file = local_file.replace('\\', '/').split('/data')[0]
print(local_file)
data = load_dataset(local_file, split='test').remove_columns('__index_level_0__')
print(data)
sequences = data['sequence']
print(sequences[0])
sequences = sorted(sequences, key=len, reverse=True)

# Load the ESM tokenizer and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model_names = ['facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t12_35M_UR50D', 'facebook/esm2_t30_150M_UR50D', 'facebook/esm2_t33_650M_UR50D']
model_names = ['facebook/esm2_t12_35M_UR50D', 'facebook/esm2_t30_150M_UR50D', 'facebook/esm2_t33_650M_UR50D']
#model_names = ['facebook/esm2_t6_8M_UR50D']

tokenizer = EsmTokenizer.from_pretrained(model_names[0])
mask_rate = 0.15
batch_size = 4

results = []
for model_name in model_names:
    model = EsmForMaskedLM.from_pretrained(model_name).to(device).eval()
    total_loss = 0.0
    count = 0
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc=f'Evaluating {model_name}'):
            batch = sequences[i:i+batch_size]
            # Tokenize the sequence
            encoding = tokenizer(batch, return_tensors='pt', padding=True, add_special_tokens=True, truncation=True, max_length=1024)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            # Identify which tokens are eligible for masking (non-special, non-pad)
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for st_id in tokenizer.all_special_ids:
                special_token_mask = special_token_mask | (input_ids == st_id)

            # We only mask non-special tokens
            non_special_indices = (~special_token_mask) & (input_ids != tokenizer.pad_token_id)

            candidate_positions = non_special_indices[0].nonzero(as_tuple=True)[0]
            num_to_mask = max(1, int(len(candidate_positions) * mask_rate))

            # Choose random positions to mask, fixed seed ensures reproducibility
            masked_positions = random.sample(candidate_positions.tolist(), num_to_mask)
            masked_positions = torch.tensor(masked_positions)

            # Prepare labels
            labels = input_ids.clone()
            labels[:] = -100
            labels[0, masked_positions] = input_ids[0, masked_positions]

            # Following the standard MLM approach:
            # 80% of the masked tokens -> [MASK]
            # 10% -> random token
            # 10% -> original token (unchanged)
            num_mask = masked_positions.size(0)
            num_mask80 = int(num_mask * 0.8)
            num_mask10_1 = int(num_mask * 0.1)
            num_mask10_2 = num_mask - num_mask80 - num_mask10_1

            shuffled = masked_positions[torch.randperm(num_mask)]
            mask80 = shuffled[:num_mask80]
            rand10 = shuffled[num_mask80:num_mask80+num_mask10_1]
            same10 = shuffled[num_mask80+num_mask10_1:]

            # Store original tokens for masked positions
            true_tokens = input_ids[0, masked_positions].numpy()

            # 80% replaced with [MASK]
            input_ids[0, mask80] = tokenizer.mask_token_id

            # 10% replaced with random tokens
            random_tokens = torch.randint(len(tokenizer), (rand10.size(0),))
            input_ids[0, rand10] = random_tokens

            # 10% remain the same, so we do nothing
            # Compute loss and predictions
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device))
            logits = out.logits
            loss = out.loss.item()
            
            # Get predictions for masked positions
            pred_tokens = torch.argmax(logits[0, masked_positions], dim=-1).cpu().numpy()
            
            all_true.extend(true_tokens)
            all_pred.extend(pred_tokens)
            
            total_loss += loss
            count += 1

    average_loss = total_loss / count
    
    # Calculate metrics
    precision = precision_score(all_true, all_pred, average='weighted')
    recall = recall_score(all_true, all_pred, average='weighted')
    f1 = f1_score(all_true, all_pred, average='weighted')
    accuracy = accuracy_score(all_true, all_pred)
    mcc = matthews_corrcoef(all_true, all_pred)
    
    print(f'Results for {model_name}:')
    print(f'Average loss: {average_loss}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'MCC: {mcc:.4f}')
    
    results.append({
        'model': model_name,
        'average_loss': round(average_loss, 4),
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
results_df.to_csv('esm2_benchmark_results.csv', index=False)
