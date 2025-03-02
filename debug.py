import torch
from transformers import PreTrainedTokenizerFast # type: ignore
from model.model import GPT
from utils.config import FullConfig

config = FullConfig()
config.vocab_size = 100  
config.n_layers = 2      
config.n_heads = 2       
config.sequence_length = 32
config.n_embd = 64      
config.batch_size = 1
config.head_mode = 'hyp'
config.attn_mode = 'euc'
config.k_lr = 1.

def encode_text(tokenizer, text):
    return tokenizer.encode(text, return_tensors="pt")

def decode_tokens(tokenizer, tokens):
    if "tinystories_char" in config.data_path:
        return ''.join(tokenizer.convert_ids_to_tokens(tokens.cpu().tolist()))
    return tokenizer.decode(tokens.cpu().tolist(), skip_special_tokens=True)


batch_size = config.batch_size
seq_len = config.sequence_length
n_embd = config.n_embd
device = torch.device('cpu')
input_tensor = torch.randn(batch_size, seq_len, n_embd)

tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tinystories_char/char_tokenizer.json")
config.vocab_size = tokenizer.vocab_size

model = GPT(config)
prompt = "Once upon a time in a"  # Customize as per your dataset
context = encode_text(tokenizer, prompt)
generated_tokens = model.generate_text(context, max_length=50, temperature=1.0, top_k=50)
generated_text = decode_tokens(tokenizer, generated_tokens[0])
print(generated_text)
