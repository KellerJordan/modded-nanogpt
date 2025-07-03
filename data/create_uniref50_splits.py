import argparse
from datasets import load_dataset, DatasetDict, concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument('--token', type=str, default=None)

args = parser.parse_args()

if args.token:
    import huggingface_hub
    huggingface_hub.login(token=args.token)

data = load_dataset('agemagician/uniref50_09012025').remove_columns('id').remove_columns('name').shuffle(seed=11)
data = data.rename_column('text', 'sequence')
print(data)

data = concatenate_datasets([data['train'], data['validation'], data['test']])

data = data.train_test_split(test_size=20000, seed=22)

train = data['train']
valid = data['test']
valid = valid.train_test_split(test_size=10000, seed=33)
test = valid['test']
valid = valid['train']

data = DatasetDict({
    'train': train,
    'valid': valid,
    'test': test
})

print(data)

data.push_to_hub('Synthyra/uniref50')