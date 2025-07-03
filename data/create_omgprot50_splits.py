import argparse
from datasets import load_dataset, DatasetDict

parser = argparse.ArgumentParser()
parser.add_argument('--token', type=str, defualt=None)

args = parser.parse_args()

if args.token:
    import huggingface_hub
    huggingface_hub.login(token=args.token)

data = load_dataset('tattabio/OMG_prot50', split='train').remove_columns('id').shuffle(seed=11)
#data = data.cast_column('sequence', Value(dtype='string'))
print(data)

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

data.push_to_hub('Synthyra/omg_prot50')