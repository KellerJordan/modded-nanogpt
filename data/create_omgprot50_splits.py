"""
Training data is from OMG_prot50
We split 10,000 random examples for validation
We add new examples from UniprotKB between 8/17/2024 and 12/20/2024 to 10,000 more random examples for testing

Uniprot query:
(12/20/2024) for new sequences between 8/17/2024 to 12/20/2024
Filter for proteins that exist at at least the transcript level (no homology or predicted sequences)
(date_created:[2024-08-17 TO 2024-12-20])
"""
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets, Value


data = load_dataset('tattabio/OMG_prot50', split='train').remove_columns('id').shuffle(seed=11)
#data = data.cast_column('sequence', Value(dtype='string'))
print(data)

data = data.train_test_split(test_size=20000, seed=22)

train = data['train']
valid = data['test']
valid = valid.train_test_split(test_size=10000, seed=33)
test = valid['test']
valid = valid['train']

# Load and process UniProt data
df = pd.read_csv('data/uniprotkb_date_created_2024_08_17_TO_20_2024_12_20.tsv', sep='\t')[['Sequence']]
df = df.rename(columns={'Sequence': 'sequence'})
df = df.drop_duplicates(subset=['sequence'], keep='first')

# Convert UniProt data to HuggingFace dataset
uniprot_dataset = Dataset.from_pandas(df).remove_columns('__index_level_0__')
uniprot_dataset = uniprot_dataset.cast_column('sequence', Value(dtype='large_string'))

# Combine UniProt data with test set
test = concatenate_datasets([test, uniprot_dataset])

data = DatasetDict({
    'train': train,
    'valid': valid,
    'test': test
})

print(data)

data.push_to_hub('Synthyra/omg_prot50')