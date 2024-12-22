from datasets import load_dataset
from huggingface_hub import hf_hub_download


# Adjust repo_id and filename to your dataset
local_file = hf_hub_download(
    repo_id="Synthyra/omg_prot50",
    filename="data/test-00000-of-00001.parquet",
    repo_type="dataset"  # very important, since it's a dataset repo not a model repo
)
local_file = local_file.replace('\\', '/').split('/data')[0]
print(local_file)
data = load_dataset(local_file, split='test').remove_columns('__index_level_0__')
print(data)
print(data['id'][0])
print(data['sequence'][0])
