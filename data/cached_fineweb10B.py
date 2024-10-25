import os
import sys
from huggingface_hub import hf_hub_download
# Download the GPT-2 tokens of Fineweb10B from huggingface. This
# saves about an hour of startup time compared to regenerating them.
def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), 'fineweb10B')
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                        repo_type="dataset", local_dir=local_dir)
get("fineweb_val_%06d.bin" % 0)
num_chunks = 103 # full fineweb10B. Each chunk is ~98.5M tokens
if len(sys.argv) >= 2: # we can pass an argument to download less
    num_chunks = int(sys.argv[1])
for i in range(1, num_chunks+1):
    get("fineweb_train_%06d.bin" % i)
