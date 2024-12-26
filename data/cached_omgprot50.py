import os
import sys
from huggingface_hub import hf_hub_download
# Download the omgprot50 tokens from huggingface. This
# saves about an hour of startup time compared to regenerating them.
def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), 'omgprot50')
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id="lapp0/omg_prot50_packed", filename=fname,
                        repo_type="dataset", local_dir=local_dir)
get("omgprot50_val_%06d.bin" % 0)
get("omgprot50_test_%06d.bin" % 0)
num_chunks = 442  # Each chunk is 100M tokens
if len(sys.argv) >= 2: # we can pass an argument to download less
    num_chunks = int(sys.argv[1])
for i in range(1, num_chunks+1):
    get("omgprot50_train_%06d.bin" % i)
