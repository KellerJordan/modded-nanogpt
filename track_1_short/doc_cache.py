"""The in-document cache as chain links. Per position and context length k (0: the document so far): of the earlier
positions of its document with the same last k tokens, the share the target followed, weighted by bins of their count."""
import torch

ORDERS, EDGES, BOS, P, B = (0, 1, 2, 3, 4, 8, 16), (1, 2, 3, 5, 9, 17), 50256, 134217689, 1103515245
NBINS = len(EDGES) + 1


def _ranks(key):  # per element, how many earlier elements share its key
    ks, order = torch.sort(key, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(key), device=key.device) - torch.searchsorted(ks, ks)
    return rank


@torch.no_grad()
def doc_components(x, y, first, edges):  # per order: (q, the count's bin or -1, the bin count)
    x, y, idx = x.long(), y.long(), torch.arange(len(x), device=x.device)
    seg = torch.cumsum(((x == BOS) | first).long(), 0)
    off, h, out = idx - torch.searchsorted(seg, seg), torch.zeros_like(x), []
    for k in range(max(ORDERS) + 1):  # h: a hash of the last k tokens (mod a 27-bit prime); contexts: (document, h)
        if k in ORDERS:
            ctx = torch.where(off >= k - 1, seg * P + h, -1 - idx)
            cnt, hit = _ranks(ctx), _ranks(ctx * 65536 + y)
            out.append((hit / cnt.clamp_min(1), torch.where(cnt > 0, torch.bucketize(cnt, edges, right=True), -1), NBINS))
        h = (h * B + torch.roll(x, k) + 1) % P
    return out


class DocChain:  # one rank's links over [its validation chunks, its kept fit steps of S tokens], a CUDA graph on a stream
    def __init__(self, n_val, chunk, n_fit, S, device):
        i = torch.arange(n_val + n_fit, device=device)
        self.n_val, self.chunk, self.first = n_val, chunk, torch.where(i < n_val, i % chunk, (i - n_val) % S) == 0
        self.x, self.y = torch.zeros(2, len(i), dtype=torch.int32, device=device)
        self.edges, self.stream, self.graph = torch.tensor(EDGES, device=device), torch.cuda.Stream(device), torch.cuda.CUDAGraph()
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.stream):
            doc_components(self.x, self.y, self.first, self.edges)  # warm, before the capture
        with torch.cuda.graph(self.graph, stream=self.stream, capture_error_mode="thread_local"):
            self.out = doc_components(self.x, self.y, self.first, self.edges)

    def launch(self, val_tokens, offsets, f_tok, f_tgt):
        self.stream.wait_stream(torch.cuda.current_stream(self.x.device))
        with torch.cuda.stream(self.stream):
            torch.cat([val_tokens[o:o + self.chunk] for o in offsets] + [f_tok], out=self.x)
            torch.cat([val_tokens[o + 1:o + self.chunk + 1] for o in offsets] + [f_tgt], out=self.y)
            self.graph.replay()
        for t in (val_tokens, f_tok, f_tgt):
            t.record_stream(self.stream)

    def part(self, c=None):  # the links of validation chunk c, or (None) of the fit positions
        torch.cuda.current_stream(self.x.device).wait_stream(self.stream)
        s = slice(self.n_val, None) if c is None else slice(c * self.chunk, (c + 1) * self.chunk)
        return [(q[s], g[s], n) for q, g, n in self.out]
