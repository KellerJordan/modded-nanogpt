"""
BPE tokenizer with optional forced morpheme merges.

Before statistical BPE runs, a curated list of morphemes (prefixes/suffixes)
are force-merged into dedicated tokens. The remaining vocabulary trains
statistically as standard BPE. This gives the model explicit compositional
building blocks (negation, nominalization, etc.) for reasoning.

Usage:
    trainer = MorphologicalBPETrainer(pattern=GPT4_SPLIT_PATTERN)
    trainer.train(text, vocab_size=4096, forced_morphemes=["un", "re", "ing", "tion"])
    enc = trainer.to_tiktoken(name="morph_bpe")
    ids = enc.encode("unhappiness")
"""

import regex as re
from collections import Counter, defaultdict
import tiktoken


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def _merge_inplace(ids, pair, new_id):
    """Replace all non-overlapping occurrences of pair with new_id, in place."""
    i = 0
    while i < len(ids) - 1:
        if ids[i] == pair[0] and ids[i + 1] == pair[1]:
            ids[i] = new_id
            ids.pop(i + 1)
        else:
            i += 1


class MorphologicalBPETrainer:
    """
    A BPE tokenizer trainer that supports forced morpheme merges.

    The merge dictionary and vocab are built incrementally:
      - Bytes 0-255: single-byte tokens
      - 256..256+F-1: forced morpheme merges (if any)
      - 256+F..: statistical BPE merges
    """

    def __init__(self, pattern=None):
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.merges = {}       # (int, int) -> int
        self.vocab = {}        # int -> bytes
        self.forced_merge_count = 0

    def _apply_forced_merges(self, ids_list, morphemes):
        """
        Apply forced morpheme merges before statistical BPE.

        Morphemes are sorted by byte length (shortest first) so that
        shorter morphemes (e.g. "in") are merged before longer ones
        (e.g. "inter"), enabling prefix sharing of merge IDs.

        Returns the number of new merges created.
        """
        if not morphemes:
            return 0

        # Sort by byte length, stable order
        sorted_morphemes = sorted(morphemes, key=lambda m: len(m.encode("utf-8")))
        merges_created = 0

        for morpheme in sorted_morphemes:
            morpheme_bytes = morpheme.encode("utf-8")
            # Skip single-byte morphemes (already tokens 0-255)
            if len(morpheme_bytes) <= 1:
                continue

            # Build token sequence for this morpheme's bytes
            token_seq = list(morpheme_bytes)

            # Walk left-to-right, merging pairs until a single token
            while len(token_seq) >= 2:
                pair = (token_seq[0], token_seq[1])

                if pair in self.merges:
                    # Reuse existing merge (from a shorter morpheme)
                    merged_id = self.merges[pair]
                else:
                    # Assign next available ID
                    merged_id = 256 + len(self.merges)
                    self.merges[pair] = merged_id
                    self.vocab[merged_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
                    merges_created += 1

                    # Apply this merge to all words
                    for ids in ids_list:
                        _merge_inplace(ids, pair, merged_id)

                # Collapse first two tokens into the merged token
                token_seq = [merged_id] + token_seq[2:]

        return merges_created

    def train(self, text, vocab_size, forced_morphemes=None, verbose=False):
        """
        Train BPE on text with optional forced morpheme merges.

        Args:
            text: training text (string)
            vocab_size: target vocabulary size (>= 256)
            forced_morphemes: optional list of morpheme strings to force-merge first
            verbose: print merge progress
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # Reset state
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.forced_merge_count = 0

        # Split text into chunks and collapse identical chunks
        text_chunks = re.findall(self.compiled_pattern, text)
        counts = Counter(text_chunks)
        unique_chunks = list(counts.keys())
        chunk_counts = [counts[ch] for ch in unique_chunks]

        # Convert to byte-level token sequences
        ids_list = [list(ch.encode("utf-8")) for ch in unique_chunks]

        # Phase 1: Apply forced morpheme merges
        if forced_morphemes:
            self.forced_merge_count = self._apply_forced_merges(ids_list, forced_morphemes)
            if verbose:
                print(f"Forced morpheme merges: {self.forced_merge_count} merges "
                      f"from {len(forced_morphemes)} morphemes")

        merges_done = len(self.merges)

        # Phase 2: Statistical BPE for remaining merge budget
        # Build initial pair counts
        stats = defaultdict(int)
        positions = defaultdict(set)  # pair -> set of chunk indices

        for chunk_idx, (chunk_ids, count) in enumerate(zip(ids_list, chunk_counts)):
            for pair in zip(chunk_ids, chunk_ids[1:]):
                stats[pair] += count
                positions[pair].add(chunk_idx)

        while merges_done < num_merges:
            if not stats:
                break

            # Find the pair with the highest count
            pair = max(stats, key=stats.get)
            if stats[pair] <= 0:
                break

            # Assign next available ID
            new_id = 256 + merges_done
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose and (merges_done % 100 == 0 or merges_done < 10):
                print(f"merge {merges_done}/{num_merges}: {pair} -> {new_id} "
                      f"({self.vocab[new_id]}) freq={stats[pair]}")

            # Apply merge to affected chunks and track count changes
            affected_chunks = positions[pair]
            count_changes = defaultdict(int)

            for chunk_idx in affected_chunks:
                chunk_ids = ids_list[chunk_idx]
                chunk_count = chunk_counts[chunk_idx]
                ix = 0
                while ix < len(chunk_ids) - 1:
                    if chunk_ids[ix] == pair[0] and chunk_ids[ix + 1] == pair[1]:
                        # Track pair count changes around the merge site
                        if ix > 0:
                            count_changes[(chunk_ids[ix - 1], chunk_ids[ix])] -= chunk_count
                        count_changes[pair] -= chunk_count
                        if ix + 2 < len(chunk_ids):
                            count_changes[(chunk_ids[ix + 1], chunk_ids[ix + 2])] -= chunk_count

                        # Apply merge
                        chunk_ids[ix] = new_id
                        chunk_ids.pop(ix + 1)

                        # Track new pairs
                        if ix > 0:
                            count_changes[(chunk_ids[ix - 1], chunk_ids[ix])] += chunk_count
                        if ix + 1 < len(chunk_ids):
                            count_changes[(chunk_ids[ix], chunk_ids[ix + 1])] += chunk_count
                    else:
                        ix += 1

            # Apply incremental count changes
            for changed_pair, delta in count_changes.items():
                if changed_pair == pair:
                    continue
                stats[changed_pair] += delta
                # Update positions
                for chunk_idx in affected_chunks:
                    chunk_ids = ids_list[chunk_idx]
                    has_pair = any(
                        (chunk_ids[j], chunk_ids[j + 1]) == changed_pair
                        for j in range(len(chunk_ids) - 1)
                    )
                    if has_pair:
                        positions[changed_pair].add(chunk_idx)
                    else:
                        positions[changed_pair].discard(chunk_idx)

            # Remove merged pair
            del stats[pair]
            del positions[pair]

            merges_done += 1

        if verbose:
            print(f"Training complete: {merges_done} total merges "
                  f"({self.forced_merge_count} forced + "
                  f"{merges_done - self.forced_merge_count} statistical)")

    def train_from_iterator(self, text_iterator, vocab_size, forced_morphemes=None, verbose=False):
        """
        Train BPE from a streaming text iterator.

        Collects unique chunks and their counts from the iterator,
        then runs the same merge algorithm.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # Reset state
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.forced_merge_count = 0

        # Stream text and accumulate chunk counts
        chunk_counter = Counter()
        total_chars = 0
        for text in text_iterator:
            chunks = re.findall(self.compiled_pattern, text)
            chunk_counter.update(chunks)
            total_chars += len(text)

        if verbose:
            print(f"Processed {total_chars:,} chars, {len(chunk_counter):,} unique chunks")

        unique_chunks = list(chunk_counter.keys())
        chunk_counts = [chunk_counter[ch] for ch in unique_chunks]
        ids_list = [list(ch.encode("utf-8")) for ch in unique_chunks]

        # Phase 1: Forced morpheme merges
        if forced_morphemes:
            self.forced_merge_count = self._apply_forced_merges(ids_list, forced_morphemes)
            if verbose:
                print(f"Forced morpheme merges: {self.forced_merge_count} merges "
                      f"from {len(forced_morphemes)} morphemes")

        merges_done = len(self.merges)

        # Phase 2: Statistical BPE
        stats = defaultdict(int)
        positions = defaultdict(set)

        for chunk_idx, (chunk_ids, count) in enumerate(zip(ids_list, chunk_counts)):
            for pair in zip(chunk_ids, chunk_ids[1:]):
                stats[pair] += count
                positions[pair].add(chunk_idx)

        while merges_done < num_merges:
            if not stats:
                break
            pair = max(stats, key=stats.get)
            if stats[pair] <= 0:
                break

            new_id = 256 + merges_done
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose and (merges_done % 500 == 0):
                print(f"merge {merges_done}/{num_merges}: {pair} -> {new_id} freq={stats[pair]}")

            affected_chunks = positions[pair]
            count_changes = defaultdict(int)

            for chunk_idx in affected_chunks:
                chunk_ids = ids_list[chunk_idx]
                chunk_count = chunk_counts[chunk_idx]
                ix = 0
                while ix < len(chunk_ids) - 1:
                    if chunk_ids[ix] == pair[0] and chunk_ids[ix + 1] == pair[1]:
                        if ix > 0:
                            count_changes[(chunk_ids[ix - 1], chunk_ids[ix])] -= chunk_count
                        count_changes[pair] -= chunk_count
                        if ix + 2 < len(chunk_ids):
                            count_changes[(chunk_ids[ix + 1], chunk_ids[ix + 2])] -= chunk_count
                        chunk_ids[ix] = new_id
                        chunk_ids.pop(ix + 1)
                        if ix > 0:
                            count_changes[(chunk_ids[ix - 1], chunk_ids[ix])] += chunk_count
                        if ix + 1 < len(chunk_ids):
                            count_changes[(chunk_ids[ix], chunk_ids[ix + 1])] += chunk_count
                    else:
                        ix += 1

            for changed_pair, delta in count_changes.items():
                if changed_pair == pair:
                    continue
                stats[changed_pair] += delta
                for chunk_idx in affected_chunks:
                    chunk_ids = ids_list[chunk_idx]
                    has_pair = any(
                        (chunk_ids[j], chunk_ids[j + 1]) == changed_pair
                        for j in range(len(chunk_ids) - 1)
                    )
                    if has_pair:
                        positions[changed_pair].add(chunk_idx)
                    else:
                        positions[changed_pair].discard(chunk_idx)

            del stats[pair]
            del positions[pair]
            merges_done += 1

        if verbose:
            print(f"Training complete: {merges_done} total merges "
                  f"({self.forced_merge_count} forced + "
                  f"{merges_done - self.forced_merge_count} statistical)")

    def encode(self, text):
        """Encode text into token IDs."""
        text_chunks = re.findall(self.compiled_pattern, text)
        all_ids = []
        for chunk in text_chunks:
            ids = list(chunk.encode("utf-8"))
            while len(ids) >= 2:
                # Find the pair with the lowest merge index (highest priority)
                pairs = {(ids[i], ids[i + 1]): i for i in range(len(ids) - 1)}
                best_pair = min(
                    pairs.keys(),
                    key=lambda p: self.merges.get(p, float("inf"))
                )
                if best_pair not in self.merges:
                    break
                new_id = self.merges[best_pair]
                ids = self._apply_merge(ids, best_pair, new_id)
            all_ids.extend(ids)
        return all_ids

    def decode(self, ids):
        """Decode token IDs back to text."""
        parts = []
        for idx in ids:
            if idx in self.vocab:
                parts.append(self.vocab[idx])
            else:
                parts.append(bytes([idx]))
        return b"".join(parts).decode("utf-8", errors="replace")

    @staticmethod
    def _apply_merge(ids, pair, new_id):
        """Apply a single merge to a token list (returns new list)."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def get_mergeable_ranks(self):
        """Build the mergeable_ranks dict for tiktoken: bytes -> rank."""
        mergeable_ranks = {}

        # Single bytes: rank 0..255
        for i in range(256):
            mergeable_ranks[bytes([i])] = i

        # Merged tokens: sorted by merge order (token ID)
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        for (left, right), merged_id in sorted_merges:
            left_bytes = self.vocab[left]
            right_bytes = self.vocab[right]
            merged_bytes = left_bytes + right_bytes
            mergeable_ranks[merged_bytes] = merged_id

        return mergeable_ranks

    def to_tiktoken(self, name="morphological_bpe", special_tokens=None):
        """
        Export to a tiktoken Encoding for efficient inference.

        Args:
            name: encoding name
            special_tokens: dict of special token strings to IDs.
                           If None, adds <|endoftext|> at next available ID.
        """
        mergeable_ranks = self.get_mergeable_ranks()

        if special_tokens is None:
            next_id = max(mergeable_ranks.values()) + 1
            special_tokens = {"<|endoftext|>": next_id}

        enc = tiktoken.Encoding(
            name=name,
            pat_str=self.pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return enc
