
def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def print_tokens(tokenizer, text, name):
    ids = tokenizer.encode(text)
    print(f"\n{name}")
    print("ids:", ids)
    print("count:", len(ids))

    for idx in ids:
        raw = tokenizer.vocab[idx]
        piece = raw.decode("utf-8", errors="replace")
        print(f"{idx:>4} | {raw!r:<15} | {piece!r}")

class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self.pattern = {}

    def train(self, txt, vocab_size, verbose=False):
        pass

    def decode(self, ids):
        pass

    def encode(self, text):
        pass

    def _create_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def _encode_chunk(self, ids):
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

