from base64 import decode

from base_tokenizer import Tokenizer, get_stats, merge



class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        tokens = text.encode("utf-8")
        ids = list(tokens)

        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        num_merges = vocab_size - 256 # ensure space for every single byte
        for i in range(num_merges):
            stats = get_stats(ids)

            if not stats:
                break

            pair = max(stats, key=lambda p: stats[p])

            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx}, occurred {stats[pair]} times")

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        return self._encode_chunk(ids)


    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text