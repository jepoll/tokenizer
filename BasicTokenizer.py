from base64 import decode


class BasicTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def train(self, text, vocab_size, verbose=False):
        tokens = text.encode("utf-8")
        ids = list(tokens)

        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        num_merges = vocab_size - 256 # ensure space for every single byte
        for i in range(num_merges):
            stats = self._get_stats(ids)

            if not stats:
                break

            pair = max(stats, key=lambda p: stats[p])

            idx = 256 + i
            ids = self._merge(ids, pair, idx)
            self.merges[pair] = idx
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx}, occurred {stats[pair]} times")

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        tokens = text.encode("utf-8")
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text


    @staticmethod
    def _merge(ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i +=2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    @staticmethod
    def _get_stats(ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts