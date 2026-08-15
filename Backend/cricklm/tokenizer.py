"""
tokenizer.py — CrickLM Tokenizer
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional


# Special tokens 
PAD_TOKEN = "<PAD>"   
UNK_TOKEN = "<UNK>"   
BOS_TOKEN = "<BOS>"   
EOS_TOKEN = "<EOS>"   

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class CrickTokenizer:

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size

        
        self.stoi: dict = {}   
        self.itos: dict = {}   

        
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.stoi[tok] = i
            self.itos[i] = tok

        self.is_built = False

    # Text cleaning 
    def _tokenize_text(self, text: str) -> List[str]:
        text = text.lower().strip()
        
        text = re.sub(r"([!?.,;:()\[\]\"'-])", r" \1 ", text)
        
        text = re.sub(r"\s+", " ", text)
        return [t for t in text.split(" ") if t]

    # Vocabulary construction 
    def build_vocab(self, corpus: str) -> None:
        print(f"Building vocabulary from {len(corpus):,} characters...")

        tokens = self._tokenize_text(corpus)
        print(f"  Total tokens in corpus: {len(tokens):,}")

        freq = Counter(tokens)
        print(f"  Unique words found: {len(freq):,}")

        max_words = self.vocab_size - len(SPECIAL_TOKENS)
        most_common = freq.most_common(max_words)

        for idx, (word, count) in enumerate(most_common):
            token_id = idx + len(SPECIAL_TOKENS)   
            self.stoi[word] = token_id
            self.itos[token_id] = word

        self.vocab_size = len(self.stoi)
        self.is_built = True

        coverage = sum(c for _, c in most_common) / max(len(tokens), 1)
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Corpus coverage: {coverage:.1%}  (fraction of tokens in vocab)")

    # Encode / Decode 
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        
        if not self.is_built:
            raise RuntimeError("Vocabulary not built yet. Call build_vocab() first.")

        words = self._tokenize_text(text)
        ids = [self.stoi.get(w, UNK_ID) for w in words]

        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]

        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        
        special_ids = {PAD_ID, BOS_ID, EOS_ID} if skip_special else set()
        words = []
        for i in ids:
            if i in special_ids:
                continue
            if i == UNK_ID and skip_special:
                continue
            words.append(self.itos.get(i, UNK_TOKEN))

        text = " ".join(words)
        text = re.sub(r" ([!?.,;:()\[\]\"'])", r"\1", text)
        return text

    def pad_sequence(self, ids: List[int], length: int) -> List[int]:
        """Pad or truncate a sequence to exactly `length` tokens."""
        if len(ids) >= length:
            return ids[:length]
        return ids + [PAD_ID] * (length - len(ids))

    # Persistence 
    def save(self, path: str) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            
            "itos": {str(k): v for k, v in self.itos.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"Tokenizer saved → {path}")

    @classmethod
    def load(cls, path: str) -> "CrickTokenizer":
        """Load a previously saved vocabulary."""
        data = json.loads(Path(path).read_text())
        tok = cls(vocab_size=data["vocab_size"])
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        tok.is_built = True
        print(f"Tokenizer loaded ← {path}  (vocab_size={tok.vocab_size:,})")
        return tok

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"CrickTokenizer(vocab_size={self.vocab_size}, built={self.is_built})"