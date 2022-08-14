from typing import Optional, Union, List
from collections import defaultdict
import numpy as np


class Tokenizer:
    def __init__(self,
                 max_seq_length,
                 **kwargs
                 ):
        self.standard_tokens = [
            'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H',
            'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
        ]
        self.special_tokens = {
            "start": '<cls>',
            # "end": '<eos>',
            "unknown": '<unk>',
            "pad": '<pad>',
            "mask": '<mask>'
        }

        self.all_tokens = self.standard_tokens + list(self.special_tokens.values())
        self.vocab_size = len(self.all_tokens)

        token_to_index_map_ = {
            token: i
            for i, token in enumerate(self.all_tokens)
        }
        self.special_indices = {
            token_name: token_to_index_map_[token]
            for token_name, token in self.special_tokens.items()
        }

        # Make unknown special token default
        token_to_index_map = defaultdict(
            lambda: self.special_indices["unknown"],
            **token_to_index_map_
        )

        self.token_to_index_map = token_to_index_map

        # Add 1 to account for the start token
        self.max_seq_length = max_seq_length + 1

    def tokenize(self, text: str):
        return text

    def encode(self, text: str):
        # TODO: Currently, does not crop to desired length
        encoded = [self.token_to_index_map[token] for token in self.tokenize(text)]
        encoded = [self.special_indices["start"]] + encoded
        return encoded

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def batch_encode(self, batch_text: List[str]):

        def pad_to_dense(M, to_size, value):
            Z = np.empty((len(M), to_size))
            Z.fill(value)

            for i, row in enumerate(M):
                # Crop if the sequence is longer than maximum allowed size
                # TODO: replace with random cropping
                if len(row) > to_size:
                    row = row[:to_size]
                Z[i, :len(row)] = row

            return Z

        batch_text_encoded = [self.encode(seq) for seq in batch_text]
        batch_text_padded = pad_to_dense(
            M=batch_text_encoded,
            to_size=self.max_seq_length,
            value=self.special_indices["pad"]
        )
        batch_text_padded = batch_text_padded.astype(np.int64)

        return batch_text_padded
