import numpy as np
import torch
from matplotlib import pyplot as plt

from protein_transformer.modules import PositionalEncoding
import seaborn as sns

def random_mask_batch(tokens, tokenizer, **kwargs):
    percent_sequences = kwargs.get("percent_sequences", 0.15)

    # Select 15% of tokens
    #   * 80% <mask>
    #   * 10% <random>
    #   * 10% <original>

    def get_new_token(token):
        prob = torch.FloatTensor(1).uniform_()

         # Mask out with a probability of 80%
        if prob < 0.8:
            return tokenizer.special_indices["mask"]

        # Replace with a random token with a probability of 10%
        elif prob < 0.9:
            return torch.randint(len(tokenizer.standard_tokens), size=(1,))

        # Keep the same with a probability of 10%
        return token

    to_modify = torch.FloatTensor(len(tokens)).uniform_() < percent_sequences

    labels = torch.clone(tokens)
    labels = torch.where(labels == tokenizer.special_indices["pad"], -1 , labels)

    # tokens[to_modify] = tokens[to_modify].apply_(get_new_token)

    masked_tokens = tokens[to_modify].apply_(get_new_token)
    masked_labels = labels[to_modify]

    return masked_tokens, masked_labels


def display_positional_encoding(pe):
    # Visualization snippet from:
    # https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
    pe = pe.T
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
    pos = ax.imshow(pe, cmap="RdGy", extent=(1, pe.shape[1] + 1, pe.shape[0] + 1, 1))
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("Hidden dimension")
    ax.set_title("Positional encoding over hidden dimensions")
    # ax.set_xticks([1] + [i * 10 for i in range(1, 1 + pe.shape[1] // 10)])
    # ax.set_yticks([1] + [i * 10 for i in range(1, 1 + pe.shape[0] // 10)])
    # plt.xticks(rotation=30)
    plt.show()