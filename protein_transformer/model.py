import math

import torch
from torch import optim, nn
import torch.nn.functional as F
import pytorch_lightning as pl

from protein_transformer.modules import PositionalEncoding, Encoder, CosineWarmupScheduler, FeedForwardNet
from protein_transformer.utils import random_mask_batch


class Transformer(pl.LightningModule):
    def __init__(self,
                 tokenizer,
                 embed_dim=100,
                 num_layers=2,
                 dropout=0.1,
                 num_heads=2,
                 lr=0.001,
                 num_warmup_steps=50,
                 output_fnn_hidden_dim=100,
                 **kwargs
                 ):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=embed_dim)
        self.pe = PositionalEncoding(embed_dim=embed_dim, max_len=tokenizer.max_seq_length)
        self.encoder = Encoder(embed_dim=embed_dim, num_layers=num_layers, dropout=dropout, num_heads=num_heads)
        self.output_net = FeedForwardNet(in_dim=embed_dim, out_dim=tokenizer.vocab_size,
                                         hidden_dim=output_fnn_hidden_dim).to(torch.double)

        self.tokenizer = tokenizer
        self.embed_dim = embed_dim
        self.num_warmup_steps = num_warmup_steps
        self.lr = lr
        self.percent_sequences = kwargs.get("percent_sequences", 0.15)
        self.warmup_max_iters = kwargs.get("warmup_max_iters", 2000)

    def forward(self, x, logits=True):
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.pe(x)
        x = self.encoder(x)
        x = self.output_net(x)

        if logits:
            return x

        return F.softmax(x, dim=1)

    def training_step(self, batch):
        masked_x, labels = random_mask_batch(batch, self.tokenizer, percent_sequences=self.percent_sequences)

        logits = self.forward(masked_x)

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index=-1
        )

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.num_warmup_steps, max_iters=self.warmup_max_iters
        )

        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration
