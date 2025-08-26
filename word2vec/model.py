import torch.nn as nn
import torch.nn.functional as F

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_size=256):
        super().__init__()
        self.embeddings = nn.Linear(vocab_size, embedding_size)
        self.output = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        return F.sigmoid(self.output(x))