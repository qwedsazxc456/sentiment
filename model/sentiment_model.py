import torch
import torch.nn as nn
from transformers import AutoModel

roberta_pretrain = AutoModel.from_pretrained("klue/roberta-small")

for p in roberta_pretrain.parameters():
    p.requires_grad_(False)


class Cls(nn.Module):
    def __init__(
        self,
        roberta_pretrain=roberta_pretrain,
        n_sentiment1=6,
        n_sentiment2=58,
        dim=768,
        vocab_size=32002,
    ):
        super().__init__()
        self.roberta_pretrain = roberta_pretrain
        roberta_pretrain.resize_token_embeddings(vocab_size)
        self.linear1 = nn.Linear(dim, n_sentiment1)
        self.linear2 = nn.Linear(dim, n_sentiment2)
        self.linear3 = nn.Linear(dim, dim)
        self.linear4 = nn.Linear(dim, dim)
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, idx1, idx2):
        out = self.roberta_pretrain(input_ids, attention_mask)["last_hidden_state"]
        sen1, sen2 = torch.gather(
            out, 1, idx1.reshape(-1, 1, 1).repeat(1, 1, out.size(-1))
        ), torch.gather(out, 1, idx2.reshape(-1, 1, 1).repeat(1, 1, out.size(-1)))
        sen1 = self.tanh(self.linear3(sen1))
        sen2 = self.tanh(self.linear4(sen2))
        out1 = self.linear1(sen1)
        out2 = self.linear2(sen2)
        return out1.squeeze(1), out2.squeeze(1)
