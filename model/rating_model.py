import torch
import torch.nn as nn
from transformers import AutoModel

bert = AutoModel.from_pretrained("klue/roberta-small")


def init(n, p):
    if "weight" in n:
        torch.nn.init.xavier_normal_(p)
    elif "bias" in n:
        torch.nn.init.zeros_(p)


for p in bert.parameters():
    p.requires_grad_(False)

for n, p in bert.pooler.named_parameters():
    init(n, p)
    p.requires_grad_(True)


class Cls(nn.Module):
    def __init__(self, bert=bert, n_label=2, dim=768):
        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(dim, n_label)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask)["pooler_output"]
        out = self.linear(out)
        return out
