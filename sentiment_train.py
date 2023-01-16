from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
from model.sentiment_model import Cls
import torch.nn as nn
from torch import optim
import torch


train = pd.read_excel(
    "/home/sangyeon/sentiment/감성대화말뭉치(최종데이터)_Training.xlsx", index_col="Unnamed: 0"
)
train.reset_index(drop=True, inplace=True)
validation = pd.read_excel(
    "/home/sangyeon/sentiment/감성대화말뭉치(최종데이터)_Validation.xlsx", index_col="Unnamed: 0"
)
validation.reset_index(drop=True, inplace=True)

sentiment_dict1 = {"분노": 0, "기쁨": 1, "불안": 2, "당황": 3, "슬픔": 4, "상처": 5}
sentiment_dict2 = {
    "노여워하는": 0,
    "느긋": 1,
    "걱정스러운": 2,
    "당혹스러운": 3,
    "당황": 4,
    "마비된": 5,
    "만족스러운": 6,
    "배신당한": 7,
    "버려진": 8,
    "부끄러운": 9,
    "분노": 10,
    "불안": 11,
    "비통한": 12,
    "상처": 13,
    "성가신": 14,
    "스트레스 받는": 15,
    "슬픔": 16,
    "신뢰하는": 17,
    "신이 난": 18,
    "실망한": 19,
    "악의적인": 20,
    "안달하는": 21,
    "안도": 22,
    "억울한": 23,
    "열등감": 24,
    "염세적인": 25,
    "외로운": 26,
    "우울한": 27,
    "고립된": 28,
    "좌절한": 29,
    "후회되는": 30,
    "혐오스러운": 31,
    "한심한": 32,
    "자신하는": 33,
    "기쁨": 34,
    "툴툴대는": 35,
    "남의 시선을 의식하는": 36,
    "회의적인": 37,
    "죄책감의": 38,
    "혼란스러운": 39,
    "초조한": 40,
    "흥분": 41,
    "충격 받은": 42,
    "취약한": 43,
    "편안한": 44,
    "방어적인": 45,
    "질투하는": 46,
    "두려운": 47,
    "눈물이 나는": 48,
    "짜증내는": 49,
    "조심스러운": 50,
    "낙담한": 51,
    "환멸을 느끼는": 52,
    "희생된": 53,
    "감사하는": 54,
    "구역질 나는": 55,
    "괴로워하는": 56,
    "가난한, 불우한": 57,
}


tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")
n_added_token = tokenizer.add_special_tokens(
    {"additional_special_tokens": ["[대분류]", "[소분류]"]}
)
tokenizer.model_max_length = 200

new_vocab_size = tokenizer.vocab_size + n_added_token


class dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.sentiment1 = data.감정_대분류
        self.sentiment2 = data.감정_소분류
        self.utterance1 = data.사람문장1
        self.utterance2 = data.사람문장2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids, attention_mask = tokenizer(
            self.utterance1[index]
            + "[SEP]"
            + self.utterance2[index]
            + " 감정 대분류 : [대분류] - 감정 소분류 : [소분류]",
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
            add_special_tokens=False,
        ).values()
        input_ids.squeeze_()
        attention_mask.squeeze_()
        idx1 = torch.argwhere(input_ids == 32000)
        idx2 = torch.argwhere(input_ids == 32001)
        sentiment1 = sentiment_dict1[self.sentiment1[index]]
        sentiment2 = sentiment_dict2[self.sentiment2[index]]
        return input_ids, attention_mask, idx1, idx2, sentiment1, sentiment2


train_set = dataset(train)
val_set = dataset(validation)

train_loader = DataLoader(dataset=train_set, batch_size=128)

val_loader = DataLoader(dataset=val_set, batch_size=128)

device = "cuda"

model = Cls(vocab_size=new_vocab_size).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters())

epochs = 1000
best_accuracy = 0
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    train_collect1 = 0
    train_collect2 = 0
    train_collect_all = 0
    val_collect1 = 0
    val_collect2 = 0
    val_collect_all = 0

    model.train()
    for data in train_loader:
        input_ids, attention_mask, idx1, idx2, sentiment1, sentiment2 = data
        input_ids, attention_mask, idx1, idx2, sentiment1, sentiment2 = (
            input_ids.to(device),
            attention_mask.to(device),
            idx1.to(device),
            idx2.to(device),
            sentiment1.to(device),
            sentiment2.to(device),
        )
        optimizer.zero_grad()
        out1, out2 = model(input_ids, attention_mask, idx1, idx2)
        loss1 = loss_fn(out1, sentiment1)
        loss2 = loss_fn(out2, sentiment2)
        loss = loss1 + loss2
        train_loss += loss
        loss.backward()
        optimizer.step()
        train_collect1 += (torch.argmax(out1, dim=-1) == sentiment1).sum().item()
        train_collect2 += (torch.argmax(out2, dim=-1) == sentiment2).sum().item()
        train_collect_all += (
            torch.logical_and(
                (torch.argmax(out1, dim=-1) == sentiment1),
                (torch.argmax(out2, dim=-1) == sentiment2),
            )
            .sum()
            .item()
        )

    train_accuracy1 = train_collect1 / len(train)
    train_accuracy2 = train_collect2 / len(train)
    train_accuracy3 = train_collect_all / len(train)

    with torch.no_grad():
        model.eval()
        for data in val_loader:
            input_ids, attention_mask, idx1, idx2, sentiment1, sentiment2 = data
            input_ids, attention_mask, idx1, idx2, sentiment1, sentiment2 = (
                input_ids.to(device),
                attention_mask.to(device),
                idx1.to(device),
                idx2.to(device),
                sentiment1.to(device),
                sentiment2.to(device),
            )
            out1, out2 = model(input_ids, attention_mask, idx1, idx2)
            loss1 = loss_fn(out1, sentiment1)
            loss2 = loss_fn(out2, sentiment2)
            loss = loss1 + loss2
            val_loss += loss
            val_collect1 += (torch.argmax(out1, dim=-1) == sentiment1).sum().item()
            val_collect2 += (torch.argmax(out2, dim=-1) == sentiment2).sum().item()
            val_collect_all += (
                torch.logical_and(
                    (torch.argmax(out1, dim=-1) == sentiment1),
                    (torch.argmax(out2, dim=-1) == sentiment2),
                )
                .sum()
                .item()
            )

    val_accuracy1 = val_collect1 / len(validation)
    val_accuracy2 = val_collect2 / len(validation)
    val_accuracy3 = val_collect_all / len(validation)

    if val_accuracy3 > best_accuracy:
        best_accuracy = val_accuracy3
        torch.save(model.state_dict(), "/home/sangyeon/sentiment/sentiment_state_2.pt")
