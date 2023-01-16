from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
from model import Cls
import torch.nn as nn
from torch import optim
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-tp", "--train_data_path", default="ratings_data/ratings_train.txt", type=str
    )
    parser.add_argument(
        "-vp",
        "--validation_data_path",
        default="ratings_data/ratings_test.txt",
        type=str,
    )
    parser.add_argument("-tk", "--tokenizer", default="klue/bert-base", type=str)
    parser.add_argument("-ml", "--max_length", default=150, type=int)
    parser.add_argument("-bs", "--batch_size", default=512, type=int)
    parser.add_argument("-ep", "--epochs", default=100, type=int)
    parser.add_argument(
        "-sp", "--save_path", default="state/movie_ratings_state", type=str
    )

    args = parser.parse_args()

train = pd.read_csv(args.train_data_path, sep="\t")
train.dropna(axis=0, how="any", inplace=True)
train.reset_index(drop=True, inplace=True)
test = pd.read_csv(args.validation_data_path, sep="\t")
test.dropna(axis=0, how="any", inplace=True)
test.reset_index(drop=True, inplace=True)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
tokenizer.model_max_length = args.max_length


class dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.document = data.document
        self.label = data.label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids, attention_mask = tokenizer(
            self.document[index],
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
        ).values()
        input_ids.squeeze_()
        attention_mask.squeeze_()
        label = self.label[index]
        return input_ids, attention_mask, label


train_set = dataset(train)
val_set = dataset(test)

train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size)

val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Cls().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters())

epochs = args.epochs
best_accuracy = 0
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    collect = 0

    model.train()
    for data in train_loader:
        input_ids, attention_mask, label = data
        input_ids, attention_mask, label = (
            input_ids.to(device),
            attention_mask.to(device),
            label.to(device),
        )
        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
        loss = loss_fn(output, label)
        train_loss += loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        for data in val_loader:
            input_ids, attention_mask, label = data
            input_ids, attention_mask, label = (
                input_ids.to(device),
                attention_mask.to(device),
                label.to(device),
            )
            output = model(input_ids, attention_mask)
            loss = loss_fn(output, label)
            val_loss += loss
            collect += (torch.argmax(output, dim=-1) == label).sum().item()

    accuracy = collect / len(test)

    print(f"epoch:{epoch}")
    print(f"train_loss:{train_loss}")
    print(f"val_loss:{val_loss}")
    print(f"accutacy:{accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), args.save_path)
