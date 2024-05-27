import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import tqdm
import random
import math
import os
import csv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
class GPT2Dataset(IterableDataset):
    def __init__(self, data, tokenizer, split="train", n_tokens=1024, seed=0, data_size=1.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_tokens = n_tokens
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.tokenized = []
        ind = list(range(len(data)))
        if split == "train":
            random.seed(seed)
            random.shuffle(ind)
            new_len = int(len(ind) * data_size)
            ind = ind[:new_len]
        print("read %s entries from the raw dataset" % str(len(ind)))
        print("tokenizing...")
        for i in tqdm.tqdm(ind):
            line = data[i]
            if line["text"] == "":
                continue
            temp = self.tokenizer(line["text"], padding=False, truncation=False)
            input_ids =  temp["input_ids"] + [self.eos_token_id]
            attention_mask = temp["attention_mask"] + [1]
            self.tokenized.append({"input_ids": input_ids, "attention_mask": attention_mask})
        print("processed %s entries in the tokenized dataset" % str(len(self.tokenized)))
        self.start = 0
        self.end = len(self.tokenized)
        
        
    def __iter__(self):
        input_ids_list = []
        attention_mask_list = []
        for i, sent in enumerate(self.tokenized[self.start:self.end]):
            input_ids = sent["input_ids"]
            attention_mask = sent["attention_mask"]
            while len(input_ids) > 0:
                init_input_ids_list_len = len(input_ids_list)
                init_attention_mask_list_len = len(attention_mask_list)
                input_ids_list += input_ids[:self.n_tokens - len(input_ids_list)]
                attention_mask_list += attention_mask[:self.n_tokens - len(attention_mask_list)]
                input_ids = input_ids[self.n_tokens - init_input_ids_list_len:]
                attention_mask = attention_mask[self.n_tokens - init_attention_mask_list_len:]
                if len(input_ids_list) >= self.n_tokens:
                    yield {"input_ids": input_ids_list, "attention_mask": attention_mask_list}
                    input_ids_list = []
                    attention_mask_list = []

class GPT2Collator:
    def __call__(self, batch):
        input_ids_list = []
        attention_mask_list = []
        for item in batch:
            input_ids = item["input_ids"] 
            attention_mask =item["attention_mask"]
            input_ids_list.append(torch.tensor(input_ids, dtype=torch.int))
            attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.int))
        return {"input_ids": torch.stack(input_ids_list), "attention_mask": torch.stack(attention_mask_list)}


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    overall_start = dataset.start
    overall_end = dataset.end
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


def get_dataloader(dataset, collator, batch_size, worker_init_fn=None, num_workers=0, drop_last=True):
    return DataLoader(dataset, collate_fn=collator, batch_size=batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=drop_last)


def load_synthetic_dataset(data, valid=4000, test=4000):
    with open(data, "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = [list(map(int, rec)) for rec in reader]
    train = data[:-(valid + test)]
    valid = data[len(train):-test]
    test = data[len(train) + len(valid):]
    return train, valid, test


class SyntheticTokenizer:
    def __init__(self, nvocab):
        self.nvocab = nvocab
        self.eos_token_id = self.nvocab


class SyntheticDataset(IterableDataset):
    def __init__(self, data, split="train", n_tokens=1024, seed=0, data_size=1.0):
        super().__init__()
        self.data = data
        self.tokenized = []
        ind = list(range(len(data)))
        if split == "train":
            random.seed(seed)
            random.shuffle(ind)
            new_len = int(len(ind) * data_size)
            ind = ind[:new_len]
        print("read %s entries from the raw dataset" % str(len(ind)))
        for i in tqdm.tqdm(ind):
            line = data[i]
            input_ids =  line 
            attention_mask = [1 for _ in range(len(line))]
            self.tokenized.append({"input_ids": input_ids, "attention_mask": attention_mask})
        print("processed %s entries in the tokenized dataset" % str(len(self.tokenized)))
        self.start = 0
        self.end = len(self.tokenized)
        
        
    def __iter__(self):
        input_ids_list = []
        attention_mask_list = []
        for i, sent in enumerate(self.tokenized[self.start:self.end]):
            input_ids = sent["input_ids"]
            attention_mask = sent["attention_mask"]
            yield {"input_ids": input_ids, "attention_mask": attention_mask}



