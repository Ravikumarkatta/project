import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class BibleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(train_texts, train_labels, val_texts, val_labels, 
                       tokenizer_name, batch_size=32, max_length=512):
    train_dataset = BibleDataset(
        train_texts, train_labels, tokenizer_name, max_length
    )
    val_dataset = BibleDataset(
        val_texts, val_labels, tokenizer_name, max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )

    return train_loader, val_loader
