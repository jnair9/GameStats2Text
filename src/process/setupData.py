import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer


class GameStatsTextDataset(Dataset):
    """
    PyTorch Dataset for game statistics + Q&A pairs.

    Expects a CSV with columns: date, question, answer, plus numeric stat features.
    Converts 'MP' ("MM:SS") to float minutes and ensures all features are numeric.
    """
    def __init__(self,
                 csv_file: str,
                 tokenizer_name: str = 'gpt2',
                 max_length: int = 256):
        # Load dataframe
        df = pd.read_csv(csv_file)

        # Convert 'MP' from "MM:SS" string to float minutes
        if 'MP' in df.columns:
            df['MP'] = df['MP'].apply(lambda x: float(str(x).split(':')[0]) + float(str(x).split(':')[1]) / 60.0)

        # Identify stat columns (all except date, question, answer)
        self.feature_cols = [c for c in df.columns if c not in ['date', 'question', 'answer']]

        # Ensure numeric columns
        df[self.feature_cols] = df[self.feature_cols].apply(pd.to_numeric, errors='coerce')
        # Optionally: drop rows with NaNs in features
        # df.dropna(subset=self.feature_cols, inplace=True)

        # Convert stats to float tensor
        self.stats = torch.tensor(df[self.feature_cols].values, dtype=torch.float32)

        # Store raw text
        self.questions = df['question'].astype(str).tolist()
        self.answers = df['answer'].astype(str).tolist()

        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        # Ensure we have a PAD token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

        # Tokenize questions
        self.encodings = self.tokenizer(
            self.questions,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Tokenize answers as labels
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                self.answers,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
        self.labels = labels['input_ids']

    def __len__(self):
        return len(self.stats)

    def __getitem__(self, idx):
        return {
            'stats': self.stats[idx],
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }


def collate_fn(batch):
    """
    Custom collate function to batch samples.
    """
    stats = torch.stack([b['stats'] for b in batch])
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    return {
        'stats': stats,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


if __name__ == '__main__':
    # Example usage
    import os
    dataset = GameStatsTextDataset(csv_file=os.path.join('data', 'dataset.csv'))
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    print({k: v.shape for k, v in batch.items()})
