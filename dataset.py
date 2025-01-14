from PIL import Image
import torch
from torch.utils.data import Dataset
import os
from utils import remove_colors

class CustomDataset(Dataset):
    def __init__(self, root_dir ,df, vocab, transform=None):
        self.root_dir = root_dir
        self.df = df

        self.transform = transform
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location).convert("RGB")

        if self.transform:
            img = self.transform(img)

        caption = self.captions[idx]
        caption_vec = [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(caption_vec)
