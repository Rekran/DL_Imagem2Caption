import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from models import EncoderDecoder
from dataset import CustomDataset
from utils import CapsCollate, save_image_with_caption
import torchvision.transforms as T
import sys
import random
import pandas as pd
import numpy as np
from vocabulary import Vocabulary

# Paths
images_path = '/home/aria/.cache/kagglehub/datasets/adityajn105/flickr8k/versions/1/Images'
caption_path = '/home/aria/.cache/kagglehub/datasets/adityajn105/flickr8k/versions/1/captions.txt'
output_dir = "models/image-cap-model_flip_random_no_color_on_caption"

# Parâmetros
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 30
EMBED_SIZE = 512
HIDDEN_SIZE = 512   
NUM_LAYERS = 2
NUM_WORKER = 1

# Transformação
transforms = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),

])

df = pd.read_csv(caption_path)

n_imagens = df.shape[0]

groups = np.arange(n_imagens) // 5

unique_groups = np.unique(groups)

train_groups, test_groups = train_test_split(unique_groups, test_size=0.2)

df_train = df[np.isin(groups, train_groups)].reset_index(drop=True)
df_test = df[np.isin(groups, test_groups)].reset_index(drop=True)

vocab = Vocabulary(freq_threshold = 1)
vocab.build_vocab(df_train['caption'].tolist())

train_dataset = CustomDataset(images_path, df_train, vocab, transform=transforms)
test_dataset = CustomDataset(images_path, df_test, vocab, transform=transforms)

test_index = random.sample(range(0, len(test_dataset)), 10)


# Obter o índice de padding
pad_idx = vocab.stoi["<PAD>"]

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    shuffle=True,
    collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    shuffle=False,
    collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
)

# Inicializando modelo, otimizador e critério de perda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncoderDecoder(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

# Função para treinar uma época
def train_one_epoch(epoch):
    model.train()
    total_loss = 0

    print(f"\n--- Training Epoch {epoch} ---")

    for batch_idx, (images, captions) in enumerate(train_loader):
        images, captions = images.to(device), captions.to(device)

        # Forward pass
        outputs = model(images, captions)

        # Calcular perda
        loss = criterion(outputs.view(-1, len(vocab)), captions.view(-1))

        # Backward pass e otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f"\rBatch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}", end="")
        sys.stdout.flush()

    print(f"\nEpoch {epoch} - Average Train Loss: {total_loss / len(train_loader):.4f}")



def evaluate_and_save(epoch):
    model.eval()
    total_loss = 0

    # Criar diretório para a época
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    if epoch == 0:
        os.makedirs(output_dir+"/best", exist_ok=True)

    with torch.no_grad():
        for images, captions in test_loader:
            images, captions = images.to(device), captions.to(device)
            outputs = model(images, captions)
            loss = criterion(outputs.view(-1, len(vocab)), captions.view(-1))
            total_loss += loss.item()

        print(f"Epoch {epoch}, Test Loss: {total_loss / len(test_loader):.4f}")

        for i in range(len(test_index)):

            img, _ = test_dataset[test_index[i]]
            img = img.unsqueeze(0).to(device)  

            # Extrair features e gerar legenda
            features = model.encoder(img)
            caption = model.decoder.generate_caption(features.unsqueeze(0), vocab=vocab)
            caption_text = ' '.join(caption)

            # Salvar a imagem com legenda
            save_image_with_caption(img.cpu(), caption_text, epoch_dir, i)
    return total_loss / len(test_loader)


best_loss = 100000000000
# Loop de treinamento e avaliação
for epoch in range(EPOCHS):
    train_one_epoch(epoch)
    loss = evaluate_and_save(epoch)
    if loss < best_loss:
      best_loss = loss
      torch.save(model.state_dict(), os.path.join(f'{output_dir}/best/model.bin'))
