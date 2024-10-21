import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models import EncoderDecoder
from dataset import CustomDataset
from utils import CapsCollate, save_image_with_caption, save_metrics
import torchvision.transforms as T
import sys
import pandas as pd
import numpy as np
from vocabulary import Vocabulary
from rouge_score import rouge_scorer



# Paths
# images_path = '/home/aria/.cache/kagglehub/datasets/adityajn105/flickr8k/versions/1/Images'
# caption_path = '/home/aria/.cache/kagglehub/datasets/adityajn105/flickr8k/versions/1/captions.txt'
# output_dir = "models/image-cap-model_flip_random_rouge_8k_attetion_512x512"

images_path = '/home/aria/.cache/kagglehub/datasets/adityajn105/flickr30k/versions/1/Images'
caption_path = '/home/aria/.cache/kagglehub/datasets/adityajn105/flickr30k/versions/1/captions.txt'
output_dir = "models/image-cap-model_flip_random_rouge_30k_attetion_512x512"

# Parameters
BATCH_SIZE = 15
LEARNING_RATE = 0.0001
EPOCHS = 36
EMBED_SIZE = 512
HIDDEN_SIZE = 512   
NUM_LAYERS = 5
NUM_WORKER = 1

# Transformations
transforms = T.Compose([
    T.Resize((512, 512)),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])
# random.seed(42)

df = pd.read_csv(caption_path)

df.at[19999,'caption'] = "A dog runs across the grassy field ."

n_images = df.shape[0]
groups = np.arange(n_images) // 5
train_groups, test_groups = train_test_split(np.unique(groups), test_size=0.2)

df_train = df[np.isin(groups, train_groups)].reset_index(drop=True)
df_test = df[np.isin(groups, test_groups)].reset_index(drop=True)


vocab = Vocabulary(freq_threshold=1)
vocab.build_vocab(df_train['caption'].tolist())
os.makedirs(os.path.join(output_dir, 'vocab'), exist_ok=True)
vocab.save_vocab(os.path.join(output_dir,'vocab','vocab.json'))

train_dataset = CustomDataset(images_path, df_train, vocab, transform=transforms)
test_dataset = CustomDataset(images_path, df_test, vocab, transform=transforms)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EncoderDecoder(
    embed_size=300,
    vocab_size = len(vocab),
    attention_dim=512,
    encoder_dim=2048,
    decoder_dim=512
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

def train_one_epoch(epoch):
    model.train()
    total_loss = 0

    print(f"\n--- Training Epoch {epoch} ---")

    for batch_idx, (images, captions) in enumerate(train_loader):
        images, captions = images.to(device), captions.to(device)

        # Forward pass
        outputs,alph = model(images, captions)

        # Compute loss
        targets = captions[:,1:]
        loss = criterion(outputs.view(-1, len(vocab)), targets.reshape(-1))

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 5 == 0:
            print(f"\rBatch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}", end="")
            sys.stdout.flush()

    print(f"\nEpoch {epoch} - Average Train Loss: {total_loss / len(train_loader):.4f}")

def evaluate_and_save(epoch):
    model.eval()
    total_loss = 0
    all_rouge_scores = [] 

    epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    if epoch == 0:
        os.makedirs(output_dir + "/best", exist_ok=True)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    with torch.no_grad():
        for images, captions in test_loader:
            images, captions = images.to(device), captions.to(device)

            outputs, alph = model(images, captions)
            targets = captions[:, 1:]
            loss = criterion(outputs.view(-1, len(vocab)), targets.reshape(-1))
            total_loss += loss.item()
            pred_vec, pred_vec_text, alp = [], [], []

            for i in range(len(images)):
                img, gt_caption = images[i].unsqueeze(0), captions[i]

                features = model.encoder(img)  # Extract features
                predicted_caption, _ = model.decoder.generate_caption(features, vocab=vocab, max_len=20)
                predicted_caption_text = ' '.join(predicted_caption)
                alp.append(_)

                gt_caption_text = ' '.join([
                    vocab.itos[token.item()] 
                    for token in gt_caption if token.item() in vocab.itos
                ])

                rouge_score = scorer.score(gt_caption_text, predicted_caption_text)
                all_rouge_scores.append(rouge_score)
                pred_vec.append(predicted_caption)
                pred_vec_text.append(predicted_caption_text)

        for j in range(len(images)):
            save_image_with_caption(images[j].cpu(), pred_vec_text[j], epoch_dir, j, pred_vec[j] , alp[j])
            if j==15:
                break

    avg_rouge = {
        "rouge1": np.mean([score["rouge1"].fmeasure for score in all_rouge_scores]),
        "rouge2": np.mean([score["rouge2"].fmeasure for score in all_rouge_scores]),
        "rougeL": np.mean([score["rougeL"].fmeasure for score in all_rouge_scores]),
    }

    print(f"Epoch {epoch}, Test Loss: {total_loss / len(test_loader):.4f}")
    print(f"ROUGE Scores: {avg_rouge}")

    # Return both loss and ROUGE-L as primary metrics
    return total_loss / len(test_loader), avg_rouge["rougeL"]


best_rougeL = -np.inf  # Track best ROUGE-L score

for epoch in range(EPOCHS):
    train_one_epoch(epoch)
    loss, rougeL = evaluate_and_save(epoch)

    # Save the model if it has the best ROUGE-L score so far
    if rougeL > best_rougeL:
        best_rougeL = rougeL
        save_metrics(epoch, loss, best_rougeL, f"{output_dir}/best/history.json")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'rougeL': rougeL,
        }, os.path.join(f'{output_dir}/best/model.bin'))
