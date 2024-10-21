import matplotlib.pyplot as plt
import os
from torch.nn.utils.rnn import pad_sequence
import torch
import json
import math
import numpy as np

def show_image(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()


def save_image_with_caption(img, caption, save_dir, index, result, attention_plot):
    img = img.squeeze(0).permute(1, 2, 0).numpy()

    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406  

    temp_image = np.clip(img, 0, 1)

    plt.figure()
    plt.imshow(temp_image)
    plt.title(caption)
    plt.axis('off')

    save_path = os.path.join(save_dir, f"image_{index}.png")
    plt.savefig(save_path)
    plt.close()

    plot_attention(temp_image, result, attention_plot, save_dir, index)


class CapsCollate:
    def __init__(self, pad_idx, batch_first=True):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        # Separar imagens e legendas
        imgs = [item[0].unsqueeze(0) for item in batch]
        captions = [item[1] for item in batch]

        # Empilhar imagens em um tensor (batch_size, 3, 224, 224)
        imgs = torch.cat(imgs, dim=0)

        # Preencher as legendas até o mesmo comprimento
        captions_padded = pad_sequence(captions, batch_first=self.batch_first, padding_value=self.pad_idx)

        return imgs, captions_padded



CORES = [
    "red", "blue", "green", "yellow", "black", "white", "brown", "gray", 
    "orange", "purple", "pink", "cyan", "magenta", "gold", "silver", 
    "beige", "ivory", "maroon", "navy", "teal", "violet", "lime"
]


def remove_colors(caption):
    words = caption.split() 
    filtered_words = [word for word in words if word.lower() not in CORES]
    return ' '.join(filtered_words)  


def save_metrics(epoch, loss, best_rougeL,filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            metrics_history = json.load(f)
    else:
        metrics_history = []

    metrics_history.append({
        'epoch': epoch,
        'loss': loss,
        'best_rougeL': best_rougeL
    })

    with open(filepath, 'w') as f:
        json.dump(metrics_history, f, indent=4)



def get_caps_from(features_tensors, model, vocab):
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to("cuda"))
        caps,alphas = model.decoder.generate_caption(features,vocab=vocab)
        caption = ' '.join(caps)
        show_image(features_tensors[0],title=caption)
    
    return caps,alphas

def plot_attention(img, result, attention_plot, save_dir, index):

    img[0] = img[0] * 0.229 + 0.485
    img[1] = img[1] * 0.224 + 0.456
    img[2] = img[2] * 0.225 + 0.406
    
    temp_image = np.clip(img, 0, 1)

    len_result = len(result)


    cols = 5  
    rows = math.ceil(len_result / cols)  
    fig_width = cols * 4  
    fig_height = rows * 4  

    fig = plt.figure(figsize=(fig_width, fig_height))

    for l in range(len_result):
        size = int(math.sqrt(attention_plot[l].size)) 
        temp_att = attention_plot[l].reshape(size, size)

        ax = fig.add_subplot(rows, cols, l + 1)
        ax.set_title(result[l])
        ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=ax.get_images()[0].get_extent())
        ax.axis('off') 

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'image_{index}_att.png'))
    plt.close(fig)