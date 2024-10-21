import matplotlib.pyplot as plt
import os
from torch.nn.utils.rnn import pad_sequence
import torch

def show_image(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()


# Função para salvar a imagem com a legenda
def save_image_with_caption(img, caption, save_dir, index):
    img = img.squeeze(0).permute(1, 2, 0).numpy()  # Converter para numpy e ajustar dimensões

    plt.figure()
    plt.imshow(img)
    plt.title(caption)
    plt.axis('off')

    # Salvar a imagem no diretório da época
    save_path = os.path.join(save_dir, f"image_{index}.png")
    plt.savefig(save_path)
    plt.close()



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