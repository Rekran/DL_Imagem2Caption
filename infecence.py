import torch
import torchvision.transforms as T
from PIL import Image
from models import EncoderDecoder  
from vocabulary import Vocabulary
from utils import show_image
import numpy as np

# --- Configurações de caminho ---
MODEL_PATH = "models/image-cap-model_flip_random_rouge_8k_attetion_512x512/best/model.bin"
VOCAB_PATH = "models/image-cap-model_flip_random_rouge_8k_attetion_512x512/vocab/vocab.json"

# --- Transformações da imagem ---
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])


def load_model(model_path, vocab_path, device):
    """Carrega o modelo e o vocabulário."""
    vocab = Vocabulary(1)
    vocab.load_vocab(vocab_path)
  
    # Inicializar modelo
    model = EncoderDecoder(
        embed_size=300,
        vocab_size=len(vocab),
        attention_dim=512,
        encoder_dim=2048,
        decoder_dim=512
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Modelo carregado com sucesso!")
    
    return model, vocab

def generate_caption(image_path, model, vocab, device):
    """Gera a legenda para uma imagem fornecida."""

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Adicionar dimensão batch

    with torch.no_grad():
        features = model.encoder(image_tensor)
        result, _ = model.decoder.generate_caption(features, vocab=vocab, max_len=20)
    return result

def main(path):
    """Função principal para rodar a inferência."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model, vocab = load_model(MODEL_PATH, VOCAB_PATH, device)

    # Gerar legenda
    caption = generate_caption(path, model, vocab, device)
    print(f"Legenda Gerada: {caption}")

    # Exibir imagem com a legenda
    image = Image.open(path).convert("RGB")
    show_image(T.ToTensor()(image), title=caption)

if __name__ == "__main__":

    main("teste.jpg")
