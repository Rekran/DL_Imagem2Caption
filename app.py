import os
import torch
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from models import EncoderDecoder  
from vocabulary import Vocabulary

# --- Configurações de caminho ---
MODEL_PATH = "models/image-cap-model_flip_random_rouge_8k_attetion_512x512/best/model.bin"
VOCAB_PATH = "models/image-cap-model_flip_random_rouge_8k_attetion_512x512/vocab/vocab.json"

# --- Transformações da imagem ---
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

app = FastAPI()

# Configurar o diretório estático
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

# Carregar modelo e vocabulário globalmente
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, vocab_path, device):
    """Carrega o modelo e o vocabulário."""
    vocab = Vocabulary(1)
    vocab.load_vocab(vocab_path)
  
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
    
    return model, vocab

model, vocab = load_model(MODEL_PATH, VOCAB_PATH, device)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Retorna a página inicial com um formulário de upload de arquivo."""
    return """
    <html>
        <head>
            <title>Gerador de Legendas de Imagens</title>
        </head>
        <body>
            <h1>Carregar Imagem para Legenda</h1>
            <form action="/uploadfile/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*" required>
                <input type="submit">
            </form>
        </body>
    </html>
    """

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    """Processa o arquivo enviado e gera uma legenda."""
    # Salvar a imagem temporariamente
    temp_file_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Gerar legenda
    caption = generate_caption(temp_file_path)

    # Retornar a resposta HTML com a imagem e a legenda
    return HTMLResponse(content=f"""
    <html>
        <head>
            <title>Legenda Gerada</title>
        </head>
        <body>
            <h1>{caption}</h1>
            <img src="/temp/{file.filename}" alt="Uploaded Image" style="max-width: 400px; max-height: 400px;">
            <br>
            <a href="/">Carregar outra imagem</a>
        </body>
    </html>
    """)

def generate_caption(image_path):
    """Gera a legenda para uma imagem fornecida."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Adicionar dimensão batch

    with torch.no_grad():
        features = model.encoder(image_tensor)
        result, _ = model.decoder.generate_caption(features, vocab=vocab, max_len=20)
    return (" ").join(result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
