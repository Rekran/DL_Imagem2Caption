import torch
from torchvision import transforms
from models import EncoderDecoder
from dataset import CustomDataset
from PIL import Image


def load_model(model_path, vocab_size, embed_size=400, hidden_size=512, num_layers=2):
    model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_caption(model, image_path, dataset):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        features = model.encoder(img)
        caps = model.decoder.generate_caption(features.unsqueeze(0), vocab=dataset.vocab)
        return ' '.join(caps)
