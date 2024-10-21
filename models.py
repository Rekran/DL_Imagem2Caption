import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, fine_tune=False):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        
        if not fine_tune:
            for param in resnet.parameters():
                param.requires_grad_(False)
        else:
            for name, param in resnet.named_parameters():
                if "layer5" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images).view(images.size(0), -1)
        return self.embed(features)

    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, features, captions):
        embeds = self.embedding(captions[:,:-1])
        x = torch.cat((features.unsqueeze(1),embeds),dim=1)
        x, _ = self.lstm(x)
        return self.fcn(x)
    
    def generate_caption(self,inputs,hidden=None,max_len=30,vocab=None):
        # Inference part
        # Given the image features generate the captions

        batch_size = inputs.size(0)

        captions = []

        for i in range(max_len):
            output,hidden = self.lstm(inputs,hidden)
            output = self.fcn(output)
            output = output.view(batch_size,-1)


            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)

            #save the generated word
            captions.append(predicted_word_idx.item())

            #end if <EOS detected>
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break

            #send generated word as the next caption
            inputs = self.embedding(predicted_word_idx.unsqueeze(0))

        #covert the vocab idx to words and return sentence
        return [vocab.itos[idx] for idx in captions]
    
class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, drop_prob)

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)
