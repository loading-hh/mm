import itertools
from PIL import Image
import torch, open_clip

model_name = 'ViT-B-32' # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.tokenize
# tokenizer = open_clip.get_tokenizer(model_name)

ckpt = torch.load(fr"E:\BaiduNetdiskDownload\huggingface\remoteclip\RemoteCLIP-{model_name}.pt", map_location="cuda")
model.load_state_dict(ckpt)
# print(message)
print(model.ln_final.weight.device)
model = model.cuda().eval()
print(model.ln_final.weight.device)

text_queries = [
    ["A airport."],
    ["A busy airport with many airplanes."], 
    ["Satellite view of Hohai University."], 
    ["A building next to a lake."], 
    ["Many people in a stadium."], 
    ["a cute cat"],
    ]
text_queries = list(itertools.chain(*text_queries))
text = tokenizer(text_queries)
image = preprocess(Image.open(r"E:\BaiduNetdiskDownload\airport.jpg")).unsqueeze(0)

with torch.no_grad():
    image_features = model.encode_image(image.cuda())
    text_features = model.encode_text(text.cuda())
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

print(f'Predictions of {model_name}:')
for query, prob in zip(text_queries, text_probs):
    print(f"{query:<40} {prob * 100:5.1f}%")

def _freeze_modules(model):
    model.eval()
    for _, module in model.named_modules():
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
    return
    for name, module in self.model.named_modules():
        for frozen_name in self.frozen_modules:
            if name.startswith(frozen_name):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
                break
            
_freeze_modules(model)