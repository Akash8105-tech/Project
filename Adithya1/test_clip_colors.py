import torch
import open_clip
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

COLOR_CLASSES = [
    {
        "name": "Crimson Red / Rosso Corsa",
        "hex": "#DC2626",
        "prompts": ["a photo of a red car", "a red automobile", "a car with red exterior paint", "a crimson red sports car"]
    },
    {
        "name": "Pure White / Pearl White",
        "hex": "#F8FAFC",
        "prompts": ["a photo of a white car", "a white automobile", "a car with white exterior paint", "a pearl white car"]
    },
    {
        "name": "Gloss Black / Obsidian",
        "hex": "#18181B",
        "prompts": ["a photo of a black car", "a black automobile", "a car with gloss black exterior paint", "a black sports car"]
    },
    {
        "name": "Metallic Silver / Titanium Grey",
        "hex": "#A0A5AA",
        "prompts": ["a photo of a silver car", "a silver automobile", "a car with metallic silver or grey paint", "a silver sedan"]
    },
    {
        "name": "Deep Metallic Blue / Navy",
        "hex": "#2563EB",
        "prompts": ["a photo of a blue car", "a blue automobile", "a car with blue exterior paint", "a deep blue car"]
    },
    {
        "name": "Racing Yellow / Gold",
        "hex": "#EAB308",
        "prompts": ["a photo of a yellow car", "a yellow automobile", "a car with yellow paint", "a bright yellow sports car"]
    },
    {
        "name": "Papaya Orange / Sunset Amber",
        "hex": "#EA580C",
        "prompts": ["a photo of an orange car", "an orange automobile", "a car with orange paint"]
    },
    {
        "name": "British Racing Green / Emerald",
        "hex": "#16A34A",
        "prompts": ["a photo of a green car", "a green automobile", "a car with green paint"]
    }
]

color_feats = []
with torch.no_grad():
    for c in COLOR_CLASSES:
        tokens = tokenizer(c["prompts"])
        f = model.encode_text(tokens)
        f /= f.norm(dim=-1, keepdim=True)
        avg_f = f.mean(dim=0, keepdim=True)
        avg_f /= avg_f.norm(dim=-1, keepdim=True)
        color_feats.append(avg_f)
    color_matrix = torch.cat(color_feats, dim=0)

files = [
    ('web-demo/samples/ferrari.jpg',     'Crimson Red / Rosso Corsa'),
    ('web-demo/samples/bmw.jpg',         'Metallic Silver / Titanium Grey'),
    ('web-demo/samples/porsche.jpg',     'Gloss Black / Obsidian'),
    ('web-demo/samples/polo.jpg',        'Deep Metallic Blue / Navy'),
    ('web-demo/samples/tesla.jpg',       'Pure White / Pearl White'),
    ('web-demo/samples/lamborghini.jpg', 'Pure White / Pearl White')
]

print('=' * 80)
for fp, exp in files:
    img = Image.open(fp).convert('RGB')
    t = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        f = model.encode_image(t)
        f /= f.norm(dim=-1, keepdim=True)
        sim = (100.0 * f @ color_matrix.T).softmax(dim=-1)[0]
        top_idx = sim.argmax().item()
        res = COLOR_CLASSES[top_idx]
        is_ok = (res["name"] == exp)
        status = "PASS" if is_ok else "FAIL"
        print(f"{fp:<32} -> {res['name']:<32} ({sim[top_idx]*100:5.1f}%) [{status}]")
print('=' * 80)
