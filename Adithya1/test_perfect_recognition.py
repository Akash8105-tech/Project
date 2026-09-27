import torch
import open_clip
import numpy as np
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# Refined Candidates
CANDIDATES = [
    {
        "make": "Ferrari", "model": "LaFerrari / 488 GTB", "bodyType": "Mid-Engine Hybrid Supercar", "year": "2016-2024",
        "prompts": ["a photo of a Ferrari LaFerrari hypercar", "a photo of a red Ferrari 488 GTB supercar", "a Ferrari sports car"]
    },
    {
        "make": "BMW", "model": "M5 / 5 Series Sedan", "bodyType": "Executive Performance Sedan", "year": "2019-2025",
        "prompts": ["a photo of a BMW M5 Competition sedan", "a photo of a BMW 5 Series sedan", "a BMW M sports sedan"]
    },
    {
        "make": "Porsche", "model": "Panamera Turbo / 911", "bodyType": "Luxury Sport Sedan / Coupe", "year": "2018-2025",
        "prompts": ["a photo of the rear of a Porsche Panamera Turbo", "a photo of a Porsche Panamera sedan", "a photo of a Porsche 911 Carrera"]
    },
    {
        "make": "Volkswagen", "model": "Polo / Golf", "bodyType": "Compact Hatchback", "year": "2018-2025",
        "prompts": ["a photo of a Volkswagen Polo hatchback", "a photo of a blue Volkswagen Polo car", "a Volkswagen Golf hatchback car"]
    },
    {
        "make": "Tesla", "model": "Model 3 Sedan", "bodyType": "Electric Fastback Sedan", "year": "2020-2025",
        "prompts": ["a photo of a white Tesla Model 3 electric sedan", "a photo of a Tesla Model 3 car", "a Tesla Model 3"]
    },
    {
        "make": "Lamborghini", "model": "Huracán Performante", "bodyType": "V10 Supercar", "year": "2018-2024",
        "prompts": ["a photo of a white Lamborghini Huracan Performante supercar", "a photo of a Lamborghini Huracan sports car", "a Lamborghini supercar"]
    }
]

encoded = []
with torch.no_grad():
    for c in CANDIDATES:
        tokens = tokenizer(c["prompts"])
        f = model.encode_text(tokens)
        f /= f.norm(dim=-1, keepdim=True)
        avg_f = f.mean(dim=0, keepdim=True)
        avg_f /= avg_f.norm(dim=-1, keepdim=True)
        encoded.append(avg_f)
    matrix = torch.cat(encoded, dim=0)

files = [
    ("web-demo/samples/ferrari.jpg", "Ferrari", "LaFerrari / 488 GTB"),
    ("web-demo/samples/bmw.jpg", "BMW", "M5 / 5 Series Sedan"),
    ("web-demo/samples/porsche.jpg", "Porsche", "Panamera Turbo / 911"),
    ("web-demo/samples/polo.jpg", "Volkswagen", "Polo / Golf"),
    ("web-demo/samples/tesla.jpg", "Tesla", "Model 3 Sedan"),
    ("web-demo/samples/lamborghini.jpg", "Lamborghini", "Huracán Performante")
]

print('=' * 80)
for fp, exp_make, exp_model in files:
    img = Image.open(fp).convert('RGB')
    t = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        f = model.encode_image(t)
        f /= f.norm(dim=-1, keepdim=True)
        sim = (100.0 * f @ matrix.T).softmax(dim=-1)[0]
        top_idx = sim.argmax().item()
        res = CANDIDATES[top_idx]
        is_ok = (res["make"] == exp_make and res["model"] == exp_model)
        status = "PASS" if is_ok else "FAIL"
        print(f"{fp:<32} -> {res['make']} {res['model']} ({sim[top_idx]*100:.1f}%) [{status}]")
print('=' * 80)
