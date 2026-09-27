import torch
import open_clip
import io
import urllib.request
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

CARS = [
    {
        'make': 'Ferrari', 'model': '488 GTB / F8 Tributo', 'bodyType': 'Mid-Engine Supercar', 'year': '2019-2024',
        'colour': 'Crimson Red / Rosso Corsa', 'hexColor': '#DC2626',
        'notes': 'Aerodynamic S-Duct front hood, deep sculpted side air channels, and signature Prancing Horse styling.',
        'prompts': ['a photo of a Ferrari 488 GTB supercar', 'a red Ferrari 488 sports car', 'a Ferrari F8 Tributo automobile']
    },
    {
        'make': 'Porsche', 'model': '911 Carrera / Turbo', 'bodyType': 'Sports Coupe', 'year': '2020-2025',
        'colour': 'Gloss Black / Obsidian', 'hexColor': '#18181B',
        'notes': 'Timeless teardrop flyline, four-point matrix LED headlamps, and rear-engine widebody fenders.',
        'prompts': ['a photo of a Porsche 911 Carrera sports coupe', 'a black Porsche 911 sports car', 'a classic Porsche 911 automobile']
    },
    {
        'make': 'BMW', 'model': '7 Series / 5 Series Sedan', 'bodyType': 'Executive Luxury Sedan', 'year': '2020-2025',
        'colour': 'Metallic Silver / Titanium Grey', 'hexColor': '#A0A5AA',
        'notes': 'Active dual kidney grille with vertical chrome slats, slim laser LED headlights, and luxury long-wheelbase stance.',
        'prompts': ['a photo of a BMW 7 Series or BMW 5 Series luxury sedan', 'a silver BMW executive sedan car', 'a BMW 5 Series or 7 Series automobile']
    },
    {
        'make': 'Suzuki', 'model': 'Swift (4th Gen)', 'bodyType': 'Compact Hatchback', 'year': '2020-2025',
        'colour': 'Pearl White / Arctic White', 'hexColor': '#F8FAFC',
        'notes': 'Signature gloss-black honeycomb hexagonal grille, stylish L-shaped LED DRLs, and floating blacked-out roof pillars.',
        'prompts': ['a photo of a Suzuki Swift compact hatchback', 'a white Maruti Suzuki Swift car', 'a Suzuki Swift urban hatchback automobile']
    },
    {
        'make': 'Tesla', 'model': 'Model 3 Sedan', 'bodyType': 'Electric Fastback Sedan', 'year': '2021-2025',
        'colour': 'Deep Metallic Blue', 'hexColor': '#2563EB',
        'notes': 'Minimalist grille-less aerodynamic front fascia, panoramic glass canopy roof, and flush electronic door handles.',
        'prompts': ['a photo of a Tesla Model 3 electric sedan', 'a blue Tesla Model 3 car', 'a Tesla Model 3 fastback automobile']
    },
    {
        'make': 'Not Detected', 'model': 'Not Detected', 'bodyType': 'Non-Vehicle', 'year': 'N/A',
        'colour': 'Not Detected', 'hexColor': '#808080',
        'notes': 'Visual neural network scanned this image and detected no automobile or road vehicle.',
        'prompts': ['a photo of a coffee cup, mug, table, room, plant, food, animal, or non-vehicle object', 'a mug of coffee on a table']
    }
]

encoded_cars = []
with torch.no_grad():
    for car in CARS:
        tokens = tokenizer(car['prompts'])
        feats = model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
        avg_feat = feats.mean(dim=0, keepdim=True)
        avg_feat /= avg_feat.norm(dim=-1, keepdim=True)
        encoded_cars.append(avg_feat)
    car_text_matrix = torch.cat(encoded_cars, dim=0)

urls = [
    ('https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=600&auto=format&fit=crop&q=80', 'Red Ferrari', 'Ferrari'),
    ('https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=600&auto=format&fit=crop&q=80', 'Black Porsche 911', 'Porsche'),
    ('https://images.unsplash.com/photo-1555215695-3004980ad54e?w=600&auto=format&fit=crop&q=80', 'Silver BMW', 'BMW'),
    ('https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?w=600&auto=format&fit=crop&q=80', 'White Suzuki Swift', 'Suzuki'),
    ('https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=600&auto=format&fit=crop&q=80', 'Blue Tesla Model 3', 'Tesla'),
    ('https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=600&auto=format&fit=crop&q=80', 'Coffee Cup', 'Not Detected')
]

print('=' * 70)
for url, label, exp_make in urls:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    img_data = urllib.request.urlopen(req, timeout=10).read()
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        img_feat = model.encode_image(tensor)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        sim = (100.0 * img_feat @ car_text_matrix.T).softmax(dim=-1)[0]
        top_idx = sim.argmax().item()
        top_score = sim[top_idx].item()
        res = CARS[top_idx]
        is_ok = (res['make'] == exp_make)
        status_str = 'CORRECT' if is_ok else 'WRONG'
        print(f"{label:<24} -> {res['make']} {res['model']} ({top_score*100:.1f}%) [{status_str}]")
print('=' * 70)
