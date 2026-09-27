import torch
import open_clip
import io
import urllib.request
from PIL import Image

print("[Test] Initializing ViT-B-32 with Standard OpenAI Ensemble...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# Official OpenAI Zero-Shot Template Ensemble for Cars & Vehicles
TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of a clean {}.",
    "a close-up photo of a {}.",
    "a photo of a nice {}.",
    "a photo of a {} on the road.",
    "a photo of the {} car.",
    "a {} automobile.",
    "a {} vehicle."
]

CANDIDATES = [
    # Supercars & Sports Cars
    {"name": "Ferrari 488 GTB", "make": "Ferrari", "model": "488 GTB", "bodyType": "Mid-Engine Supercar", "year": "2019-2024", "notes": "Aerodynamic S-Duct front hood, deep side air channels, and Prancing Horse styling."},
    {"name": "Ferrari SF90 Stradale", "make": "Ferrari", "model": "SF90 Stradale", "bodyType": "Hybrid Supercar", "year": "2021-2025", "notes": "C-shaped matrix LED headlights, twin-turbo hybrid V8, and active rear aero."},
    {"name": "Ferrari Roma", "make": "Ferrari", "model": "Roma", "bodyType": "Grand Tourer Coupe", "year": "2020-2025", "notes": "Shark-nose front styling, body-colored grille, and sweeping grand tourer lines."},
    {"name": "Ferrari 458 Italia", "make": "Ferrari", "model": "458 Italia", "bodyType": "Supercar", "year": "2010-2016", "notes": "Triple exhaust pipes, aeroelastic winglets, and high-revving naturally aspirated V8."},

    {"name": "Porsche 911 Carrera", "make": "Porsche", "model": "911 Carrera", "bodyType": "Sports Coupe", "year": "2020-2025", "notes": "Iconic rear-engine teardrop silhouette, round matrix headlamps, and continuous rear lightbar."},
    {"name": "Porsche 911 GT3", "make": "Porsche", "model": "911 GT3", "bodyType": "Track Sports Car", "year": "2021-2025", "notes": "Swan-neck rear wing, vented hood nostrils, and wide track aero."},
    {"name": "Porsche 718 Cayman", "make": "Porsche", "model": "718 Cayman", "bodyType": "Mid-Engine Sports Coupe", "year": "2018-2025", "notes": "Mid-engine sports coupe with sharp side air scoops and four-point DRLs."},
    {"name": "Porsche Cayenne", "make": "Porsche", "model": "Cayenne", "bodyType": "Luxury SUV", "year": "2020-2025", "notes": "Sports car front styling, four-point LED headlights, and luxury SUV stance."},
    {"name": "Porsche Taycan", "make": "Porsche", "model": "Taycan", "bodyType": "Electric Sport Sedan", "year": "2020-2025", "notes": "Air-curtain headlights, four-door sports sedan flyline, and 800V fast charging."},

    {"name": "Lamborghini Huracan", "make": "Lamborghini", "model": "Huracán", "bodyType": "V10 Supercar", "year": "2016-2024", "notes": "Sharp hexagonal creases, Y-shaped LED lighting, and naturally aspirated V10."},
    {"name": "Lamborghini Aventador", "make": "Lamborghini", "model": "Aventador", "bodyType": "V12 Supercar", "year": "2018-2023", "notes": "Iconic scissor doors, massive side radiators, and flagship V12 sound."},
    {"name": "Lamborghini Urus", "make": "Lamborghini", "model": "Urus", "bodyType": "Super SUV", "year": "2019-2025", "notes": "Aggressive Y-bonnet intakes, coupe SUV roofline, and twin-turbo V8."},

    {"name": "Mercedes-AMG GT", "make": "Mercedes-Benz", "model": "AMG GT", "bodyType": "Sports Coupe", "year": "2020-2025", "notes": "Panamericana vertical chrome slat grille, long bonnet, and short rear deck."},
    {"name": "Mercedes-Benz S-Class", "make": "Mercedes-Benz", "model": "S-Class", "bodyType": "Luxury Sedan", "year": "2021-2025", "notes": "Stately chrome multi-slat grille, flush door handles, and Digital Light headlamps."},
    {"name": "Mercedes-Benz C-Class", "make": "Mercedes-Benz", "model": "C-Class", "bodyType": "Executive Sedan", "year": "2020-2025", "notes": "Star-pattern diamond grille, power domes on bonnet, and AMG Line styling."},
    {"name": "Mercedes-Benz G-Wagon G-Class", "make": "Mercedes-Benz", "model": "G-Class", "bodyType": "Luxury Off-Roader", "year": "2019-2025", "notes": "Iconic boxy silhouette, round headlamps, and side-exit exhaust."},

    {"name": "BMW 7 Series", "make": "BMW", "model": "7 Series", "bodyType": "Luxury Flagship Sedan", "year": "2020-2025", "notes": "Active dual kidney grille, slim laser headlights, and luxury long-wheelbase stance."},
    {"name": "BMW 5 Series", "make": "BMW", "model": "5 Series", "bodyType": "Executive Sedan", "year": "2020-2025", "notes": "Sculpted kidney grille, L-shaped LED daytime running lights, and balanced proportions."},
    {"name": "BMW 3 Series", "make": "BMW", "model": "3 Series", "bodyType": "Compact Executive Sedan", "year": "2020-2025", "notes": "Sporty short front overhang, sculpted dual kidney grilles, and driver-focused cockpit."},
    {"name": "BMW M4 Competition", "make": "BMW", "model": "M4", "bodyType": "Performance Coupe", "year": "2021-2025", "notes": "Vertical frameless dual kidney grilles, carbon fiber roof, and M quad exhausts."},
    {"name": "BMW X5", "make": "BMW", "model": "X5", "bodyType": "Luxury SAV / SUV", "year": "2020-2025", "notes": "Prominent upright kidney grilles, muscular shoulder line, and split tailgate."},

    {"name": "Audi R8", "make": "Audi", "model": "R8", "bodyType": "Mid-Engine Supercar", "year": "2019-2024", "notes": "Carbon sideblades, wide honeycomb grille, and naturally aspirated V10."},
    {"name": "Audi A6", "make": "Audi", "model": "A6", "bodyType": "Executive Sedan", "year": "2020-2025", "notes": "Singleframe hexagonal grille, matrix LED headlamps, and Quattro design."},
    {"name": "Audi A4", "make": "Audi", "model": "A4", "bodyType": "Compact Executive Sedan", "year": "2020-2025", "notes": "Singleframe grille, sharp LED daytime running lights, and clean shoulder lines."},

    {"name": "Tesla Model 3", "make": "Tesla", "model": "Model 3", "bodyType": "Electric Sedan", "year": "2020-2025", "notes": "Grille-less front fascia, panoramic glass roof, and flush door handles."},
    {"name": "Tesla Model Y", "make": "Tesla", "model": "Model Y", "bodyType": "Electric Crossover", "year": "2021-2025", "notes": "Elevated crossover stance, high roofline, and minimalist electric design."},
    {"name": "Tesla Cybertruck", "make": "Tesla", "model": "Cybertruck", "bodyType": "Electric Pickup Truck", "year": "2023-2025", "notes": "Stainless steel exoskeleton, angular triangular shape, and full-width lightbar."},

    {"name": "Suzuki Swift", "make": "Suzuki", "model": "Swift", "bodyType": "Compact Hatchback", "year": "2020-2025", "notes": "Gloss-black honeycomb hexagonal grille, L-shaped DRLs, and floating roof pillars."},
    {"name": "Maruti Suzuki Swift", "make": "Suzuki", "model": "Swift (India)", "bodyType": "Compact Hatchback", "year": "2020-2025", "notes": "Signature Indian urban hatchback with honeycomb grille and sporty stance."},
    {"name": "Suzuki Baleno", "make": "Suzuki", "model": "Baleno", "bodyType": "Hatchback", "year": "2020-2025", "notes": "NEXWave chrome-accent grille and aerodynamic hatchback silhouette."},
    {"name": "Suzuki Jimny", "make": "Suzuki", "model": "Jimny", "bodyType": "Compact 4x4 Off-Roader", "year": "2020-2025", "notes": "Retro boxy styling, five-slot upright vertical grille, and round headlamps."},

    {"name": "Toyota Fortuner", "make": "Toyota", "model": "Fortuner", "bodyType": "Rugged SUV", "year": "2019-2025", "notes": "Bold chrome grille, high approach angle, and ladder-frame chassis."},
    {"name": "Toyota Camry", "make": "Toyota", "model": "Camry", "bodyType": "Hybrid Sedan", "year": "2019-2025", "notes": "Keen Look front styling, wide lower grille slats, and refined sedan stance."},
    {"name": "Toyota Supra", "make": "Toyota", "model": "GR Supra", "bodyType": "Sports Coupe", "year": "2020-2025", "notes": "Double-bubble roof, central air intake, and athletic rear ducktail spoiler."},

    {"name": "Hyundai Creta", "make": "Hyundai", "model": "Creta", "bodyType": "Compact SUV", "year": "2020-2025", "notes": "Parametric Jewel grille with integrated hidden DRLs and split headlamps."},
    {"name": "Hyundai i20", "make": "Hyundai", "model": "i20", "bodyType": "Premium Hatchback", "year": "2020-2025", "notes": "Sensuous sportiness styling, cascading black grille, and Z-shaped taillights."},
    {"name": "Hyundai Verna", "make": "Hyundai", "model": "Verna", "bodyType": "Sedan", "year": "2020-2025", "notes": "Fastback roofline, horizon LED lightbar, and parametric jewel accents."},

    {"name": "Honda Civic", "make": "Honda", "model": "Civic", "bodyType": "Sedan", "year": "2020-2025", "notes": "Low beltline, sleek fastback profile, and honeycomb dashboard styling."},
    {"name": "Honda City", "make": "Honda", "model": "City", "bodyType": "Sedan", "year": "2020-2025", "notes": "Solid Wing Face chrome grille and full LED headlamps."},

    {"name": "Ford Mustang", "make": "Ford", "model": "Mustang GT", "bodyType": "Muscle Car", "year": "2018-2025", "notes": "Tri-bar LED daytime lights, muscular haunches, and aggressive pony grille."},
    {"name": "Chevrolet Corvette", "make": "Chevrolet", "model": "Corvette C8", "bodyType": "Mid-Engine Sports Car", "year": "2020-2025", "notes": "Mid-engine proportions, dramatic side air intakes, and sharp edge lines."},

    {"name": "Tata Nexon", "make": "Tata", "model": "Nexon", "bodyType": "Compact SUV", "year": "2020-2025", "notes": "Connected LED light bar, parametric front grille, and coupe SUV stance."},
    {"name": "Mahindra Thar", "make": "Mahindra", "model": "Thar", "bodyType": "Off-Road 4x4", "year": "2020-2025", "notes": "Vertical slat retro grille, round headlights, and flared wheel arches."}
]

# Non-car candidate
NON_CAR = {"name": "cup of coffee, furniture, or non-vehicle object", "make": "Not Detected", "model": "Not Detected", "bodyType": "Non-Vehicle", "year": "N/A", "notes": "No vehicle detected in image."}
ALL_CANDIDATES = CANDIDATES + [NON_CAR]

print(f"[Test] Encoding {len(ALL_CANDIDATES)} candidates with {len(TEMPLATES)} templates each...")
encoded_tensors = []
with torch.no_grad():
    for cand in ALL_CANDIDATES:
        cand_prompts = [t.format(cand["name"]) for t in TEMPLATES]
        tokens = tokenizer(cand_prompts)
        f = model.encode_text(tokens)
        f /= f.norm(dim=-1, keepdim=True)
        avg_f = f.mean(dim=0, keepdim=True)
        avg_f /= avg_f.norm(dim=-1, keepdim=True)
        encoded_tensors.append(avg_f)
    matrix = torch.cat(encoded_tensors, dim=0)

# Test Suite
test_samples = [
    ("https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=600&auto=format&fit=crop&q=80", "Red Ferrari", "Ferrari"),
    ("https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=600&auto=format&fit=crop&q=80", "Black Porsche 911", "Porsche"),
    ("https://images.unsplash.com/photo-1555215695-3004980ad54e?w=600&auto=format&fit=crop&q=80", "Silver BMW", "BMW"),
    ("https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?w=600&auto=format&fit=crop&q=80", "White Suzuki Swift", "Suzuki"),
    ("https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=600&auto=format&fit=crop&q=80", "Blue Tesla Model 3", "Tesla"),
    ("https://images.unsplash.com/photo-1617814076367-b759c7d7e738?w=600&auto=format&fit=crop&q=80", "Green Mercedes AMG GT", "Mercedes-Benz"),
    ("https://images.unsplash.com/photo-1544829099-b9a0c07fad1a?w=600&auto=format&fit=crop&q=80", "Yellow Lamborghini", "Lamborghini"),
    ("https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=600&auto=format&fit=crop&q=80", "Coffee Cup", "Not Detected")
]

print("\n" + "=" * 90)
print(f"{'Input Target':<24} | {'Predicted Make & Model':<40} | {'Score':<8} | {'Status'}")
print("=" * 90)

for url, label, exp_make in test_samples:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    img_data = urllib.request.urlopen(req, timeout=10).read()
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        img_feat = model.encode_image(tensor)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        sim = (100.0 * img_feat @ matrix.T).softmax(dim=-1)[0]
        top_idx = sim.argmax().item()
        top_score = sim[top_idx].item()
        res = ALL_CANDIDATES[top_idx]
        
        is_ok = (res['make'] == exp_make)
        status_str = "PASS" if is_ok else "FAIL"
        full_pred = f"{res['make']} {res['model']}"
        print(f"{label:<24} | {full_pred:<40} | {top_score*100:5.1f}%  | [{status_str}]")

print("=" * 90)
