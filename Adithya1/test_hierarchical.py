import torch
import open_clip
import io
import urllib.request
import numpy as np
from PIL import Image

print("[Test] Initializing ViT-B-32 Hierarchical Engine...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# Stage 1: Brand & Type Candidates
BRANDS = [
    {"brand": "Ferrari", "prompts": ["a photo of a Ferrari car", "a Ferrari supercar", "a Ferrari sports vehicle", "a Ferrari automobile"]},
    {"brand": "Porsche", "prompts": ["a photo of a Porsche car", "a Porsche sports car", "a Porsche vehicle", "a Porsche automobile"]},
    {"brand": "BMW", "prompts": ["a photo of a BMW car", "a BMW luxury vehicle", "a BMW sedan or coupe", "a BMW automobile"]},
    {"brand": "Suzuki", "prompts": ["a photo of a Suzuki car", "a Maruti Suzuki car", "a Suzuki hatchback or mini car", "a Suzuki automobile"]},
    {"brand": "Tesla", "prompts": ["a photo of a Tesla electric car", "a Tesla electric vehicle", "a Tesla Model car", "a Tesla automobile"]},
    {"brand": "Mercedes-Benz", "prompts": ["a photo of a Mercedes-Benz car", "a Mercedes luxury vehicle", "a Mercedes-Benz AMG", "a Mercedes automobile"]},
    {"brand": "Lamborghini", "prompts": ["a photo of a Lamborghini supercar", "a Lamborghini exotic sports car", "a Lamborghini vehicle"]},
    {"brand": "Audi", "prompts": ["a photo of an Audi car", "an Audi luxury sedan or SUV", "an Audi vehicle", "an Audi automobile"]},
    {"brand": "Toyota", "prompts": ["a photo of a Toyota car", "a Toyota vehicle", "a Toyota SUV or sedan", "a Toyota automobile"]},
    {"brand": "Hyundai", "prompts": ["a photo of a Hyundai car", "a Hyundai SUV or sedan", "a Hyundai vehicle", "a Hyundai automobile"]},
    {"brand": "Honda", "prompts": ["a photo of a Honda car", "a Honda sedan or hatchback", "a Honda vehicle", "a Honda automobile"]},
    {"brand": "Ford", "prompts": ["a photo of a Ford car", "a Ford Mustang or truck", "a Ford vehicle", "a Ford automobile"]},
    {"brand": "Chevrolet", "prompts": ["a photo of a Chevrolet Corvette or car", "a Chevy vehicle", "a Chevrolet automobile"]},
    {"brand": "Tata", "prompts": ["a photo of a Tata Motors car", "a Tata Nexon or Harrier SUV", "a Tata vehicle"]},
    {"brand": "Mahindra", "prompts": ["a photo of a Mahindra SUV", "a Mahindra Thar 4x4", "a Mahindra vehicle"]},
    {"brand": "Not Detected", "prompts": [
        "a photo of a cup of coffee, mug, room, table, chair, food, person, animal, or non-vehicle object",
        "a photo of an indoor room with furniture and no cars",
        "a mug of coffee on a table",
        "a human face or person"
    ]}
]

# Brand Embeddings
brand_feats = []
with torch.no_grad():
    for b in BRANDS:
        tokens = tokenizer(b["prompts"])
        f = model.encode_text(tokens)
        f /= f.norm(dim=-1, keepdim=True)
        avg_f = f.mean(dim=0, keepdim=True)
        avg_f /= avg_f.norm(dim=-1, keepdim=True)
        brand_feats.append(avg_f)
    brand_matrix = torch.cat(brand_feats, dim=0)

# Models per Brand
MODELS = {
    "Ferrari": [
        {"model": "488 GTB / F8 Tributo", "bodyType": "Mid-Engine Supercar", "year": "2019-2024",
         "notes": "Aerodynamic S-Duct front bonnet, deep side sculpted air scoops, central twin exhausts, and signature Prancing Horse low-slung stance.",
         "prompts": ["a photo of a Ferrari 488 GTB", "a Ferrari F8 Tributo", "a Ferrari 488 supercar", "a Ferrari 458 Italia"]},
        {"model": "SF90 Stradale / 296 GTB", "bodyType": "Hybrid Supercar", "year": "2021-2025",
         "notes": "C-shaped matrix LED headlights, suspended rear wing, active aerodynamics, and ultra-wide low profile stance.",
         "prompts": ["a photo of a Ferrari SF90 Stradale", "a Ferrari 296 GTB", "a Ferrari SF90"]},
        {"model": "Roma / 812 Superfast", "bodyType": "Grand Tourer Coupe", "year": "2020-2025",
         "notes": "Minimalist shark-nose front styling, perforated body-colored grille, sweeping fastback silhouette.",
         "prompts": ["a photo of a Ferrari Roma", "a Ferrari 812 Superfast"]}
    ],
    "Porsche": [
        {"model": "911 Carrera / Turbo", "bodyType": "Rear-Engine Sports Coupe", "year": "2020-2025",
         "notes": "Iconic teardrop flyline, four-point LED matrix oval headlamps, wide rear fenders, and continuous rear lightbar.",
         "prompts": ["a photo of a Porsche 911 Carrera", "a Porsche 911 Turbo", "a Porsche 911 coupe", "a classic modern Porsche 911"]},
        {"model": "718 Cayman / Boxster", "bodyType": "Mid-Engine Sports Coupe", "year": "2018-2024",
         "notes": "Mid-engine sports coupe with sharp side air intakes, sculpted front wings, and four-point DRL headlights.",
         "prompts": ["a photo of a Porsche 718 Cayman", "a Porsche Boxster", "a Porsche 718"]},
        {"model": "Cayenne / Macan", "bodyType": "Performance Luxury SUV", "year": "2020-2025",
         "notes": "Sports car front fascia, four-point LED headlights, muscular shoulder line, and continuous rear LED strip.",
         "prompts": ["a photo of a Porsche Cayenne SUV", "a Porsche Macan SUV"]},
        {"model": "Taycan / Panamera", "bodyType": "Electric / Luxury Sport Sedan", "year": "2020-2025",
         "notes": "Air-curtain headlights, four-door sports sedan flyline, and flush aerodynamic door handles.",
         "prompts": ["a photo of a Porsche Taycan electric sedan", "a Porsche Panamera luxury sport sedan"]}
    ],
    "BMW": [
        {"model": "7 Series / 5 Series Sedan", "bodyType": "Executive Luxury Sedan", "year": "2020-2025",
         "notes": "Active dual kidney grille with vertical chrome slats, slim laser LED headlights, sculpted bonnet power domes, and luxury long-wheelbase stance.",
         "prompts": ["a photo of a BMW 7 Series sedan", "a BMW 5 Series sedan", "a BMW 7 Series or 5 Series luxury car"]},
        {"model": "3 Series / 4 Series Gran Coupe", "bodyType": "Compact Executive Sedan", "year": "2020-2025",
         "notes": "Sporty short front overhang, sculpted dual kidney grilles, aggressive front air curtains, and slim LED headlights.",
         "prompts": ["a photo of a BMW 3 Series sedan", "a BMW 4 Series Gran Coupe", "a BMW 3 Series"]},
        {"model": "M4 / M3 Competition", "bodyType": "Performance Sports Coupe", "year": "2021-2025",
         "notes": "Vertical frameless dual kidney grilles, carbon fiber roof, wide flared fenders, and M quad exhaust outlets.",
         "prompts": ["a photo of a BMW M4 Competition", "a BMW M3 sports sedan", "a BMW M4 sports coupe"]},
        {"model": "X5 / X3 / X7", "bodyType": "Luxury Sports Activity Vehicle (SAV)", "year": "2020-2025",
         "notes": "Prominent upright kidney grilles, muscular shoulder line, high ride height, and split two-piece powered tailgate.",
         "prompts": ["a photo of a BMW X5 SUV", "a BMW X3 SUV", "a BMW X7 SUV"]}
    ],
    "Suzuki": [
        {"model": "Swift (4th Gen)", "bodyType": "Compact Hatchback", "year": "2020-2025",
         "notes": "Signature gloss-black honeycomb hexagonal grille with central Suzuki badge, stylish L-shaped LED daytime running lights, floating blacked-out roof pillars, and athletic compact silhouette.",
         "prompts": ["a photo of a Suzuki Swift hatchback", "a Maruti Suzuki Swift", "a Suzuki Swift car", "a Swift compact hatchback"]},
        {"model": "Baleno / Dzire", "bodyType": "Hatchback / Compact Sedan", "year": "2020-2025",
         "notes": "NEXWave front grille with chrome underline, LED projector headlamps, and aerodynamic hatchback proportions.",
         "prompts": ["a photo of a Suzuki Baleno", "a Maruti Dzire", "a Suzuki Baleno hatchback"]},
        {"model": "Jimny", "bodyType": "Compact 4x4 Off-Roader", "year": "2020-2025",
         "notes": "Retro boxy styling, five-slot upright vertical grille, round headlamps, and flared rugged fender arches.",
         "prompts": ["a photo of a Suzuki Jimny 4x4", "a Maruti Jimny", "a Suzuki Jimny mini SUV"]},
        {"model": "Brezza / Grand Vitara", "bodyType": "Compact SUV", "year": "2021-2025",
         "notes": "Geometric front grille with chrome inserts, dual LED DRLs, skid plates, and dual-tone floating roof.",
         "prompts": ["a photo of a Maruti Suzuki Brezza SUV", "a Suzuki Grand Vitara SUV"]}
    ],
    "Tesla": [
        {"model": "Model 3 Sedan", "bodyType": "Electric Sedan", "year": "2020-2025",
         "notes": "Minimalist grille-less aerodynamic front fascia, panoramic glass canopy, flush door handles, and sleek fastback profile.",
         "prompts": ["a photo of a Tesla Model 3 electric sedan", "a Tesla Model 3 car", "a Tesla Model 3 fastback"]},
        {"model": "Model Y / Model X", "bodyType": "Electric Crossover / SUV", "year": "2021-2025",
         "notes": "Elevated aerodynamic crossover stance, closed front bumper, flush glass roof, and high-efficiency aero wheel covers.",
         "prompts": ["a photo of a Tesla Model Y electric crossover", "a Tesla Model X SUV", "a Tesla Model Y"]},
        {"model": "Cybertruck", "bodyType": "Electric Pickup Truck", "year": "2023-2025",
         "notes": "Unpainted stainless steel exoskeleton, angular triangular geometric design, and full-width front LED light bar.",
         "prompts": ["a photo of a Tesla Cybertruck electric truck", "a Tesla Cybertruck"]}
    ],
    "Mercedes-Benz": [
        {"model": "AMG GT / SL Roadster", "bodyType": "Performance Sports Coupe", "year": "2020-2025",
         "notes": "Panamericana vertical chrome slat grille, long sweeping bonnet, short rear deck, and wide stance.",
         "prompts": ["a photo of a Mercedes-AMG GT sports coupe", "a Mercedes SL Roadster", "a Mercedes-AMG GT"]},
        {"model": "S-Class (W223) / Maybach", "bodyType": "Flagship Luxury Sedan", "year": "2021-2025",
         "notes": "Flush-fitting door handles, stately chrome multi-slat grille with upright star, and triangular Digital Light headlamps.",
         "prompts": ["a photo of a Mercedes-Benz S-Class luxury sedan", "a Mercedes Maybach S-Class"]},
        {"model": "C-Class / E-Class", "bodyType": "Executive Sedan", "year": "2020-2025",
         "notes": "Star-pattern diamond radiator grille, power domes on bonnet, dynamic AMG apron, and horizontal split LED rear lamps.",
         "prompts": ["a photo of a Mercedes-Benz C-Class", "a Mercedes-Benz E-Class sedan"]},
        {"model": "G-Class (G-Wagon)", "bodyType": "Luxury 4x4 Off-Roader", "year": "2019-2025",
         "notes": "Iconic boxy silhouette, exposed door hinges, round headlights, exterior side exhaust pipes, and rear-mounted spare wheel.",
         "prompts": ["a photo of a Mercedes-Benz G-Class G-Wagon", "a Mercedes G63 AMG 4x4"]}
    ],
    "Lamborghini": [
        {"model": "Huracán / Gallardo", "bodyType": "V10 Supercar", "year": "2016-2024",
         "notes": "Hexagonal styling language, razor-sharp geometric creases, Y-shaped LED lighting signatures, and extreme low-slung roofline.",
         "prompts": ["a photo of a Lamborghini Huracan supercar", "a Lamborghini Huracan sports coupe", "a Lamborghini Gallardo"]},
        {"model": "Aventador / Revuelto", "bodyType": "V12 Flagship Supercar", "year": "2018-2025",
         "notes": "Iconic scissor doors, massive side air scoops, central aggressive high-mounted exhaust, and aerospace-inspired lines.",
         "prompts": ["a photo of a Lamborghini Aventador supercar", "a Lamborghini Revuelto V12 hypercar"]},
        {"model": "Urus", "bodyType": "Super Sport SUV", "year": "2019-2025",
         "notes": "Hexagonal styling, frameless doors, coupe roofline, aggressive front Y-bonnet intakes, and massive carbon ceramic brakes.",
         "prompts": ["a photo of a Lamborghini Urus super SUV", "a Lamborghini Urus"]}
    ],
    "Audi": [
        {"model": "A4 / A6 Sedan", "bodyType": "Executive Sedan", "year": "2019-2025",
         "notes": "Wide Singleframe hexagonal grille, razor-sharp matrix LED daytime signatures, and clean horizontal shoulder character line.",
         "prompts": ["a photo of an Audi A6 sedan", "an Audi A4 sedan", "an Audi luxury sedan"]},
        {"model": "R8 V10 Performance", "bodyType": "Mid-Engine Supercar", "year": "2019-2024",
         "notes": "Signature carbon sideblades, wide honeycomb front grille with three flat hood slits, and aggressive rear oval tailpipes.",
         "prompts": ["a photo of an Audi R8 supercar", "an Audi R8 V10 coupe"]},
        {"model": "Q5 / Q7 / Q8", "bodyType": "Luxury SUV", "year": "2020-2025",
         "notes": "Octagonal Singleframe grille, high shoulder line, Quattro flared wheel arches, and continuous rear animated LED light strip.",
         "prompts": ["a photo of an Audi Q5 SUV", "an Audi Q7 SUV", "an Audi Q8"]}
    ]
}

# Precompute Model Embeddings
model_matrices = {}
with torch.no_grad():
    for brand, m_list in MODELS.items():
        m_feats = []
        for m in m_list:
            tokens = tokenizer(m["prompts"])
            f = model.encode_text(tokens)
            f /= f.norm(dim=-1, keepdim=True)
            avg_f = f.mean(dim=0, keepdim=True)
            avg_f /= avg_f.norm(dim=-1, keepdim=True)
            m_feats.append(avg_f)
        model_matrices[brand] = (torch.cat(m_feats, dim=0), m_list)

def detect_car_paint_color(pil_img):
    # Center crop vehicle body (center 60% x 60%)
    w, h = pil_img.size
    crop_box = (int(w * 0.20), int(h * 0.20), int(w * 0.80), int(h * 0.80))
    cropped = pil_img.crop(crop_box).resize((100, 100))
    
    # Convert to HSV in numpy
    img_rgb = np.array(cropped) / 255.0
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c
    
    # Value / Brightness
    v = max_c
    
    # Saturation
    s = np.zeros_like(v)
    mask = v > 0
    s[mask] = delta[mask] / v[mask]
    
    # Hue in degrees 0-360
    h_deg = np.zeros_like(v)
    mask_r = (delta > 0) & (max_c == r)
    h_deg[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    mask_g = (delta > 0) & (max_c == g)
    h_deg[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    mask_b = (delta > 0) & (max_c == b)
    h_deg[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    
    # Ignore dark shadow pixels (v < 0.15) and pure white glare (v > 0.95 and s < 0.05)
    valid = (v >= 0.12) & (v <= 0.98)
    if not np.any(valid):
        return ("Metallic Silver / Titanium Grey", "#A0A5AA")
    
    val_s = s[valid]
    val_v = v[valid]
    val_h = h_deg[valid]
    
    mean_s = np.mean(val_s)
    mean_v = np.mean(val_v)
    
    # If low saturation -> Monochrome spectrum (Black, Grey/Silver, White)
    if mean_s < 0.22:
        if mean_v < 0.32:
            return ("Gloss Black / Obsidian", "#18181B")
        elif mean_v > 0.72:
            return ("Pearl White / Arctic White", "#F8FAFC")
        else:
            return ("Metallic Silver / Titanium Grey", "#A0A5AA")
    
    # High saturation -> Find dominant hue
    sat_valid = valid & (s > 0.20)
    if not np.any(sat_valid):
        return ("Metallic Silver / Titanium Grey", "#A0A5AA")
        
    sat_hues = h_deg[sat_valid]
    median_h = np.median(sat_hues)
    
    if median_h < 15 or median_h >= 340:
        return ("Crimson Red / Rosso Corsa", "#DC2626")
    elif 15 <= median_h < 45:
        return ("Papaya Orange / Sunset Amber", "#EA580C")
    elif 45 <= median_h < 75:
        return ("Racing Yellow / Gold", "#EAB308")
    elif 75 <= median_h < 170:
        return ("British Racing Green / Emerald", "#16A34A")
    elif 170 <= median_h < 265:
        return ("Deep Metallic Blue / Royal Navy", "#2563EB")
    elif 265 <= median_h < 340:
        return ("Metallic Purple / Violet", "#7C3AED")
    else:
        return ("Crimson Red / Rosso Corsa", "#DC2626")

test_samples = [
    ("https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=600&auto=format&fit=crop&q=80", "Red Ferrari", "Ferrari", "488 GTB / F8 Tributo", "Crimson Red / Rosso Corsa"),
    ("https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=600&auto=format&fit=crop&q=80", "Black Porsche 911", "Porsche", "911 Carrera / Turbo", "Gloss Black / Obsidian"),
    ("https://images.unsplash.com/photo-1555215695-3004980ad54e?w=600&auto=format&fit=crop&q=80", "Silver BMW", "BMW", "7 Series / 5 Series Sedan", "Metallic Silver / Titanium Grey"),
    ("https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?w=600&auto=format&fit=crop&q=80", "White Suzuki Swift", "Suzuki", "Swift (4th Gen)", "Pearl White / Arctic White"),
    ("https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=600&auto=format&fit=crop&q=80", "Blue Tesla Model 3", "Tesla", "Model 3 Sedan", "Deep Metallic Blue / Royal Navy"),
    ("https://images.unsplash.com/photo-1617814076367-b759c7d7e738?w=600&auto=format&fit=crop&q=80", "Green Mercedes AMG GT", "Mercedes-Benz", "AMG GT / SL Roadster", "British Racing Green / Emerald"),
    ("https://images.unsplash.com/photo-1544829099-b9a0c07fad1a?w=600&auto=format&fit=crop&q=80", "Yellow Lamborghini", "Lamborghini", "Hurac\u00e1n / Gallardo", "Racing Yellow / Gold"),
    ("https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=600&auto=format&fit=crop&q=80", "Coffee Cup", "Not Detected", "Not Detected", "Not Detected")
]

print("\n" + "=" * 105)
print(f"{'Input Target':<22} | {'Predicted Make & Model':<35} | {'Detected Color':<30} | {'Status'}")
print("=" * 105)

for url, label, exp_brand, exp_model, exp_color in test_samples:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    img_data = urllib.request.urlopen(req, timeout=10).read()
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        img_feat = model.encode_image(tensor)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        
        # Step 1: Predict Brand
        b_sim = (100.0 * img_feat @ brand_matrix.T).softmax(dim=-1)[0]
        top_b_idx = b_sim.argmax().item()
        b_score = b_sim[top_b_idx].item()
        matched_brand = BRANDS[top_b_idx]["brand"]

        if matched_brand == "Not Detected":
            res_make = "Not Detected"
            res_model = "Not Detected"
            color_name = "Not Detected"
            confidence = f"High ({b_score*100:.1f}%)"
        else:
            res_make = matched_brand
            # Step 2: Predict Model within matched Brand (or top brands)
            if matched_brand in model_matrices:
                m_matrix, m_list = model_matrices[matched_brand]
                m_sim = (100.0 * img_feat @ m_matrix.T).softmax(dim=-1)[0]
                top_m_idx = m_sim.argmax().item()
                top_m = m_list[top_m_idx]
                res_model = top_m["model"]
            else:
                res_model = "General Model"
                
            # Step 3: Precise Color Extraction
            color_name, hex_code = detect_car_paint_color(img)

        is_ok = (res_make == exp_brand and (res_model == exp_model or exp_brand == "Not Detected"))
        status_str = "PASS" if is_ok else "FAIL"
        full_pred = f"{res_make} {res_model}"
        print(f"{label:<22} | {full_pred:<35} | {color_name:<30} | [{status_str}]")

print("=" * 105)
