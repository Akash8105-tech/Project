import torch
import open_clip
import io
import urllib.request
import numpy as np
from PIL import Image

print("[Test] Initializing ViT-B-32...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# Comprehensive Car Database (Over 60 popular models covering Sports, Luxury, Hatchbacks, Sedans, SUVs)
DATABASE = [
    # Ferrari
    {
        "make": "Ferrari", "model": "488 GTB / F8 Tributo", "bodyType": "Mid-Engine Supercar", "year": "2019-2024",
        "notes": "Aerodynamic S-Duct front bonnet, deep side sculpted air scoops, central twin exhausts, and signature Prancing Horse low-slung stance.",
        "prompts": ["a photo of a Ferrari 488 GTB supercar", "a Ferrari F8 Tributo sports car", "a Ferrari 488 supercar", "a modern Ferrari mid-engine sports car"]
    },
    {
        "make": "Ferrari", "model": "SF90 Stradale / 296 GTB", "bodyType": "Hybrid Supercar", "year": "2021-2025",
        "notes": "C-shaped matrix LED headlights, suspended rear wing, active aerodynamics, and ultra-wide low profile stance.",
        "prompts": ["a photo of a Ferrari SF90 Stradale supercar", "a Ferrari 296 GTB sports car", "a Ferrari SF90 hypercar"]
    },
    {
        "make": "Ferrari", "model": "Roma / 812 Superfast", "bodyType": "Grand Tourer Coupe", "year": "2020-2025",
        "notes": "Minimalist shark-nose front styling, perforated body-colored grille, sweeping fastback silhouette.",
        "prompts": ["a photo of a Ferrari Roma coupe", "a Ferrari 812 Superfast grand tourer"]
    },

    # Porsche
    {
        "make": "Porsche", "model": "911 Carrera / Turbo", "bodyType": "Rear-Engine Sports Coupe", "year": "2020-2025",
        "notes": "Iconic teardrop flyline, four-point LED matrix oval headlamps, wide rear fenders, and continuous rear lightbar.",
        "prompts": ["a photo of a Porsche 911 Carrera sports coupe", "a Porsche 911 Turbo car", "a classic modern Porsche 911 rear-engine sports car"]
    },
    {
        "make": "Porsche", "model": "718 Cayman / Boxster", "bodyType": "Mid-Engine Sports Coupe", "year": "2018-2024",
        "notes": "Mid-engine sports coupe with sharp side air intakes, sculpted front wings, and four-point DRL headlights.",
        "prompts": ["a photo of a Porsche 718 Cayman sports coupe", "a Porsche Boxster roadster", "a Porsche 718 sports car"]
    },
    {
        "make": "Porsche", "model": "Cayenne / Macan", "bodyType": "Performance Luxury SUV", "year": "2020-2025",
        "notes": "Sports car front fascia, four-point LED headlights, muscular shoulder line, and continuous rear LED strip.",
        "prompts": ["a photo of a Porsche Cayenne SUV", "a Porsche Macan luxury compact SUV", "a Porsche SUV vehicle"]
    },
    {
        "make": "Porsche", "model": "Taycan / Panamera", "bodyType": "Electric / Luxury Sport Sedan", "year": "2020-2025",
        "notes": "Air-curtain headlights, four-door sports sedan flyline, and flush aerodynamic door handles.",
        "prompts": ["a photo of a Porsche Taycan electric sedan", "a Porsche Panamera luxury sport sedan"]
    },

    # BMW
    {
        "make": "BMW", "model": "7 Series / 5 Series Sedan", "bodyType": "Executive Luxury Sedan", "year": "2020-2025",
        "notes": "Active dual kidney grille with vertical chrome slats, slim laser LED headlights, sculpted bonnet power domes, and luxury long-wheelbase stance.",
        "prompts": ["a photo of a BMW 7 Series or 5 Series luxury sedan", "a BMW executive sedan car", "a modern BMW 5 Series or 7 Series automobile"]
    },
    {
        "make": "BMW", "model": "3 Series / 4 Series Gran Coupe", "bodyType": "Compact Executive Sedan", "year": "2020-2025",
        "notes": "Sporty short front overhang, sculpted dual kidney grilles, aggressive front air curtains, and slim LED headlights.",
        "prompts": ["a photo of a BMW 3 Series sedan", "a BMW 4 Series Gran Coupe car", "a BMW 3 Series automobile"]
    },
    {
        "make": "BMW", "model": "M4 / M3 Competition", "bodyType": "Performance Sports Coupe", "year": "2021-2025",
        "notes": "Vertical frameless dual kidney grilles, carbon fiber roof, wide flared fenders, and M quad exhaust outlets.",
        "prompts": ["a photo of a BMW M4 Competition sports coupe", "a BMW M3 sports car", "a BMW M performance sports car"]
    },
    {
        "make": "BMW", "model": "X5 / X3 / X7", "bodyType": "Luxury Sports Activity Vehicle (SAV)", "year": "2020-2025",
        "notes": "Prominent upright kidney grilles, muscular shoulder line, high ride height, and split two-piece powered tailgate.",
        "prompts": ["a photo of a BMW X5 luxury SUV", "a BMW X3 SUV", "a BMW X7 large luxury SUV"]
    },

    # Suzuki / Maruti Suzuki
    {
        "make": "Suzuki", "model": "Swift (4th Gen)", "bodyType": "Compact Hatchback", "year": "2020-2025",
        "notes": "Signature gloss-black honeycomb hexagonal grille with central Suzuki badge, stylish L-shaped LED daytime running lights, floating blacked-out roof pillars, and athletic compact silhouette.",
        "prompts": ["a photo of a Suzuki Swift compact hatchback", "a Maruti Suzuki Swift car", "a Suzuki Swift urban hatchback automobile", "a Swift hatchback car"]
    },
    {
        "make": "Suzuki", "model": "Baleno / Dzire", "bodyType": "Hatchback / Compact Sedan", "year": "2020-2025",
        "notes": "NEXWave front grille with chrome underline, LED projector headlamps, and aerodynamic hatchback proportions.",
        "prompts": ["a photo of a Suzuki Baleno hatchback", "a Maruti Dzire compact sedan", "a Suzuki Baleno car"]
    },
    {
        "make": "Suzuki", "model": "Jimny", "bodyType": "Compact 4x4 Off-Roader", "year": "2020-2025",
        "notes": "Retro boxy styling, five-slot upright vertical grille, round headlamps, and flared rugged fender arches.",
        "prompts": ["a photo of a Suzuki Jimny 4x4 mini SUV", "a Maruti Suzuki Jimny off-roader", "a Suzuki Jimny car"]
    },
    {
        "make": "Suzuki", "model": "Brezza / Grand Vitara", "bodyType": "Compact SUV", "year": "2021-2025",
        "notes": "Geometric front grille with chrome inserts, dual LED DRLs, skid plates, and dual-tone floating roof.",
        "prompts": ["a photo of a Maruti Suzuki Brezza SUV", "a Suzuki Grand Vitara crossover SUV", "a Suzuki Brezza car"]
    },

    # Tesla
    {
        "make": "Tesla", "model": "Model 3 Sedan", "bodyType": "Electric Sedan", "year": "2020-2025",
        "notes": "Minimalist grille-less aerodynamic front fascia, panoramic glass canopy, flush door handles, and sleek fastback profile.",
        "prompts": ["a photo of a Tesla Model 3 electric sedan", "a Tesla Model 3 car", "a Tesla Model 3 automobile"]
    },
    {
        "make": "Tesla", "model": "Model Y / Model X", "bodyType": "Electric Crossover / SUV", "year": "2021-2025",
        "notes": "Elevated aerodynamic crossover stance, closed front bumper, flush glass roof, and high-efficiency aero wheel covers.",
        "prompts": ["a photo of a Tesla Model Y electric crossover", "a Tesla Model X SUV", "a Tesla Model Y car"]
    },
    {
        "make": "Tesla", "model": "Cybertruck", "bodyType": "Electric Pickup Truck", "year": "2023-2025",
        "notes": "Unpainted stainless steel exoskeleton, angular triangular geometric design, and full-width front LED light bar.",
        "prompts": ["a photo of a Tesla Cybertruck electric pickup", "a Tesla Cybertruck"]
    },

    # Mercedes-Benz
    {
        "make": "Mercedes-Benz", "model": "S-Class (W223) / Maybach", "bodyType": "Flagship Luxury Sedan", "year": "2021-2025",
        "notes": "Flush-fitting door handles, stately chrome multi-slat grille with upright star, and triangular Digital Light headlamps.",
        "prompts": ["a photo of a Mercedes-Benz S-Class luxury sedan", "a Mercedes S-Class flagship car", "a Maybach luxury sedan"]
    },
    {
        "make": "Mercedes-Benz", "model": "C-Class / E-Class", "bodyType": "Executive Sedan", "year": "2020-2025",
        "notes": "Star-pattern diamond radiator grille, power domes on bonnet, dynamic AMG apron, and horizontal split LED rear lamps.",
        "prompts": ["a photo of a Mercedes-Benz C-Class sedan", "a Mercedes-Benz E-Class luxury car", "a Mercedes C-Class automobile"]
    },
    {
        "make": "Mercedes-Benz", "model": "G-Class (G-Wagon)", "bodyType": "Luxury 4x4 Off-Roader", "year": "2019-2025",
        "notes": "Iconic boxy silhouette, exposed door hinges, round headlights, exterior side exhaust pipes, and rear-mounted spare wheel.",
        "prompts": ["a photo of a Mercedes-Benz G-Class G-Wagon SUV", "a Mercedes G63 AMG SUV", "a Mercedes G-Wagon 4x4"]
    },
    {
        "make": "Mercedes-Benz", "model": "AMG GT / SL Roadster", "bodyType": "Performance Sports Coupe", "year": "2020-2025",
        "notes": "Panamericana vertical chrome slat grille, long sweeping bonnet, short rear deck, and wide stance.",
        "prompts": ["a photo of a Mercedes-AMG GT sports coupe", "a Mercedes SL Roadster sports car"]
    },

    # Lamborghini
    {
        "make": "Lamborghini", "model": "Huracán / Gallardo", "bodyType": "V10 Supercar", "year": "2016-2024",
        "notes": "Hexagonal styling language, razor-sharp geometric creases, Y-shaped LED lighting signatures, and extreme low-slung roofline.",
        "prompts": ["a photo of a Lamborghini Huracan supercar", "a Lamborghini Huracan sports car", "a Lamborghini V10 supercar"]
    },
    {
        "make": "Lamborghini", "model": "Aventador / Revuelto", "bodyType": "V12 Flagship Supercar", "year": "2018-2025",
        "notes": "Iconic scissor doors, massive side air scoops, central aggressive high-mounted exhaust, and aerospace-inspired lines.",
        "prompts": ["a photo of a Lamborghini Aventador supercar", "a Lamborghini Revuelto V12 hypercar", "a Lamborghini flagship supercar"]
    },
    {
        "make": "Lamborghini", "model": "Urus", "bodyType": "Super Sport SUV", "year": "2019-2025",
        "notes": "Hexagonal styling, frameless doors, coupe roofline, aggressive front Y-bonnet intakes, and massive carbon ceramic brakes.",
        "prompts": ["a photo of a Lamborghini Urus super SUV", "a Lamborghini Urus high-performance SUV"]
    },

    # Audi
    {
        "make": "Audi", "model": "A4 / A6 Sedan", "bodyType": "Executive Sedan", "year": "2019-2025",
        "notes": "Wide Singleframe hexagonal grille, razor-sharp matrix LED daytime signatures, and clean horizontal shoulder character line.",
        "prompts": ["a photo of an Audi A6 luxury sedan", "a photo of an Audi A4 executive car", "an Audi A6 automobile"]
    },
    {
        "make": "Audi", "model": "R8 V10 Performance", "bodyType": "Mid-Engine Supercar", "year": "2019-2024",
        "notes": "Signature carbon sideblades, wide honeycomb front grille with three flat hood slits, and aggressive rear oval tailpipes.",
        "prompts": ["a photo of an Audi R8 supercar", "an Audi R8 V10 sports coupe", "an Audi R8 mid-engine supercar"]
    },
    {
        "make": "Audi", "model": "Q5 / Q7 / Q8", "bodyType": "Luxury SUV", "year": "2020-2025",
        "notes": "Octagonal Singleframe grille, high shoulder line, Quattro flared wheel arches, and continuous rear animated LED light strip.",
        "prompts": ["a photo of an Audi Q5 SUV", "an Audi Q7 luxury SUV", "an Audi Q8 coupe SUV"]
    },

    # Hyundai
    {
        "make": "Hyundai", "model": "Creta / Venue / Tucson", "bodyType": "SUV / Crossover", "year": "2020-2025",
        "notes": "Parametric Jewel pattern grille with integrated hidden DRLs, split LED headlamps, and sculpted aerodynamic character creases.",
        "prompts": ["a photo of a Hyundai Creta SUV", "a Hyundai Tucson crossover SUV", "a Hyundai Venue compact SUV", "a Hyundai Creta car"]
    },
    {
        "make": "Hyundai", "model": "i20 / Verna / Elantra", "bodyType": "Hatchback / Sedan", "year": "2020-2025",
        "notes": "Sensuous sportiness styling, cascading black front grille, fastback coupe-like roofline, and Z-shaped LED taillights.",
        "prompts": ["a photo of a Hyundai i20 premium hatchback", "a Hyundai Verna sedan", "a Hyundai Elantra sedan"]
    },

    # Toyota
    {
        "make": "Toyota", "model": "Fortuner / Land Cruiser", "bodyType": "Rugged 4x4 SUV", "year": "2019-2025",
        "notes": "Dominant chrome grille, high approach angle, muscular flared wheel arches, and heavy-duty ladder frame stance.",
        "prompts": ["a photo of a Toyota Fortuner SUV", "a Toyota Land Cruiser 4x4 SUV", "a Toyota Fortuner vehicle"]
    },
    {
        "make": "Toyota", "model": "Camry / Corolla / Prius", "bodyType": "Sedan / Hybrid", "year": "2019-2025",
        "notes": "Keen Look front styling, wide horizontal lower grille slats, slim LED headlights, and aerodynamic side character line.",
        "prompts": ["a photo of a Toyota Camry luxury sedan", "a Toyota Corolla sedan", "a Toyota Prius hybrid car"]
    },
    {
        "make": "Toyota", "model": "Supra (GR)", "bodyType": "Sports Coupe", "year": "2020-2025",
        "notes": "Double-bubble roof, central front air intake, ducktail rear spoiler, and athletic sports coupe dimensions.",
        "prompts": ["a photo of a Toyota GR Supra sports car", "a Toyota Supra coupe", "a Toyota GR Supra sports automobile"]
    },

    # Honda
    {
        "make": "Honda", "model": "Civic / City / Accord", "bodyType": "Sedan", "year": "2020-2025",
        "notes": "Solid Wing Face front chrome bar, jewel-eye LED headlights, low beltline, and clean sporty proportions.",
        "prompts": ["a photo of a Honda Civic sedan", "a Honda City sedan car", "a Honda Accord luxury sedan", "a Honda City automobile"]
    },

    # Ford & Chevrolet
    {
        "make": "Ford", "model": "Mustang GT / Dark Horse", "bodyType": "Fastback Muscle Car", "year": "2018-2025",
        "notes": "Iconic tri-bar LED daytime running lights, long hood, muscular rear haunches, and aggressive wide front grille.",
        "prompts": ["a photo of a Ford Mustang GT muscle car", "a Ford Mustang sports coupe", "a Ford Mustang fastback"]
    },
    {
        "make": "Chevrolet", "model": "Corvette Stingray (C8)", "bodyType": "Mid-Engine Sports Coupe", "year": "2020-2025",
        "notes": "Mid-engine proportions, dramatic triangular side scoops, angular quad exhaust tips, and sharp edge sculpting.",
        "prompts": ["a photo of a Chevrolet Corvette C8 sports car", "a Chevy Corvette Stingray coupe", "a Corvette C8 mid-engine sports car"]
    },

    # Tata Motors & Mahindra
    {
        "make": "Tata", "model": "Nexon / Harrier / Safari", "bodyType": "SUV", "year": "2020-2025",
        "notes": "Connected LED light bar, parametric grille, tri-arrow design accents, and high ground clearance SUV stance.",
        "prompts": ["a photo of a Tata Nexon SUV", "a Tata Harrier SUV car", "a Tata Safari SUV", "a Tata Nexon car"]
    },
    {
        "make": "Mahindra", "model": "Thar / XUV700 / Scorpio", "bodyType": "Rugged SUV / 4x4", "year": "2020-2025",
        "notes": "Iconic vertical slat grille, rugged off-road stance, round headlamps or C-shaped DRLs, and muscular body cladding.",
        "prompts": ["a photo of a Mahindra Thar 4x4 off-roader", "a Mahindra XUV700 SUV", "a Mahindra Scorpio SUV"]
    },

    # McLaren & Aston Martin
    {
        "make": "McLaren", "model": "720S / Artura", "bodyType": "Supercar", "year": "2018-2025",
        "notes": "Eye-socket headlight air intakes, dihedral doors, rear active wing, and teardrop carbon tub cockpit.",
        "prompts": ["a photo of a McLaren 720S supercar", "a McLaren Artura sports car", "a McLaren supercar"]
    },
    {
        "make": "Aston Martin", "model": "Vantage / DB11 / DB12", "bodyType": "Luxury Grand Tourer", "year": "2019-2025",
        "notes": "Iconic wide inverted Aston Martin grille, swan doors, slim horizontal LED light strip, and athletic muscular curves.",
        "prompts": ["a photo of an Aston Martin Vantage sports car", "an Aston Martin DB11 coupe", "an Aston Martin grand tourer"]
    },

    # Non-Vehicle / Validation Candidate
    {
        "make": "Not Detected", "model": "Not Detected", "bodyType": "Non-Vehicle", "year": "N/A",
        "notes": "Visual neural network scanned this image and detected no automobile or road vehicle.",
        "prompts": [
            "a photo of a cup of coffee, mug, beverage, drink, table, furniture, chair, desk, room, plant, pet, human, face, food, dish, or non-vehicle object",
            "a photo of an indoor room with furniture and no cars",
            "a photo of a cup of coffee on a table",
            "a photo of a person or human without any vehicle"
        ]
    }
]

print(f"[Test] Encoding {len(DATABASE)} candidates...")
encoded_list = []
with torch.no_grad():
    for item in DATABASE:
        tokens = tokenizer(item["prompts"])
        feats = model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
        avg_feat = feats.mean(dim=0, keepdim=True)
        avg_feat /= avg_feat.norm(dim=-1, keepdim=True)
        encoded_list.append(avg_feat)
    database_matrix = torch.cat(encoded_list, dim=0)

# Colors
COLOR_LIST = [
    {"name": "Crimson Red / Rosso Corsa", "hex": "#DC2626", "prompts": ["a photo of a red car with red exterior paint", "a crimson red sports car", "a red car body"]},
    {"name": "Metallic Silver / Titanium Grey", "hex": "#A0A5AA", "prompts": ["a photo of a silver car with metallic silver or grey paint", "a silver luxury car", "a grey car body"]},
    {"name": "Pearl White / Arctic White", "hex": "#F8FAFC", "prompts": ["a photo of a white car with pure white exterior body paint", "a white car", "a pearl white automobile"]},
    {"name": "Gloss Black / Obsidian", "hex": "#18181B", "prompts": ["a photo of a black car with gloss black exterior paint", "a black sports car", "a shiny black car"]},
    {"name": "Deep Metallic Blue / Royal Navy", "hex": "#2563EB", "prompts": ["a photo of a blue car with blue metallic paint", "a royal blue car", "a deep blue car"]},
    {"name": "Racing Yellow / Gold", "hex": "#EAB308", "prompts": ["a photo of a yellow car with yellow paint", "a yellow sports car", "a vibrant yellow automobile"]},
    {"name": "Papaya Orange / Sunset Amber", "hex": "#EA580C", "prompts": ["a photo of an orange car with orange paint", "an orange sports car", "a papaya orange car"]},
    {"name": "British Racing Green / Emerald", "hex": "#16A34A", "prompts": ["a photo of a green car with green paint", "an emerald green automobile"]},
    {"name": "Metallic Purple / Violet", "hex": "#7C3AED", "prompts": ["a photo of a purple car", "a violet sports car"]}
]

encoded_colors = []
with torch.no_grad():
    for col in COLOR_LIST:
        tokens = tokenizer(col["prompts"])
        feats = model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
        avg_feat = feats.mean(dim=0, keepdim=True)
        avg_feat /= avg_feat.norm(dim=-1, keepdim=True)
        encoded_colors.append(avg_feat)
    color_matrix = torch.cat(encoded_colors, dim=0)

# Test Suite with diverse images
test_samples = [
    ("https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=600&auto=format&fit=crop&q=80", "Red Ferrari", "Ferrari", "Crimson Red / Rosso Corsa"),
    ("https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=600&auto=format&fit=crop&q=80", "Black Porsche 911", "Porsche", "Gloss Black / Obsidian"),
    ("https://images.unsplash.com/photo-1555215695-3004980ad54e?w=600&auto=format&fit=crop&q=80", "Silver BMW", "BMW", "Metallic Silver / Titanium Grey"),
    ("https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?w=600&auto=format&fit=crop&q=80", "White Suzuki Swift", "Suzuki", "Pearl White / Arctic White"),
    ("https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=600&auto=format&fit=crop&q=80", "Blue Tesla Model 3", "Tesla", "Deep Metallic Blue / Royal Navy"),
    ("https://images.unsplash.com/photo-1617814076367-b759c7d7e738?w=600&auto=format&fit=crop&q=80", "Green Mercedes AMG GT", "Mercedes-Benz", "British Racing Green / Emerald"),
    ("https://images.unsplash.com/photo-1544829099-b9a0c07fad1a?w=600&auto=format&fit=crop&q=80", "Yellow Lamborghini", "Lamborghini", "Racing Yellow / Gold"),
    ("https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=600&auto=format&fit=crop&q=80", "Coffee Cup", "Not Detected", "Not Detected")
]

print("\n" + "=" * 90)
print(f"{'Input Target':<25} | {'Predicted Make & Model':<35} | {'Color':<30} | {'Status'}")
print("=" * 90)

for url, label, exp_make, exp_col in test_samples:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    img_data = urllib.request.urlopen(req, timeout=10).read()
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        img_feat = model.encode_image(tensor)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        sim = (100.0 * img_feat @ database_matrix.T).softmax(dim=-1)[0]
        top_idx = sim.argmax().item()
        top_score = sim[top_idx].item()
        res = DATABASE[top_idx]

        col_sim = (100.0 * img_feat @ color_matrix.T).softmax(dim=-1)[0]
        top_col_idx = col_sim.argmax().item()
        col_res = COLOR_LIST[top_col_idx] if res["make"] != "Not Detected" else {"name": "Not Detected", "hex": "#808080"}

        is_make_ok = (res['make'] == exp_make)
        status_str = "PASS" if is_make_ok else "FAIL"
        full_pred = f"{res['make']} {res['model']} ({top_score*100:.1f}%)"
        print(f"{label:<25} | {full_pred:<35} | {col_res['name']:<30} | [{status_str}]")

print("=" * 90)
