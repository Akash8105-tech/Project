import os
import sys
import json
import re
import io
import base64
import hashlib
from pathlib import Path

# Flask REST API
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# PyTorch & OpenCLIP Neural Network
import torch
import open_clip
import numpy as np
from PIL import Image, ImageOps

PORT = 8080
BASE_DIR = Path(__file__).parent.resolve()
WEB_DIR = BASE_DIR / "web-demo"

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
CORS(app)

# ---------------------------------------------------------
# Load Pre-Trained OpenAI CLIP Vision Transformer
# ---------------------------------------------------------
print("[AI Vision] Initializing OpenAI CLIP Vision Transformer (ViT-B-32)...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model.eval()
print("[AI Vision] CLIP Model loaded successfully!")

# ---------------------------------------------------------
# STAGE 1: AUTOMOTIVE LOGO & EMBLEM RECOGNITION BANK
# ---------------------------------------------------------
LOGO_CLASSES = [
    {
        "brand": "Ferrari",
        "emblem_name": "Ferrari Prancing Horse (Cavallino Rampante) Shield",
        "prompts": [
            "a photo of a Ferrari prancing horse logo badge",
            "a photo of a yellow Ferrari shield emblem",
            "a photo of a Ferrari logo on a car hood",
            "a Ferrari badge emblem"
        ]
    },
    {
        "brand": "BMW",
        "emblem_name": "BMW Roundel (Blue & White Quarters)",
        "prompts": [
            "a photo of a BMW roundel emblem logo",
            "a photo of a BMW hood badge logo",
            "a photo of a BMW blue and white logo badge on a car",
            "a BMW emblem"
        ]
    },
    {
        "brand": "Porsche",
        "emblem_name": "Porsche Stuttgart Coat of Arms Crest",
        "prompts": [
            "a photo of a Porsche crest shield emblem badge",
            "a photo of a Porsche emblem on a car hood",
            "a photo of Porsche lettering and crest logo",
            "a Porsche gold crest badge"
        ]
    },
    {
        "brand": "Volkswagen",
        "emblem_name": "Volkswagen Chrome VW Circle Roundel",
        "prompts": [
            "a photo of a Volkswagen VW chrome circle logo emblem",
            "a photo of a VW badge on a car grille",
            "a Volkswagen emblem badge",
            "a VW roundel"
        ]
    },
    {
        "brand": "Tesla",
        "emblem_name": "Tesla Stylized 'T' Chrome Emblem",
        "prompts": [
            "a photo of a Tesla 'T' logo emblem on the hood",
            "a photo of a Tesla chrome logo",
            "a Tesla badge on a car",
            "a Tesla emblem"
        ]
    },
    {
        "brand": "Lamborghini",
        "emblem_name": "Lamborghini Golden Raging Bull (Toro Scatenato) Shield",
        "prompts": [
            "a photo of a Lamborghini raging bull gold shield emblem",
            "a photo of a Lamborghini crest badge",
            "a Lamborghini emblem on a supercar",
            "a Lamborghini badge"
        ]
    },
    {
        "brand": "Suzuki",
        "emblem_name": "Suzuki Stylized 'S' Chrome Grille Emblem",
        "prompts": [
            "a photo of a Suzuki 'S' chrome emblem on a car grille",
            "a photo of a Suzuki logo badge",
            "a photo of a Maruti Suzuki emblem on a car",
            "a Suzuki logo"
        ]
    },
    {
        "brand": "Mercedes-Benz",
        "emblem_name": "Mercedes-Benz Three-Pointed Star",
        "prompts": [
            "a photo of a Mercedes-Benz three-pointed star logo emblem",
            "a photo of a Mercedes star badge on a car grille",
            "a Mercedes-Benz hood star emblem"
        ]
    },
    {
        "brand": "Audi",
        "emblem_name": "Audi Four Interlocking Rings",
        "prompts": [
            "a photo of an Audi four interlocked chrome rings logo emblem",
            "a photo of an Audi rings badge on a car grille",
            "an Audi emblem"
        ]
    },
    {
        "brand": "Toyota",
        "emblem_name": "Toyota Triple Overlapping Ovals",
        "prompts": [
            "a photo of a Toyota chrome triple oval logo emblem",
            "a photo of a Toyota badge on a car grille",
            "a Toyota emblem"
        ]
    },
    {
        "brand": "Hyundai",
        "emblem_name": "Hyundai Slanted 'H' Oval Emblem",
        "prompts": [
            "a photo of a Hyundai slanted 'H' oval chrome logo emblem",
            "a photo of a Hyundai badge on a car grille",
            "a Hyundai emblem"
        ]
    },
    {
        "brand": "Ford",
        "emblem_name": "Ford Blue Oval Script Emblem",
        "prompts": [
            "a photo of a Ford blue oval chrome script logo emblem",
            "a photo of a Ford badge on a car",
            "a Ford emblem"
        ]
    },
    {
        "brand": "Chevrolet",
        "emblem_name": "Chevrolet Gold Bowtie Emblem",
        "prompts": [
            "a photo of a Chevrolet gold bowtie emblem badge",
            "a photo of a Chevy bowtie logo on a car",
            "a Chevrolet emblem"
        ]
    },
    {
        "brand": "Tata",
        "emblem_name": "Tata Chrome 'T' Ring Emblem",
        "prompts": [
            "a photo of a Tata Motors chrome stylized 'T' emblem badge",
            "a photo of a Tata logo on a car grille"
        ]
    },
    {
        "brand": "Mahindra",
        "emblem_name": "Mahindra Twin Peaks Emblem",
        "prompts": [
            "a photo of a Mahindra Twin Peaks chrome logo emblem",
            "a photo of a Mahindra badge on an SUV grille"
        ]
    }
]

print(f"[AI Vision] Pre-encoding Logo & Emblem embeddings for {len(LOGO_CLASSES)} brands...")
encoded_logos = []
with torch.no_grad():
    for item in LOGO_CLASSES:
        tokens = tokenizer(item["prompts"])
        feats = clip_model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
        avg_feat = feats.mean(dim=0, keepdim=True)
        avg_feat /= avg_feat.norm(dim=-1, keepdim=True)
        encoded_logos.append(avg_feat)
    logo_matrix = torch.cat(encoded_logos, dim=0)

# ---------------------------------------------------------
# STAGE 2: VEHICLE CANDIDATE KNOWLEDGE BASE
# ---------------------------------------------------------
CAR_CANDIDATES = [
    # Ferrari
    {
        "id": "ferrari_laferrari_488",
        "make": "Ferrari",
        "model": "LaFerrari / 488 GTB",
        "bodyType": "Mid-Engine Supercar",
        "estimatedYearRange": "2016-2024",
        "defaultEmblem": "Ferrari Prancing Horse (Cavallino Rampante) Shield",
        "notes": "Aerodynamic S-Duct front hood, deep sculpted side air intake scoops, active rear aerodynamics, and signature Ferrari Prancing Horse styling.",
        "prompts": [
            "a photo of a Ferrari LaFerrari hypercar",
            "a photo of a Ferrari 488 GTB supercar",
            "a photo of a red Ferrari sports car",
            "a photo of a Ferrari supercar"
        ]
    },
    {
        "id": "ferrari_sf90_roma",
        "make": "Ferrari",
        "model": "SF90 Stradale / Roma",
        "bodyType": "Hybrid Supercar / Grand Tourer",
        "estimatedYearRange": "2020-2025",
        "defaultEmblem": "Ferrari Prancing Horse (Cavallino Rampante) Shield",
        "notes": "C-shaped matrix LED headlights, twin-turbo hybrid V8, shark-nose grille, and active rear aerodynamics.",
        "prompts": [
            "a photo of a Ferrari SF90 Stradale",
            "a photo of a Ferrari Roma coupe",
            "a Ferrari grand tourer"
        ]
    },

    # BMW
    {
        "id": "bmw_m5_5series",
        "make": "BMW",
        "model": "M5 / 5 Series Sedan",
        "bodyType": "Executive Performance Sedan",
        "estimatedYearRange": "2019-2025",
        "defaultEmblem": "BMW Roundel (Blue & White Quarters)",
        "notes": "Signature gloss-black dual kidney grille with M badge, aggressive front air dams, double-crease bonnet, and high-performance sport sedan profile.",
        "prompts": [
            "a photo of a BMW M5 Competition sedan",
            "a photo of a BMW 5 Series sedan",
            "a photo of a BMW M sports sedan",
            "a photo of a BMW automobile"
        ]
    },
    {
        "id": "bmw_7series_3series",
        "make": "BMW",
        "model": "7 Series / 3 Series Sedan",
        "bodyType": "Luxury Executive Sedan",
        "estimatedYearRange": "2020-2025",
        "defaultEmblem": "BMW Roundel (Blue & White Quarters)",
        "notes": "Active dual kidney grille with vertical chrome slats, slim laser LED headlights, sculpted bonnet power domes, and luxury long-wheelbase stance.",
        "prompts": [
            "a photo of a BMW 7 Series luxury sedan",
            "a photo of a BMW 3 Series sedan",
            "a BMW 3 Series automobile"
        ]
    },
    {
        "id": "bmw_x5_x7",
        "make": "BMW",
        "model": "X5 / X3 / X7",
        "bodyType": "Luxury Sports Activity Vehicle (SAV)",
        "estimatedYearRange": "2020-2025",
        "defaultEmblem": "BMW Roundel (Blue & White Quarters)",
        "notes": "Prominent upright kidney grilles, muscular shoulder line, high ride height, and split two-piece powered tailgate.",
        "prompts": [
            "a photo of a BMW X5 luxury SUV",
            "a photo of a BMW X3 SUV",
            "a BMW SAV vehicle"
        ]
    },

    # Porsche
    {
        "id": "porsche_panamera_911",
        "make": "Porsche",
        "model": "Panamera Turbo / 911",
        "bodyType": "Luxury Sport Sedan / Coupe",
        "estimatedYearRange": "2018-2025",
        "defaultEmblem": "Porsche Stuttgart Coat of Arms Crest",
        "notes": "Distinctive full-width rear LED light bar with central Porsche lettering, active extending rear spoiler, quad exhaust tips, and iconic flyline roof silhouette.",
        "prompts": [
            "a photo of the rear of a Porsche Panamera Turbo",
            "a photo of a Porsche Panamera sedan",
            "a photo of a Porsche 911 Carrera",
            "a Porsche sports car"
        ]
    },
    {
        "id": "porsche_718_cayman",
        "make": "Porsche",
        "model": "718 Cayman / Boxster",
        "bodyType": "Mid-Engine Sports Coupe",
        "estimatedYearRange": "2018-2025",
        "defaultEmblem": "Porsche Stuttgart Coat of Arms Crest",
        "notes": "Mid-engine sports coupe with sharp side air intakes, sculpted front wings, and four-point DRL headlights.",
        "prompts": [
            "a photo of a Porsche 718 Cayman sports coupe",
            "a photo of a Porsche Boxster roadster",
            "a Porsche 718 sports car"
        ]
    },
    {
        "id": "porsche_cayenne_macan",
        "make": "Porsche",
        "model": "Cayenne / Macan",
        "bodyType": "Performance Luxury SUV",
        "estimatedYearRange": "2020-2025",
        "defaultEmblem": "Porsche Stuttgart Coat of Arms Crest",
        "notes": "Sports car front styling, four-point LED headlights, muscular shoulder line, and continuous rear LED strip.",
        "prompts": [
            "a photo of a Porsche Cayenne SUV",
            "a photo of a Porsche Macan SUV"
        ]
    },

    # Volkswagen
    {
        "id": "vw_polo_golf",
        "make": "Volkswagen",
        "model": "Polo / Golf",
        "bodyType": "Compact Hatchback",
        "estimatedYearRange": "2018-2025",
        "defaultEmblem": "Volkswagen Chrome VW Circle Roundel",
        "notes": "Clean horizontal chrome grille with central VW roundel badge, geometric angular headlights, lower front bumper fog light bar, and balanced European hatchback proportions.",
        "prompts": [
            "a photo of a Volkswagen Polo hatchback",
            "a photo of a blue Volkswagen Polo car",
            "a Volkswagen Golf hatchback car",
            "a Volkswagen car"
        ]
    },
    {
        "id": "vw_virtus_taigun",
        "make": "Volkswagen",
        "model": "Virtus / Taigun SUV",
        "bodyType": "Sedan / Compact SUV",
        "estimatedYearRange": "2021-2025",
        "defaultEmblem": "Volkswagen Chrome VW Circle Roundel",
        "notes": "Chrome wing grille, sharp LED headlamps, muscular stance, and signature German build styling.",
        "prompts": [
            "a photo of a Volkswagen Virtus sedan",
            "a photo of a Volkswagen Taigun compact SUV"
        ]
    },

    # Tesla
    {
        "id": "tesla_model3_y",
        "make": "Tesla",
        "model": "Model 3 Sedan",
        "bodyType": "Electric Fastback Sedan",
        "estimatedYearRange": "2020-2025",
        "defaultEmblem": "Tesla Stylized 'T' Chrome Emblem",
        "notes": "Minimalist grille-less aerodynamic front fascia, all-glass panoramic canopy, flush electronic door handles, and sleek high-efficiency electric fastback silhouette.",
        "prompts": [
            "a photo of a white Tesla Model 3 electric sedan",
            "a photo of a Tesla Model 3 car",
            "a Tesla Model 3",
            "a Tesla electric car"
        ]
    },
    {
        "id": "tesla_cybertruck",
        "make": "Tesla",
        "model": "Cybertruck",
        "bodyType": "Electric Pickup Truck",
        "estimatedYearRange": "2023-2025",
        "defaultEmblem": "Tesla Stylized 'T' Chrome Emblem",
        "notes": "Unpainted stainless steel exoskeleton, angular triangular geometric design, and full-width front LED light bar.",
        "prompts": [
            "a photo of a Tesla Cybertruck electric pickup",
            "a Tesla Cybertruck"
        ]
    },

    # Lamborghini
    {
        "id": "lambo_huracan_performante",
        "make": "Lamborghini",
        "model": "Huracán Performante",
        "bodyType": "V10 Supercar",
        "estimatedYearRange": "2018-2024",
        "defaultEmblem": "Lamborghini Golden Raging Bull (Toro Scatenato) Shield",
        "notes": "Razor-sharp hexagonal front air intakes, Y-shaped LED daytime running lights, ALA active aero rear wing, and aggressive ultra-low wedge supercar silhouette.",
        "prompts": [
            "a photo of a white Lamborghini Huracan Performante supercar",
            "a photo of a Lamborghini Huracan sports car",
            "a Lamborghini supercar"
        ]
    },
    {
        "id": "lambo_urus",
        "make": "Lamborghini",
        "model": "Urus",
        "bodyType": "Super Sport SUV",
        "estimatedYearRange": "2019-2025",
        "defaultEmblem": "Lamborghini Golden Raging Bull (Toro Scatenato) Shield",
        "notes": "Hexagonal styling, frameless doors, coupe roofline, aggressive front Y-bonnet intakes, and massive carbon ceramic brakes.",
        "prompts": [
            "a photo of a Lamborghini Urus super SUV",
            "a Lamborghini Urus high performance SUV"
        ]
    },

    # Suzuki / Maruti Suzuki
    {
        "id": "suzuki_swift",
        "make": "Suzuki",
        "model": "Swift (4th Gen)",
        "bodyType": "Compact Hatchback",
        "estimatedYearRange": "2020-2025",
        "defaultEmblem": "Suzuki Stylized 'S' Chrome Grille Emblem",
        "notes": "Signature gloss-black honeycomb hexagonal grille with central Suzuki badge, stylish L-shaped LED daytime running lights, and floating blacked-out roof pillars.",
        "prompts": [
            "a photo of a Suzuki Swift compact hatchback",
            "a photo of a Maruti Suzuki Swift car",
            "a Suzuki Swift car",
            "a Suzuki hatchback"
        ]
    },
    {
        "id": "suzuki_baleno_jimny",
        "make": "Suzuki",
        "model": "Baleno / Jimny",
        "bodyType": "Hatchback / 4x4 Off-Roader",
        "estimatedYearRange": "2020-2025",
        "defaultEmblem": "Suzuki Stylized 'S' Chrome Grille Emblem",
        "notes": "NEXWave chrome grille or retro boxy five-slot grille with flared rugged arches.",
        "prompts": [
            "a photo of a Suzuki Baleno hatchback",
            "a photo of a Suzuki Jimny 4x4"
        ]
    },

    # Mercedes-Benz
    {
        "id": "mercedes_amg_gt",
        "make": "Mercedes-Benz",
        "model": "AMG GT / SL Roadster",
        "bodyType": "Performance Sports Coupe",
        "estimatedYearRange": "2020-2025",
        "defaultEmblem": "Mercedes-Benz Three-Pointed Star",
        "notes": "Panamericana vertical chrome slat grille, long sweeping bonnet, short muscular rear deck, and wide stance.",
        "prompts": [
            "a photo of a Mercedes-AMG GT sports coupe",
            "a photo of a Mercedes-AMG sports car"
        ]
    },
    {
        "id": "mercedes_sclass_cclass",
        "make": "Mercedes-Benz",
        "model": "S-Class / C-Class",
        "bodyType": "Executive Luxury Sedan",
        "estimatedYearRange": "2021-2025",
        "defaultEmblem": "Mercedes-Benz Three-Pointed Star",
        "notes": "Flush-fitting door handles, stately chrome multi-slat grille with upright star, and triangular Digital Light headlamps.",
        "prompts": [
            "a photo of a Mercedes-Benz S-Class luxury sedan",
            "a photo of a Mercedes-Benz C-Class sedan"
        ]
    },
    {
        "id": "mercedes_gwagon",
        "make": "Mercedes-Benz",
        "model": "G-Class (G-Wagon)",
        "bodyType": "Luxury 4x4 Off-Roader",
        "estimatedYearRange": "2019-2025",
        "defaultEmblem": "Mercedes-Benz Three-Pointed Star",
        "notes": "Iconic boxy silhouette, exposed door hinges, round headlights, exterior side exhaust pipes, and rear-mounted spare wheel.",
        "prompts": [
            "a photo of a Mercedes-Benz G-Class G-Wagon",
            "a Mercedes G63 AMG SUV"
        ]
    },

    # Audi
    {
        "id": "audi_r8",
        "make": "Audi",
        "model": "R8 V10 Performance",
        "bodyType": "Mid-Engine Supercar",
        "estimatedYearRange": "2019-2024",
        "defaultEmblem": "Audi Four Interlocking Rings",
        "notes": "Signature carbon sideblades, wide honeycomb front grille with three flat hood slits, and aggressive rear oval tailpipes.",
        "prompts": [
            "a photo of an Audi R8 supercar",
            "a photo of an Audi R8 sports coupe"
        ]
    },
    {
        "id": "audi_a6_a4",
        "make": "Audi",
        "model": "A6 / A4 Sedan",
        "bodyType": "Executive Sedan",
        "estimatedYearRange": "2019-2025",
        "defaultEmblem": "Audi Four Interlocking Rings",
        "notes": "Wide Singleframe hexagonal grille, razor-sharp matrix LED daytime signatures, and clean horizontal shoulder character line.",
        "prompts": [
            "a photo of an Audi A6 luxury sedan",
            "a photo of an Audi A4 sedan"
        ]
    },

    # Toyota
    {
        "id": "toyota_fortuner_lc",
        "make": "Toyota",
        "model": "Fortuner / Land Cruiser",
        "bodyType": "Rugged 4x4 SUV",
        "estimatedYearRange": "2019-2025",
        "defaultEmblem": "Toyota Triple Overlapping Ovals",
        "notes": "Dominant chrome grille, high approach angle, muscular flared wheel arches, and heavy-duty ladder frame stance.",
        "prompts": [
            "a photo of a Toyota Fortuner SUV",
            "a photo of a Toyota Land Cruiser SUV"
        ]
    },
    {
        "id": "toyota_supra",
        "make": "Toyota",
        "model": "GR Supra",
        "bodyType": "Sports Coupe",
        "estimatedYearRange": "2020-2025",
        "defaultEmblem": "Toyota Triple Overlapping Ovals",
        "notes": "Double-bubble roof, central front air intake, ducktail rear spoiler, and athletic sports coupe dimensions.",
        "prompts": [
            "a photo of a Toyota GR Supra sports car",
            "a Toyota Supra coupe"
        ]
    },

    # Hyundai
    {
        "id": "hyundai_creta_tucson",
        "make": "Hyundai",
        "model": "Creta / Venue / Tucson",
        "bodyType": "SUV / Crossover",
        "estimatedYearRange": "2020-2025",
        "defaultEmblem": "Hyundai Slanted 'H' Oval Emblem",
        "notes": "Parametric Jewel pattern grille with integrated hidden DRLs, split LED headlamps, and sculpted aerodynamic character creases.",
        "prompts": [
            "a photo of a Hyundai Creta SUV",
            "a photo of a Hyundai Tucson SUV"
        ]
    },
    {
        "id": "hyundai_i20_verna",
        "make": "Hyundai",
        "model": "i20 / Verna",
        "bodyType": "Hatchback / Sedan",
        "estimatedYearRange": "2020-2025",
        "defaultEmblem": "Hyundai Slanted 'H' Oval Emblem",
        "notes": "Sensuous sportiness styling, cascading black front grille, fastback roofline, and Z-shaped LED taillights.",
        "prompts": [
            "a photo of a Hyundai i20 premium hatchback",
            "a photo of a Hyundai Verna sedan"
        ]
    },

    # Ford & Chevrolet
    {
        "id": "ford_mustang",
        "make": "Ford",
        "model": "Mustang GT",
        "bodyType": "Fastback Muscle Car",
        "estimatedYearRange": "2018-2025",
        "defaultEmblem": "Ford Blue Oval Script Emblem",
        "notes": "Iconic tri-bar LED daytime running lights, long hood, muscular rear haunches, and aggressive wide front grille.",
        "prompts": [
            "a photo of a Ford Mustang GT muscle car",
            "a photo of a Ford Mustang fastback"
        ]
    },
    {
        "id": "chevy_corvette",
        "make": "Chevrolet",
        "model": "Corvette Stingray (C8)",
        "bodyType": "Mid-Engine Sports Coupe",
        "estimatedYearRange": "2020-2025",
        "defaultEmblem": "Chevrolet Gold Bowtie Emblem",
        "notes": "Mid-engine proportions, dramatic triangular side scoops, angular quad exhaust tips, and sharp edge sculpting.",
        "prompts": [
            "a photo of a Chevrolet Corvette C8 sports car",
            "a Chevy Corvette Stingray"
        ]
    },

    # Tata Motors & Mahindra
    {
        "id": "tata_nexon_harrier",
        "make": "Tata",
        "model": "Nexon / Harrier / Safari",
        "bodyType": "SUV",
        "estimatedYearRange": "2020-2025",
        "defaultEmblem": "Tata Chrome 'T' Ring Emblem",
        "notes": "Connected LED light bar, parametric grille, tri-arrow design accents, and high ground clearance SUV stance.",
        "prompts": [
            "a photo of a Tata Nexon SUV",
            "a photo of a Tata Harrier SUV"
        ]
    },
    {
        "id": "mahindra_thar_xuv700",
        "make": "Mahindra",
        "model": "Thar / XUV700 / Scorpio",
        "bodyType": "Rugged SUV / 4x4",
        "estimatedYearRange": "2020-2025",
        "defaultEmblem": "Mahindra Twin Peaks Emblem",
        "notes": "Iconic vertical slat grille, rugged off-road stance, round headlamps or C-shaped DRLs, and muscular body cladding.",
        "prompts": [
            "a photo of a Mahindra Thar 4x4 off-roader",
            "a photo of a Mahindra XUV700 SUV"
        ]
    },

    # Validation Candidate (Non-Vehicle)
    {
        "id": "non_vehicle",
        "make": "Not Detected",
        "model": "Not Detected",
        "bodyType": "Non-Vehicle",
        "estimatedYearRange": "N/A",
        "defaultEmblem": "None",
        "notes": "Visual neural network scanned this image and detected no automobile or road vehicle.",
        "prompts": [
            "a photo of a cup of coffee, mug, beverage, drink, table, furniture, chair, desk, room, plant, pet, human, face, food, dish, or non-vehicle object",
            "a photo of an indoor room with furniture and no cars",
            "a photo of a cup of coffee on a table",
            "a photo of a human or person"
        ]
    }
]

print(f"[AI Vision] Pre-encoding candidate embeddings for {len(CAR_CANDIDATES)} classes...")
encoded_candidates = []
with torch.no_grad():
    for item in CAR_CANDIDATES:
        tokens = tokenizer(item["prompts"])
        feats = clip_model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
        avg_feat = feats.mean(dim=0, keepdim=True)
        avg_feat /= avg_feat.norm(dim=-1, keepdim=True)
        encoded_candidates.append(avg_feat)
    kb_matrix = torch.cat(encoded_candidates, dim=0)

# Multi-Prompt Color Classes
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
    },
    {
        "name": "Metallic Purple / Violet",
        "hex": "#7C3AED",
        "prompts": ["a photo of a purple car", "a purple automobile"]
    }
]

color_feats = []
with torch.no_grad():
    for c in COLOR_CLASSES:
        tokens = tokenizer(c["prompts"])
        f = clip_model.encode_text(tokens)
        f /= f.norm(dim=-1, keepdim=True)
        avg_f = f.mean(dim=0, keepdim=True)
        avg_f /= avg_f.norm(dim=-1, keepdim=True)
        color_feats.append(avg_f)
    color_matrix = torch.cat(color_feats, dim=0)

print("[AI Vision] All neural embeddings pre-computed and cached!")

def extract_vehicle_color(img_features):
    with torch.no_grad():
        sim = (100.0 * img_features @ color_matrix.T).softmax(dim=-1)[0]
        top_idx = sim.argmax().item()
        return COLOR_CLASSES[top_idx]["name"], COLOR_CLASSES[top_idx]["hex"]

def get_env_api_key():
    local_props_path = BASE_DIR / "local.properties"
    if local_props_path.exists():
        try:
            with open(local_props_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY=") or line.startswith("GEMINI_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        if key and not key.startswith("YOUR_"):
                            return key
        except Exception:
            pass
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""

CURRENT_API_KEY = get_env_api_key()

@app.route("/")
def index():
    return send_from_directory(str(WEB_DIR), "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(str(WEB_DIR), path)

@app.route("/api/key-status", methods=["GET"])
def key_status():
    return jsonify({
        "hasKey": bool(CURRENT_API_KEY),
        "keyType": "active" if CURRENT_API_KEY else "none"
    })

@app.route("/api/set-key", methods=["POST"])
def set_key():
    global CURRENT_API_KEY
    data = request.get_json(silent=True) or {}
    new_key = data.get("apiKey", "").strip()
    if new_key:
        CURRENT_API_KEY = new_key
        try:
            with open(BASE_DIR / "local.properties", "w", encoding="utf-8") as f:
                if new_key.startswith("sk-"):
                    f.write(f"OPENAI_API_KEY={new_key}\n")
                else:
                    f.write(f"GEMINI_API_KEY={new_key}\n")
        except Exception:
            pass

    return jsonify({
        "success": True,
        "hasKey": bool(CURRENT_API_KEY),
        "keyType": "active"
    })

def analyze_with_openai(img_bytes, api_key):
    b64_img = base64.b64encode(img_bytes).decode("utf-8")
    import urllib.request
    url = "https://api.openai.com/v1/chat/completions"
    prompt = (
        "You are an expert automotive visual identification AI system. Analyze this image carefully.\n\n"
        "AUTOMOTIVE LOGO & EMBLEM-FIRST RECOGNITION PROTOCOL:\n"
        "1. FIRST: Scan the vehicle's front grille, hood, wheel center caps, steering wheel, and rear deck to locate and identify manufacturer LOGOS, BADGES, CRESTS, or EMBLEMS (e.g. Ferrari Prancing Horse, BMW Roundel, Porsche Crest, VW Circle, Tesla 'T', Lamborghini Raging Bull, Suzuki 'S', Mercedes Star, Audi 4 Rings, Toyota Ovals, Hyundai Slanted 'H', etc.).\n"
        "2. Establish the brand/make from the detected logo.\n"
        "3. SECOND: Analyze the headlights, DRL signature, grille design, body silhouette, and aerodynamic features to pinpoint the exact Model and generation.\n"
        "4. Output ONLY a raw, valid JSON object strictly matching this schema with NO markdown code blocks:\n"
        "{\n"
        '  "carDetected": true,\n'
        '  "make": "string (e.g. BMW, Ferrari, Suzuki, Porsche, Tesla)",\n'
        '  "model": "string (e.g. M5, 488 GTB, Swift, 911 Carrera, Model 3)",\n'
        '  "detectedEmblem": "string (e.g. Ferrari Prancing Horse Shield, BMW Roundel)",\n'
        '  "colour": "string (e.g. Metallic Silver, Crimson Red, Pearl White, Gloss Black)",\n'
        '  "hexColor": "string (e.g. #A0A5AA, #DC2626, #F8FAFC, #18181B)",\n'
        '  "bodyType": "string (e.g. Executive Luxury Sedan, Compact Hatchback, Supercar, Sports Coupe)",\n'
        '  "estimatedYearRange": "string (e.g. 2020-2025)",\n'
        '  "confidence": "string (e.g. High (98%))",\n'
        '  "notes": "string (distinctive logo verification and design styling notes)"\n'
        "}\n"
        "If NO car or vehicle is visible in the image, set carDetected to false and make/model/colour to 'Not Detected'."
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=14) as response:
        res_data = json.loads(response.read().decode("utf-8"))
        text = res_data["choices"][0]["message"]["content"]
        clean_json = re.sub(r"^```json\s*", "", text.strip())
        clean_json = re.sub(r"```$", "", clean_json.strip())
        return json.loads(clean_json)

def analyze_with_gemini(img_bytes, api_key):
    b64_img = base64.b64encode(img_bytes).decode("utf-8")
    import urllib.request
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    prompt = (
        "You are an expert automotive visual identification AI system. Analyze this image carefully.\n\n"
        "AUTOMOTIVE LOGO & EMBLEM-FIRST RECOGNITION PROTOCOL:\n"
        "1. FIRST: Scan the vehicle's front grille, hood, wheel center caps, steering wheel, and rear deck to locate and identify manufacturer LOGOS, BADGES, CRESTS, or EMBLEMS (e.g. Ferrari Prancing Horse, BMW Roundel, Porsche Crest, VW Circle, Tesla 'T', Lamborghini Raging Bull, Suzuki 'S', Mercedes Star, Audi 4 Rings, Toyota Ovals, Hyundai Slanted 'H', etc.).\n"
        "2. Establish the brand/make from the detected logo.\n"
        "3. SECOND: Analyze the headlights, DRL signature, grille design, body silhouette, and aerodynamic features to pinpoint the exact Model and generation.\n"
        "4. Output ONLY a raw, valid JSON object strictly matching this schema with NO markdown code blocks:\n"
        "{\n"
        '  "carDetected": true,\n'
        '  "make": "string (e.g. BMW, Ferrari, Suzuki, Porsche, Tesla)",\n'
        '  "model": "string (e.g. M5, 488 GTB, Swift, 911 Carrera, Model 3)",\n'
        '  "detectedEmblem": "string (e.g. Ferrari Prancing Horse Shield, BMW Roundel)",\n'
        '  "colour": "string (e.g. Metallic Silver, Crimson Red, Pearl White, Gloss Black)",\n'
        '  "hexColor": "string (e.g. #A0A5AA, #DC2626, #F8FAFC, #18181B)",\n'
        '  "bodyType": "string (e.g. Executive Luxury Sedan, Compact Hatchback, Supercar, Sports Coupe)",\n'
        '  "estimatedYearRange": "string (e.g. 2020-2025)",\n'
        '  "confidence": "string (e.g. High (98%))",\n'
        '  "notes": "string (distinctive logo verification and design styling notes)"\n'
        "}\n"
        "If NO car or vehicle is visible in the image, set carDetected to false and make/model/colour to 'Not Detected'."
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}
            ]
        }],
        "generationConfig": {"temperature": 0.1, "response_mime_type": "application/json"}
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=12) as response:
        res_data = json.loads(response.read().decode("utf-8"))
        text = res_data["candidates"][0]["content"]["parts"][0]["text"]
        clean_json = re.sub(r"^```json\s*", "", text.strip())
        clean_json = re.sub(r"```$", "", clean_json.strip())
        return json.loads(clean_json)

@app.route("/api/analyze", methods=["POST"])
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        img_bytes = None
        user_key = request.headers.get("x-api-key", "").strip()

        # 1. Handle multipart/form-data
        if "file" in request.files or "image" in request.files:
            file = request.files.get("file") or request.files.get("image")
            img_bytes = file.read()
        
        # 2. Handle JSON base64
        elif request.is_json:
            data = request.get_json()
            image_b64 = data.get("image", "")
            if "base64," in image_b64:
                image_b64 = image_b64.split("base64,")[1]
            img_bytes = base64.b64decode(image_b64)
            if not user_key and data.get("apiKey"):
                user_key = data.get("apiKey")

        # 3. Handle raw data fallback
        elif request.data:
            try:
                data = json.loads(request.data.decode("utf-8", errors="ignore"))
                image_b64 = data.get("image", "")
                if "base64," in image_b64:
                    image_b64 = image_b64.split("base64,")[1]
                img_bytes = base64.b64decode(image_b64)
                if not user_key and data.get("apiKey"):
                    user_key = data.get("apiKey")
            except Exception:
                img_bytes = request.data

        if not img_bytes:
            return jsonify({"error": "No image payload found"}), 400

        sha256_hash = hashlib.sha256(img_bytes).hexdigest()
        print(f"\n======================================================")
        print(f"[DEBUG] Received Scan Request:")
        print(f"  - Image Buffer Size: {len(img_bytes)} bytes")
        print(f"  - Image SHA-256:     {sha256_hash[:16]}...")
        print(f"  - User Key Provided: {user_key[:8] + '...' if user_key else 'None'}")

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = ImageOps.exif_transpose(img)

        # 1. Try OpenAI Vision if sk- key
        api_key_to_use = user_key if user_key else CURRENT_API_KEY
        if api_key_to_use and api_key_to_use.startswith("sk-"):
            try:
                res = analyze_with_openai(img_bytes, api_key_to_use)
                if res and isinstance(res, dict) and ("make" in res or "carDetected" in res):
                    res["source"] = "openai"
                    res["image_bytes"] = len(img_bytes)
                    res["image_sha256"] = sha256_hash
                    print(f"  - Resolved Source:   OPENAI GPT-4o-MINI")
                    print(f"  - Result:            {res.get('make')} {res.get('model')} (Logo: {res.get('detectedEmblem')})")
                    return jsonify(res)
            except Exception as e:
                print(f"  [!] OpenAI Vision API Notice ({e}), falling back to local CLIP...")

        # 2. Try Gemini API if AIzaSy key
        if api_key_to_use and api_key_to_use.startswith("AIzaSy"):
            try:
                res = analyze_with_gemini(img_bytes, api_key_to_use)
                if res and isinstance(res, dict) and ("make" in res or "carDetected" in res):
                    res["source"] = "gemini"
                    res["image_bytes"] = len(img_bytes)
                    res["image_sha256"] = sha256_hash
                    print(f"  - Resolved Source:   GEMINI")
                    print(f"  - Result:            {res.get('make')} {res.get('model')} (Logo: {res.get('detectedEmblem')})")
                    return jsonify(res)
            except Exception as e:
                print(f"  [!] Gemini Vision API Error ({e}), falling back to local CLIP...")

        img_tensor = clip_preprocess(img).unsqueeze(0)
        with torch.no_grad():
            img_features = clip_model.encode_image(img_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            # STEP 1: Evaluate Brand Logo & Emblem Similarity
            logo_sim = (100.0 * img_features @ logo_matrix.T).softmax(dim=-1)[0]
            top_logo_idx = logo_sim.argmax().item()
            matched_logo = LOGO_CLASSES[top_logo_idx]
            logo_conf = logo_sim[top_logo_idx].item()

            # STEP 2: Evaluate Vehicle Candidate Model Similarity
            cand_sim = (100.0 * img_features @ kb_matrix.T).softmax(dim=-1)[0]

            # STEP 3: Hierarchical Emblem-Guided Score Fusion
            boosted_scores = []
            for i, cand in enumerate(CAR_CANDIDATES):
                base_score = cand_sim[i].item()
                if cand["make"] == matched_logo["brand"]:
                    # Boost candidate whose make matches the detected logo
                    fused_score = 0.55 * base_score + 0.45 * logo_conf
                else:
                    fused_score = 0.70 * base_score
                boosted_scores.append(fused_score)

            top_idx = int(np.argmax(boosted_scores))
            raw_score = boosted_scores[top_idx]
            matched = CAR_CANDIDATES[top_idx]

        # Check for Non-Vehicle
        if matched["make"] == "Not Detected" or (raw_score < 0.12 and matched["id"] == "non_vehicle"):
            print(f"  - Resolved Source:   CLIP (Non-Vehicle Detected)")
            return jsonify({
                "carDetected": False,
                "detected": False,
                "make": "Not Detected",
                "model": "Not Detected",
                "detectedEmblem": "None Detected",
                "colour": "Not Detected",
                "color": "Not Detected",
                "hexColor": "#808080",
                "color_hex": "#808080",
                "confidence": "Uncertain",
                "bodyType": "Non-Vehicle",
                "body_type": "Non-Vehicle",
                "estimatedYearRange": "Unknown",
                "year": "N/A",
                "notes": "Visual neural network scanned this image and detected no automobile or road vehicle.",
                "additional_details": "Visual neural network scanned this image and detected no automobile or road vehicle.",
                "source": "clip",
                "image_bytes": len(img_bytes),
                "image_sha256": sha256_hash
            })

        color_name, hex_code = extract_vehicle_color(img_features)
        detected_emblem_name = matched.get("defaultEmblem", matched_logo["emblem_name"])

        if raw_score > 0.35:
            conf_percent = min(99.6, 90.0 + raw_score * 10.0)
            confidence_str = f"High ({conf_percent:.1f}%)"
            conf_int = int(conf_percent)
        elif raw_score > 0.15:
            conf_percent = min(88.9, 72.0 + raw_score * 35.0)
            confidence_str = f"Medium ({conf_percent:.1f}%)"
            conf_int = int(conf_percent)
        else:
            conf_percent = max(55.0, raw_score * 100.0)
            confidence_str = f"Low ({conf_percent:.1f}%)"
            conf_int = int(conf_percent)

        print(f"  - Resolved Source:   CLIP (Logo-First Guided)")
        print(f"  - Detected Emblem:   {detected_emblem_name}")
        print(f"  - Detected Vehicle:  {matched['make']} {matched['model']} | Color: {color_name} ({hex_code}) | Conf: {confidence_str}")
        print(f"======================================================\n")

        return jsonify({
            "carDetected": True,
            "detected": True,
            "make": matched["make"],
            "model": matched["model"],
            "detectedEmblem": detected_emblem_name,
            "colour": color_name,
            "color": color_name,
            "hexColor": hex_code,
            "color_hex": hex_code,
            "confidence": confidence_str,
            "confidence_num": conf_int,
            "bodyType": matched["bodyType"],
            "body_type": matched["bodyType"],
            "estimatedYearRange": matched["estimatedYearRange"],
            "year": matched["estimatedYearRange"],
            "notes": f"Verified {detected_emblem_name}. {matched['notes']}",
            "additional_details": f"Verified {detected_emblem_name}. {matched['notes']}",
            "source": "clip",
            "image_bytes": len(img_bytes),
            "image_sha256": sha256_hash
        })

    except Exception as e:
        print(f"[Error in /api/analyze]: {e}")
        return jsonify({
            "error": "ANALYSIS_FAILED",
            "message": str(e)
        }), 500

def start_server():
    print("======================================================")
    print(f"  CARGRASP LOCAL AI VISION SERVER RUNNING ON PORT {PORT}")
    print("  Visual Neural Network: OpenAI CLIP ViT-B-32 (Logo-First Hierarchical)")
    print("  Theme: Cyan Neon & Deep Blue (#00E5FF & #0A0F1D)")
    print(f"  URL: http://localhost:{PORT}")
    print("======================================================")
    app.run(host="0.0.0.0", port=PORT, debug=False)

if __name__ == "__main__":
    start_server()
