import numpy as np
from PIL import Image

def get_color(pil_img):
    w, h = pil_img.size
    crop = pil_img.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75))).resize((100, 100))
    arr = np.array(crop) / 255.0
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c
    v = max_c
    s = np.zeros_like(v)
    m = v > 0
    s[m] = delta[m] / v[m]
    
    valid = (v >= 0.08) & (v <= 0.98)
    s_val = s[valid]
    v_val = v[valid]
    mean_s = float(np.mean(s_val))
    mean_v = float(np.mean(v_val))
    
    # Check hue
    h_deg = np.zeros_like(v)
    mask_r = (delta > 0) & (max_c == r)
    h_deg[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    mask_g = (delta > 0) & (max_c == g)
    h_deg[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    mask_b = (delta > 0) & (max_c == b)
    h_deg[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    
    # Low saturation or dark/light extremes
    if mean_s < 0.16 or (mean_v < 0.25 and mean_s < 0.30):
        if mean_v < 0.28:
            return 'Gloss Black / Obsidian', '#18181B'
        elif mean_v > 0.50:
            return 'Pure White / Pearl White', '#F8FAFC'
        else:
            return 'Metallic Silver / Titanium Grey', '#A0A5AA'
            
    sat_mask = valid & (s > 0.18)
    if not np.any(sat_mask):
        return 'Metallic Silver / Titanium Grey', '#A0A5AA'
        
    median_h = float(np.median(h_deg[sat_mask]))
    
    if median_h < 18 or median_h >= 340:
        return 'Crimson Red / Rosso Corsa', '#DC2626'
    elif 18 <= median_h < 45:
        return 'Papaya Orange / Sunset Amber', '#EA580C'
    elif 45 <= median_h < 75:
        return 'Racing Yellow / Gold', '#EAB308'
    elif 75 <= median_h < 170:
        return 'British Racing Green / Emerald', '#16A34A'
    elif 170 <= median_h < 265:
        return 'Deep Metallic Blue / Navy', '#2563EB'
    elif 265 <= median_h < 340:
        return 'Metallic Purple / Violet', '#7C3AED'
    else:
        return 'Crimson Red / Rosso Corsa', '#DC2626'

files = [
    ('web-demo/samples/ferrari.jpg',     'Crimson Red / Rosso Corsa'),
    ('web-demo/samples/bmw.jpg',         'Metallic Silver / Titanium Grey'),
    ('web-demo/samples/porsche.jpg',     'Gloss Black / Obsidian'),
    ('web-demo/samples/polo.jpg',        'Deep Metallic Blue / Navy'),
    ('web-demo/samples/tesla.jpg',       'Pure White / Pearl White'),
    ('web-demo/samples/lamborghini.jpg', 'Pure White / Pearl White')
]

for fp, exp in files:
    img = Image.open(fp).convert('RGB')
    name, hex_c = get_color(img)
    is_ok = (name == exp)
    status = "PASS" if is_ok else "FAIL"
    print(f"{fp:<32} -> {name:<32} ({hex_c}) [{status}]")
