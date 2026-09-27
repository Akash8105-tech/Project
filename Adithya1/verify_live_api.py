import urllib.request
import json
import base64

tests = [
    ('web-demo/samples/ferrari.jpg',     'Ferrari',    'LaFerrari / 488 GTB'),
    ('web-demo/samples/bmw.jpg',         'BMW',        'M5 / 5 Series Sedan'),
    ('web-demo/samples/porsche.jpg',     'Porsche',    'Panamera Turbo / 911'),
    ('web-demo/samples/polo.jpg',        'Volkswagen', 'Polo / Golf'),
    ('web-demo/samples/tesla.jpg',       'Tesla',      'Model 3 Sedan'),
    ('web-demo/samples/lamborghini.jpg', 'Lamborghini','Huracan Performante')
]

print('=' * 105)
print(f"{'File':<34} | {'Predicted Vehicle':<38} | {'Color':<30} | {'Status'}")
print('=' * 105)

for fp, exp_make, exp_model in tests:
    with open(fp, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = json.dumps({'image': 'data:image/jpeg;base64,' + b64}).encode()
    req = urllib.request.Request('http://localhost:8080/api/analyze', data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read().decode())
        mk = data.get('make', 'N/A')
        md = data.get('model', 'N/A')
        col = data.get('colour') or data.get('color') or 'N/A'
        conf = data.get('confidence', 'N/A')
        is_ok = (mk == exp_make)
        status = 'PASS' if is_ok else 'FAIL'
        full_veh = f"{mk} {md}"
        print(f"{fp:<34} | {full_veh:<38} | {col:<30} | [{status}]")

print('=' * 105)
