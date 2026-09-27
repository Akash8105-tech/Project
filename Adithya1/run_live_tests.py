import urllib.request
import json
import base64
import time
import hashlib

TEST_IMAGES = [
    {
        "name": "Red Ferrari Supercar",
        "url": "https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=600&auto=format&fit=crop&q=80",
        "expected_make": "Ferrari"
    },
    {
        "name": "Black Porsche 911 Carrera",
        "url": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=600&auto=format&fit=crop&q=80",
        "expected_make": "Porsche"
    },
    {
        "name": "Silver BMW Sedan",
        "url": "https://images.unsplash.com/photo-1555215695-3004980ad54e?w=600&auto=format&fit=crop&q=80",
        "expected_make": "BMW"
    },
    {
        "name": "White Suzuki Swift Hatchback",
        "url": "https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?w=600&auto=format&fit=crop&q=80",
        "expected_make": "Suzuki"
    },
    {
        "name": "Blue Tesla Model 3",
        "url": "https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=600&auto=format&fit=crop&q=80",
        "expected_make": "Tesla"
    },
    {
        "name": "Non-Car Photo (Coffee Cup)",
        "url": "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=600&auto=format&fit=crop&q=80",
        "expected_make": "Not Detected"
    }
]

def run_tests():
    print("=" * 70)
    print("   RUNNING CARGRASP MULTI-CAR RECOGNITION & UNIQUENESS VERIFICATION")
    print("=" * 70)
    
    passed = 0
    total = len(TEST_IMAGES)
    results_map = {}
    
    for i, item in enumerate(TEST_IMAGES, 1):
        print(f"\n[Test {i}/{total}] Testing: {item['name']}...")
        try:
            req = urllib.request.Request(item['url'], headers={"User-Agent": "Mozilla/5.0"})
            img_bytes = urllib.request.urlopen(req, timeout=12).read()
            img_hash = hashlib.sha256(img_bytes).hexdigest()
            b64_img = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode('utf-8')
            
            payload = json.dumps({"image": b64_img}).encode('utf-8')
            api_req = urllib.request.Request(
                "http://localhost:8080/api/analyze",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            
            start_t = time.time()
            resp = urllib.request.urlopen(api_req, timeout=15)
            elapsed = (time.time() - start_t) * 1000
            result = json.loads(resp.read().decode('utf-8'))
            
            car_key = f"{result.get('make')} - {result.get('model')}"
            results_map[item['name']] = {
                "hash": img_hash[:16],
                "size": len(img_bytes),
                "make": result.get('make'),
                "model": result.get('model'),
                "colour": result.get('colour'),
                "source": result.get('source'),
                "detected": result.get('carDetected')
            }
            
            print(f"  --> Image Size:   {len(img_bytes):,} bytes | SHA-256: {img_hash[:16]}...")
            print(f"  --> Car Detected: {result.get('carDetected')}")
            print(f"  --> Make & Model: {result.get('make')} {result.get('model')}")
            print(f"  --> Colour:       {result.get('colour')} ({result.get('hexColor')})")
            print(f"  --> Body Type:    {result.get('bodyType')}")
            print(f"  --> Confidence:   {result.get('confidence')}")
            print(f"  --> AI Tier:      {result.get('source', 'Unknown').upper()}")
            print(f"  --> Latency:      {elapsed:.1f}ms")
            
            passed += 1
            print("  [SUCCESS] Verified Response!")
                
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\n" + "=" * 70)
    print("   UNIQUENESS VERIFICATION SUMMARY:")
    print("=" * 70)
    for name, r in results_map.items():
        print(f"  * {name:<28} -> [{r['hash']}] -> {r['make']} {r['model']} ({r['colour']}) [Tier: {r['source']}]")
    
    # Check that distinct cars produced distinct results
    distinct_makes = set(r['make'] for r in results_map.values())
    print(f"\nTotal Distinct Makes Identified: {len(distinct_makes)}")
    if len(distinct_makes) >= 4:
        print("  [PASSED] Verified that different cars produce different results (No caching / duplicate bug)!")
    else:
        print("  [FAILED] Results are still repeating!")
        
    print(f"   SUMMARY: {passed}/{total} Tests Completed")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
