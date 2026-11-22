import urllib.request
import json

url = "https://huggingface.co/api/datasets/bencorn/CIC-IoT-2023"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())

files = [s['rfilename'] for s in data.get('siblings', []) if s['rfilename'].endswith('.csv')]
print(f"Total CSV files available: {len(files)}")

# Group by directory
dirs = set()
for f in files:
    parts = f.split('/')
    if len(parts) >= 3:
        dirs.add(parts[2])
        
print("Available attack/traffic categories:")
for d in sorted(dirs):
    print(" -", d)
