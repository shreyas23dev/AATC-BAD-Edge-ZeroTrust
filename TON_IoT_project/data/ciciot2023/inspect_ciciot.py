import urllib.request
import io
import pandas as pd

file_path = "CSV/CSV/Benign_Final/BenignTraffic.pcap.csv"
url = f"https://huggingface.co/datasets/bencorn/CIC-IoT-2023/resolve/main/{file_path}"
print(f"Connecting to: {url}")

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-100000'})
with urllib.request.urlopen(req) as resp:
    data = resp.read()
    
# Decode lines
text = data.decode('utf-8', errors='ignore')
lines = text.splitlines()
print(f"Read {len(lines)} lines from range header.")
header = lines[0].split(',')
print(f"Total columns: {len(header)}")
print("Columns:")
for i, col in enumerate(header):
    print(f" {i+1}. {col}")

# Parse sample rows with pandas
df = pd.read_csv(io.StringIO("\n".join(lines[:-1]))) # omit last potentially partial line
print("\nSample DataFrame shape:", df.shape)
print("Label values:", df['label'].value_counts() if 'label' in df.columns else "No label col")
