import urllib.request

attack_files = [
    "CSV/CSV/DDoS-UDP_Flood/DDoS-UDP_Flood.pcap.csv",
    "CSV/CSV/DDoS-ICMP_Flood/DDoS-ICMP_Flood.pcap.csv",
    "CSV/CSV/Recon-PortScan/Recon-PortScan.pcap.csv",
    "CSV/CSV/VulnerabilityScan/VulnerabilityScan.pcap.csv",
    "CSV/CSV/DictionaryBruteForce/DictionaryBruteForce.pcap.csv"
]

for f in attack_files:
    url = f"https://huggingface.co/datasets/bencorn/CIC-IoT-2023/resolve/main/{f}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-4096'})
    try:
        with urllib.request.urlopen(req) as resp:
            data = resp.read().decode('utf-8', errors='ignore')
            lines = data.splitlines()
            header = lines[0].split(',')
            sample = lines[1].split(',') if len(lines) > 1 else []
            print(f"File: {f.split('/')[-1]}")
            print(f"  Cols: {len(header)}, First col: {header[0]}, Last col: {header[-1]}")
            print(f"  Sample length: {len(sample)}")
    except Exception as e:
        print(f"Error {f}: {e}")
