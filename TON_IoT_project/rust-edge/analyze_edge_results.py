import json
import pandas as pd
from collections import defaultdict

multi_csv = r'c:\Users\Shreyas A\PROJECTS\BAZT-TON_IoT\TON_IoT_project\rust-edge\multi_attack_eval.csv'
jsonl_file = r'c:\Users\Shreyas A\PROJECTS\BAZT-TON_IoT\TON_IoT_project\rust-edge\decisions_multi_attack_new.jsonl'

df = pd.read_csv(multi_csv, usecols=['Attack_category'])
cats = df['Attack_category'].values

records = []
with open(jsonl_file, 'r') as f:
    for line in f:
        records.append(json.loads(line))

print(f'Total records in jsonl: {len(records)}, total in CSV: {len(cats)}')

by_cat = defaultdict(lambda: {'ALLOW': 0, 'VERIFY': 0, 'BLOCK': 0})
for idx, rec in enumerate(records):
    cat = cats[idx]
    dec = rec['decision']
    by_cat[cat][dec] += 1

print(f"{'Threat Class':<25} | {'Frames':<8} | {'ALLOW':<10} | {'VERIFY':<10} | {'Pure BLOCK'}")
print('-' * 75)
total_atk_allow = 0
total_atk_verify = 0
total_atk_block = 0
total_atk_frames = 0

for cat, counts in by_cat.items():
    tot = sum(counts.values())
    al = counts['ALLOW']
    ve = counts['VERIFY']
    bl = counts['BLOCK']
    al_pct = (al / tot) * 100
    ve_pct = (ve / tot) * 100
    bl_pct = (bl / tot) * 100
    if cat != 'Normal':
        total_atk_allow += al
        total_atk_verify += ve
        total_atk_block += bl
        total_atk_frames += tot
    print(f"{cat:<25} | {tot:<8} | {al_pct:>5.2f}% ({al:<4}) | {ve_pct:>5.2f}% ({ve:<4}) | {bl_pct:>5.2f}% ({bl:<4})")

print('-' * 75)
print(f"{'Total Attack Stream':<25} | {total_atk_frames:<8} | {total_atk_allow/total_atk_frames*100:>5.2f}% ({total_atk_allow:<4}) | {total_atk_verify/total_atk_frames*100:>5.2f}% ({total_atk_verify:<4}) | {total_atk_block/total_atk_frames*100:>5.2f}% ({total_atk_block:<4})")
