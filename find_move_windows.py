import re

results = []
with open('main.py', 'r', encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(f, 1):
        if 'moveWindow' in line:
            results.append(f"{i}: {line.strip()}")

with open('move_windows_list.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))
