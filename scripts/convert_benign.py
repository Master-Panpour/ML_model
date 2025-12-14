import pandas as pd
import os

if __name__ == "__main__":
    src = os.path.join('data', 'benign_Train.txt')
    dst = os.path.join('data', 'benign_Train.csv')
    try:
        with open(src, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [l.strip() for l in f if l.strip()]
        df = pd.DataFrame(lines, columns=['url'])
        df['label'] = 0
        df.to_csv(dst, index=False)
        print(f"Converted {src} -> {dst} ({len(df)} rows)")
    except FileNotFoundError:
        print(f"Source file not found: {src}")
    except Exception as e:
        print("Conversion failed:", e)
