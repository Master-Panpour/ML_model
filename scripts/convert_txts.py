import argparse
import os
import pandas as pd


def infer_label_from_name(name: str):
    n = name.lower()
    if "benign" in n or "good" in n or "legit" in n:
        return 0
    if "malign" in n or "malicious" in n or "mal" in n or "phish" in n:
        return 1
    # unknown
    return None


def convert_file(input_path: str, output_path: str, forced_label):
    # Read file robustly: try detecting CSV-like lines, otherwise one-url-per-line
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        sample = [line.strip() for _, line in zip(range(200), f) if line.strip()]

    # If sample contains commas and looks like CSV, delegate to pandas
    use_pandas = any(("," in line and line.count(",") >= 1) for line in sample)

    if use_pandas:
        df = pd.read_csv(input_path, engine="python")
        if "url" in df.columns:
            df = df[["url"]].copy()
        else:
            # take first column as url
            df = df.iloc[:, [0]].copy()
            df.columns = ["url"]
    else:
        # simple one-url-per-line format
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
        df = pd.DataFrame(lines, columns=["url"])

    # Add label
    if forced_label is not None:
        df["label"] = forced_label
    else:
        # leave label out if unknown
        df["label"] = None

    # Write CSV
    df.to_csv(output_path, index=False)
    return len(df)


def main():
    p = argparse.ArgumentParser(description="Convert .txt URL lists to CSV with optional labels")
    p.add_argument("paths", nargs="*", help="Files or directories to scan for .txt files. If empty, current directory is used.")
    p.add_argument("--out-dir", default=".", help="Output directory for CSV files")
    p.add_argument("--label", type=int, choices=[0, 1], help="Force label for all converted files (0 benign, 1 malicious)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV files")
    args = p.parse_args()

    if not args.paths:
        args.paths = ["."]

    txt_files = []
    for pth in args.paths:
        if os.path.isdir(pth):
            for root, _, files in os.walk(pth):
                for fn in files:
                    if fn.lower().endswith(".txt"):
                        txt_files.append(os.path.join(root, fn))
        elif os.path.isfile(pth) and pth.lower().endswith(".txt"):
            txt_files.append(pth)

    if not txt_files:
        print("No .txt files found to convert.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for txt in txt_files:
        name = os.path.basename(txt)
        base = os.path.splitext(name)[0]
        out_csv = os.path.join(args.out_dir, f"{base}.csv")
        if os.path.exists(out_csv) and not args.overwrite:
            print(f"Skipping {txt} -> {out_csv} (exists). Use --overwrite to replace.")
            continue

        forced_label = args.label
        if forced_label is None:
            forced_label = infer_label_from_name(name)

        try:
            n = convert_file(txt, out_csv, forced_label)
            print(f"Converted {txt} -> {out_csv} ({n} rows). Label={'None' if forced_label is None else forced_label}")
        except Exception as e:
            print(f"Failed to convert {txt}: {e}")


if __name__ == "__main__":
    main()
