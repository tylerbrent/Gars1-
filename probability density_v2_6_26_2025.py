# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 11:27:51 2025
@author: tyler
"""
# -*- coding: utf-8 -*-
"""
Modified on Thu Jun 26 2025
@author: tyler  (edited by assistant)
"""
import pandas as pd
import numpy as np
import os

def analyze_spots_by_distance(csv_file_path):
    """
    Analyze spot distributions along neurites in three biologically-relevant zones:
      • 0–20 µm from the soma (proximal)
      • middle segment
      • final 20 µm of the neurite (distal)
    For each neurite (distance column) the script reports:
      • total neurite length (µm)
      • counts of spots in the three zones
      • basic summary statistics
      • per-column CSV + global text summary
    """
    # ---------- 1. Load data ----------
    try:
        df = pd.read_csv(csv_file_path)
        print("CSV file loaded successfully!")
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    print(f"Data shape: {df.shape}")
    print("Columns:", df.columns.tolist()[:10], "...\n")
    # ---------- 2. Ask for scale ----------
    pixels_per_micron = 11.7559791404
    # Heuristically locate distance columns
    distance_columns = [c for c in df.columns
                        if ('x' in c.lower() or 'distance' in c.lower()) and df[c].dtype != object]
    if not distance_columns:
        col_name = input("Enter the column name that holds the x-distance values: ")
        if col_name in df.columns:
            distance_columns = [col_name]
        else:
            print(f"Column '{col_name}' not found.")
            return
    # ---------- 3. Prepare summary containers ----------
    summary_lines = [
        "SPOT DISTANCE ANALYSIS SUMMARY",
        "=" * 55,
        f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input file   : {csv_file_path}",
        f"Pixels / µm  : {pixels_per_micron}",
        f"Distance cols: {', '.join(distance_columns)}",
        ""
    ]
    # Get directory and base name of input file for output naming
    input_dir = os.path.dirname(csv_file_path)
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    # ---------- 4. Process each neurite ----------
    for col in distance_columns:
        print("\n" + "="*40 + f"\nAnalyzing '{col}'")
        # Convert to microns and drop NaNs
        dist_um = df[col].astype(float) / pixels_per_micron
        dist_um = dist_um.dropna()
        neurite_len = dist_um.max()           # total length measured (µm)
        total_spots = dist_um.size
        # ----- 4a. Define the three zones -----
        prox_mask   = dist_um <= 20
        dist_thresh = max(neurite_len - 20, 20)   # prevents negative/overlap
        dist_mask   = dist_um >= dist_thresh
        mid_mask    = (~prox_mask) & (~dist_mask)
        counts = {
            "0–20 µm (proximal)" : prox_mask.sum(),
            "middle segment"     : mid_mask.sum(),
            f"last 20 µm ({neurite_len - 20:.1f}–{neurite_len:.1f} µm)" : dist_mask.sum()
        }
        # ----- 4b. Print results -----
        print(f"Neurite length : {neurite_len:.2f} µm")
        print(f"Total spots    : {total_spots}")
        for zone, n in counts.items():
            print(f"{zone:25s}: {n}")
        mean_d, med_d = dist_um.mean(), dist_um.median()
        print(f"Mean distance  : {mean_d:.2f} µm")
        print(f"Median distance: {med_d:.2f} µm")
        # ----- 4c. Save per-neurite CSV -----
        out_df = (pd.Series(counts, name="Number_of_Spots")
                    .reset_index()
                    .rename(columns={"index": "Distance_Zone"}))
        out_df.insert(1, "Neurite_Length_µm", neurite_len)
        csv_out = os.path.join(input_dir, f"{base_name}_{col.replace(' ', '_')}_probability_distribution.csv")
        out_df.to_csv(csv_out, index=False)
        print(f"Saved zone counts to '{csv_out}'")
        # ----- 4d. Append to global summary -----
        summary_lines.extend([
            f"COLUMN: {col}",
            f"  Neurite length        : {neurite_len:.2f} µm",
            f"  Total spots           : {total_spots}",
            f"  Spots 0–20 µm (prox)  : {counts['0–20 µm (proximal)']}",
            f"  Spots middle segment  : {counts['middle segment']}",
            f"  Spots last 20 µm      : {counts[list(counts.keys())[2]]}",
            f"  Mean distance         : {mean_d:.2f} µm",
            f"  Median distance       : {med_d:.2f} µm",
            "-" * 55
        ])
    # ---------- 5. Write summary file ----------
    summary_txt = os.path.join(input_dir, f"{base_name}_probability_distribution_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\nFull summary saved to '{summary_txt}'")

def main():
    print("Spot Distance Analysis Tool")
    print("="*32)
    # Hard-coded for convenience; replace or prompt as needed
    csv_path = r"C:\Users\tyler\OneDrive - Technion\Technion\Journal\9_2025\9_15_2025_qPCR and imaging\Gars Cy5_Tubb3 Cy3_Tubb 488\rep10_tubb_good\rep10_tubb rna_neurite7_filtered.csv"
    analyze_spots_by_distance(csv_path)

if __name__ == "__main__":
    main()

print("Current working directory:", os.getcwd())
