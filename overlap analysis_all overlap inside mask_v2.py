# -*- coding: utf-8 -*-
"""
Combined colocalization analysis with mask filtering - CORRECTED VERSION
Created on Thu Jul 17 2025

@author: tyler
"""

import pandas as pd
import numpy as np
import tifffile
from math import acos, sqrt, pi
from itertools import product
from pathlib import Path
import os

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def circle_intersection_area(r1, r2, d):
    """Calculate intersection area of two circles"""
    if d >= r1 + r2:  # No intersection
        return 0.0
    if d <= abs(r1 - r2):  # One circle inside another
        return min(pi*r1**2, pi*r2**2)
    
    # Partial intersection formula
    term1 = r1**2 * acos((d**2 + r1**2 - r2**2)/(2*d*r1))
    term2 = r2**2 * acos((d**2 + r2**2 - r1**2)/(2*d*r2))
    term3 = 0.5 * sqrt((-d + r1 + r2)*(d + r1 - r2)*(d - r1 + r2)*(d + r1 + r2))
    return term1 + term2 - term3

def load_protein_data(path, suffix):
    """Load and rename columns for each protein dataset"""
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: 'Spot No.'})
    return df.rename(columns=lambda x: x if x == 'Spot No.' else f'{x}_{suffix}')

def load_and_process_mask(mask_file, pixel_size_um):
    """Load and process binary mask file"""
    print(f"Loading mask from: {mask_file}")
    mask = tifffile.imread(mask_file)
    
    # Handle different mask value conventions
    unique_values = np.unique(mask)
    print(f"Mask unique values: {unique_values}")
    
    # Convert mask to binary (0 and 1) if needed
    if len(unique_values) == 2:
        if 255 in unique_values:
            # Convert 0/255 to 0/1
            mask = (mask > 0).astype(np.uint8)
            print("Converted mask from 0/255 to 0/1 format")
        elif np.max(unique_values) == 1:
            # Already in 0/1 format
            print("Mask already in 0/1 format")
    else:
        raise ValueError(f"Mask should be binary but contains {len(unique_values)} unique values")
    
    # Calculate mask metrics
    mask_area_pixels = np.count_nonzero(mask)
    total_image_pixels = mask.shape[0] * mask.shape[1]
    mask_coverage_percent = (mask_area_pixels / total_image_pixels) * 100
    
    # Calculate area in microns
    mask_area_microns = mask_area_pixels * (pixel_size_um ** 2)
    
    print(f"Mask dimensions: {mask.shape[1]} x {mask.shape[0]} pixels")
    print(f"Mask dimensions: {mask.shape[1] * pixel_size_um:.2f} x {mask.shape[0] * pixel_size_um:.2f} microns")
    print(f"Mask area: {mask_area_pixels} pixels ({mask_area_microns:.2f} μm²)")
    print(f"Mask coverage: {mask_coverage_percent:.2f}%")
    
    return mask, mask_area_pixels, mask_area_microns

def filter_spots_by_mask(df, mask, x_col, y_col, pixel_size_um):
    """Filter spots dataframe based on mask, converting coords from microns to pixels"""
    mask_height, mask_width = mask.shape
    
    # Convert spot coordinates from microns to pixels
    x_coords_pixels = np.round(df[x_col].values / pixel_size_um).astype(int)
    y_coords_pixels = np.round(df[y_col].values / pixel_size_um).astype(int)
    
    print(f"  Coordinate conversion:")
    print(f"    X range (microns): {df[x_col].min():.2f} to {df[x_col].max():.2f}")
    print(f"    Y range (microns): {df[y_col].min():.2f} to {df[y_col].max():.2f}")
    print(f"    X range (pixels): {x_coords_pixels.min()} to {x_coords_pixels.max()}")
    print(f"    Y range (pixels): {y_coords_pixels.min()} to {y_coords_pixels.max()}")
    print(f"    Mask dimensions (pixels): {mask_width} x {mask_height}")
    
    # Find coordinates within mask bounds
    valid_coords = (
        (x_coords_pixels >= 0) & (x_coords_pixels < mask_width) &
        (y_coords_pixels >= 0) & (y_coords_pixels < mask_height)
    )
    
    invalid_count = np.sum(~valid_coords)
    if invalid_count > 0:
        print(f"  Warning: {invalid_count} spots have coordinates outside mask bounds")
    
    # Filter to valid coordinates first
    valid_df = df[valid_coords].copy()
    valid_x = x_coords_pixels[valid_coords]
    valid_y = y_coords_pixels[valid_coords]
    
    # Check mask values at spot coordinates
    mask_values = mask[valid_y, valid_x]
    inside_mask = mask_values > 0
    
    # Filter dataframe to keep only spots inside mask
    filtered_df = valid_df[inside_mask].copy()
    
    print(f"  Original spots: {len(df)}")
    print(f"  Valid coordinates: {len(valid_df)}")
    print(f"  Inside mask: {len(filtered_df)}")
    print(f"  Filtered out: {len(df) - len(filtered_df)}")
    
    return filtered_df

def masked_colocalization_analysis(p1_csv, p2_csv, mask_file, pixel_size_um, output_dir='masked_colocalization_analysis'):
    """
    Perform colocalization analysis only within mask region
    
    Parameters:
    -----------
    p1_csv : str
        Path to first protein CSV file
    p2_csv : str
        Path to second protein CSV file
    mask_file : str
        Path to binary mask TIFF file
    pixel_size_um : float
        Pixel size in micrometers (e.g., 0.1 for 100nm pixels)
    output_dir : str
        Output directory for results
    """
    
    # Load mask
    mask, mask_area_pixels, mask_area_microns = load_and_process_mask(mask_file, pixel_size_um)
    
    # Load protein datasets
    print(f"\nLoading protein datasets...")
    df_p1 = load_protein_data(p1_csv, 'p1')
    df_p2 = load_protein_data(p2_csv, 'p2')
    
    print(f"Loaded {len(df_p1)} spots for protein 1")
    print(f"Loaded {len(df_p2)} spots for protein 2")
    
    # Filter both datasets by mask
    print(f"\nFiltering protein 1 spots by mask...")
    df_p1_filtered = filter_spots_by_mask(df_p1, mask, 'X_p1', 'Y_p1', pixel_size_um)
    
    print(f"\nFiltering protein 2 spots by mask...")
    df_p2_filtered = filter_spots_by_mask(df_p2, mask, 'X_p2', 'Y_p2', pixel_size_um)
    
    # Analysis parameters
    THRESHOLDS = {
        '25pct': 0.25,
        '50pct': 0.50,
        '75pct': 0.75,
        '100pct': 1.00
    }
    
    # Initialize results storage
    results = {threshold: [] for threshold in THRESHOLDS}
    p1_counts = {threshold: set() for threshold in THRESHOLDS}
    p2_counts = {threshold: set() for threshold in THRESHOLDS}
    
    print(f"\nPerforming colocalization analysis on filtered data...")
    print(f"Analyzing {len(df_p1_filtered)} x {len(df_p2_filtered)} = {len(df_p1_filtered) * len(df_p2_filtered)} spot pairs")
    
    # Process all filtered spot pairs
    for (idx1, row1), (idx2, row2) in product(df_p1_filtered.iterrows(), df_p2_filtered.iterrows()):
        # Calculate geometric properties
        r1 = sqrt(row1['Area_p1'] / pi)
        r2 = sqrt(row2['Area_p2'] / pi)
        d = calculate_distance(row1['X_p1'], row1['Y_p1'], row2['X_p2'], row2['Y_p2'])
        intersection = circle_intersection_area(r1, r2, d)
        
        # Check all thresholds
        for threshold_name, threshold_value in THRESHOLDS.items():
            if (intersection >= threshold_value * row1['Area_p1'] or 
                intersection >= threshold_value * row2['Area_p2']):
                
                # Record results
                results[threshold_name].append({
                    'Protein1_Spot': row1['Spot No.'],
                    'Protein2_Spot': row2['Spot No.'],
                    'X1': row1['X_p1'],
                    'Y1': row1['Y_p1'],
                    'X2': row2['X_p2'],
                    'Y2': row2['Y_p2'],
                    'Area1': row1['Area_p1'],
                    'Area2': row2['Area_p2'],
                    'Intersection_Area': intersection,
                    'Overlap_Pct1': intersection / row1['Area_p1'],
                    'Overlap_Pct2': intersection / row2['Area_p2'],
                    'Distance': d
                })
                
                # Track unique spots
                p1_counts[threshold_name].add(row1['Spot No.'])
                p2_counts[threshold_name].add(row2['Spot No.'])
    
    # Generate output files
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual threshold results
    for threshold_name in THRESHOLDS:
        if results[threshold_name]:
            df = pd.DataFrame(results[threshold_name])
            df.to_csv(f'{output_dir}/{threshold_name}_results_masked.csv', index=False)
            print(f'Saved {len(df)} pairs for {threshold_name} threshold')
    
    # Create comprehensive summary
    summary_data = []
    total_p1_original = len(df_p1)
    total_p2_original = len(df_p2)
    total_p1_filtered = len(df_p1_filtered)
    total_p2_filtered = len(df_p2_filtered)
    
    for threshold_name in THRESHOLDS:
        summary_data.append({
            'Threshold': threshold_name,
            'Original_P1_Spots': total_p1_original,
            'Original_P2_Spots': total_p2_original,
            'Filtered_P1_Spots': total_p1_filtered,
            'Filtered_P2_Spots': total_p2_filtered,
            'Coloc_P1_Spots': len(p1_counts[threshold_name]),
            'Coloc_P2_Spots': len(p2_counts[threshold_name]),
            'P1_Ratio_of_Filtered': len(p1_counts[threshold_name]) / total_p1_filtered if total_p1_filtered > 0 else 0,
            'P2_Ratio_of_Filtered': len(p2_counts[threshold_name]) / total_p2_filtered if total_p2_filtered > 0 else 0,
            'P1_Ratio_of_Original': len(p1_counts[threshold_name]) / total_p1_original if total_p1_original > 0 else 0,
            'P2_Ratio_of_Original': len(p2_counts[threshold_name]) / total_p2_original if total_p2_original > 0 else 0,
            'Colocalization_Pairs': len(results[threshold_name])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_dir}/summary_all_thresholds_masked.csv', index=False)
    
    # Save filtering summary
    filtering_summary = {
        'mask_file': mask_file,
        'pixel_size_um': pixel_size_um,
        'mask_area_pixels': mask_area_pixels,
        'mask_area_microns': mask_area_microns,
        'p1_original_spots': total_p1_original,
        'p2_original_spots': total_p2_original,
        'p1_filtered_spots': total_p1_filtered,
        'p2_filtered_spots': total_p2_filtered,
        'p1_retention_rate': total_p1_filtered / total_p1_original if total_p1_original > 0 else 0,
        'p2_retention_rate': total_p2_filtered / total_p2_original if total_p2_original > 0 else 0
    }
    
    filtering_df = pd.DataFrame([filtering_summary])
    filtering_df.to_csv(f'{output_dir}/filtering_summary.csv', index=False)
    
    print(f"\nMasked Analysis Summary:")
    print(f"Pixel size: {pixel_size_um} μm/pixel")
    print(f"Mask area: {mask_area_pixels} pixels ({mask_area_microns:.2f} μm²)")
    print(f"P1 spots: {total_p1_original} → {total_p1_filtered} ({total_p1_filtered/total_p1_original*100:.1f}% retained)")
    print(f"P2 spots: {total_p2_original} → {total_p2_filtered} ({total_p2_filtered/total_p2_original*100:.1f}% retained)")
    print("\nColocalization results (filtered spots only):")
    print(summary_df[['Threshold', 'Coloc_P1_Spots', 'P1_Ratio_of_Filtered', 'Coloc_P2_Spots', 'P2_Ratio_of_Filtered', 'Colocalization_Pairs']].to_string(index=False))
    print(f"\nResults saved to: {os.path.abspath(output_dir)}")
    
    return summary_df, filtering_summary

# Usage example
if __name__ == "__main__":
    # Set working directory
    os.chdir(r"C:\Users\tyler\OneDrive - Technion\Technion\Journal\9_2025\9_17_2025_imaging\GARS 495_tRNA Lys Cy5_TUBB 488\rep1")
    
    # Define file paths
    p1_csv = r"C:\Users\tyler\OneDrive - Technion\Technion\Journal\9_2025\9_17_2025_imaging\GARS 495_tRNA Lys Cy5_TUBB 488\rep1\rep1_gars_neurite4_spots.csv"
    p2_csv = r"C:\Users\tyler\OneDrive - Technion\Technion\Journal\9_2025\9_17_2025_imaging\GARS 495_tRNA Lys Cy5_TUBB 488\rep1\rep1_lys_neurite4_spots.csv"
    mask_file = r"C:\Users\tyler\OneDrive - Technion\Technion\Journal\9_2025\9_17_2025_imaging\GARS 495_tRNA Lys Cy5_TUBB 488\rep1\rep1_tub_neurite4_mask.tif"
    
    # IMPORTANT: Set your pixel size in micrometers per pixel
    pixel_size_um = 0.1  # Example: 100nm pixels = 0.1 μm/pixel
    
    # Run masked colocalization analysis
    summary, filtering_info = masked_colocalization_analysis(p1_csv, p2_csv, mask_file, pixel_size_um)
