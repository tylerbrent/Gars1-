# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 13:41:56 2025

@author: tyler
"""

import pandas as pd
import numpy as np
import tifffile
from pathlib import Path

def filter_spots_with_mask(csv_file, mask_file, output_file=None, pixel_size_um=None):
    """
    Filter spots from CSV file based on binary mask and calculate area metrics.
    
    Parameters:
    -----------
    csv_file : str or Path
        Path to CSV file containing spot coordinates
    mask_file : str or Path  
        Path to TIFF mask file
    output_file : str or Path, optional
        Output path for filtered CSV. If None, creates automatically.
    pixel_size_um : float, optional
        Pixel size in micrometers for area conversion
    
    Returns:
    --------
    dict
        Dictionary containing filtered dataframe and analysis metrics
    """
    
    # Read the mask file
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
    
    # Calculate mask area metrics
    mask_area_pixels = np.count_nonzero(mask)
    total_image_pixels = mask.shape[0] * mask.shape[1]
    mask_coverage_percent = (mask_area_pixels / total_image_pixels) * 100
    
    print(f"\nMask Area Analysis:")
    print(f"  Mask area: {mask_area_pixels} pixels")
    print(f"  Total image area: {total_image_pixels} pixels")
    print(f"  Mask coverage: {mask_coverage_percent:.2f}%")
    
    if pixel_size_um is not None:
        mask_area_um2 = mask_area_pixels * (pixel_size_um ** 2)
        print(f"  Mask area: {mask_area_um2:.2f} μm²")
    
    # Read the CSV file
    print(f"\nLoading spot data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Validate required columns
    required_columns = ['x', 'y']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"Loaded {len(df)} spots with columns: {list(df.columns)}")
    
    # Validate coordinate ranges
    mask_height, mask_width = mask.shape
    print(f"Mask dimensions: {mask_width} x {mask_height}")
    
    # Check coordinate bounds
    x_coords = df['x'].values
    y_coords = df['y'].values
    
    # Handle potential floating point coordinates by rounding
    x_coords = np.round(x_coords).astype(int)
    y_coords = np.round(y_coords).astype(int)
    
    # Find coordinates within mask bounds
    valid_coords = (
        (x_coords >= 0) & (x_coords < mask_width) &
        (y_coords >= 0) & (y_coords < mask_height)
    )
    
    invalid_count = np.sum(~valid_coords)
    if invalid_count > 0:
        print(f"Warning: {invalid_count} spots have coordinates outside mask bounds")
        print(f"X range: {np.min(x_coords)} to {np.max(x_coords)} (mask width: {mask_width})")
        print(f"Y range: {np.min(y_coords)} to {np.max(y_coords)} (mask height: {mask_height})")
    
    # Filter spots based on mask values
    # Only consider spots with valid coordinates
    valid_df = df[valid_coords].copy()
    valid_x = x_coords[valid_coords]
    valid_y = y_coords[valid_coords]
    
    # Check mask values at spot coordinates
    mask_values = mask[valid_y, valid_x]
    inside_mask = mask_values > 0
    
    # Filter dataframe to keep only spots inside mask
    filtered_df = valid_df[inside_mask].copy()
    
    # Calculate density metrics
    spots_inside_mask = len(filtered_df)
    spot_density_per_pixel = spots_inside_mask / mask_area_pixels if mask_area_pixels > 0 else 0
    
    print(f"\nFiltering Results:")
    print(f"  Original spots: {len(df)}")
    print(f"  Valid coordinates: {len(valid_df)}")
    print(f"  Inside mask: {spots_inside_mask}")
    print(f"  Filtered out: {len(df) - len(filtered_df)}")
    
    print(f"\nDensity Analysis:")
    print(f"  Spots inside mask: {spots_inside_mask}")
    print(f"  Mask area: {mask_area_pixels} pixels")
    print(f"  Spot density: {spot_density_per_pixel:.6f} spots/pixel")
    
    if pixel_size_um is not None:
        spot_density_per_um2 = spots_inside_mask / mask_area_um2 if mask_area_um2 > 0 else 0
        print(f"  Spot density: {spot_density_per_um2:.6f} spots/μm²")
    
    # Prepare results dictionary
    results = {
        'filtered_dataframe': filtered_df,
        'mask_area_pixels': mask_area_pixels,
        'total_image_pixels': total_image_pixels,
        'mask_coverage_percent': mask_coverage_percent,
        'spots_inside_mask': spots_inside_mask,
        'spot_density_per_pixel': spot_density_per_pixel,
        'original_spots': len(df),
        'valid_spots': len(valid_df),
        'filtered_spots': len(df) - len(filtered_df)
    }
    
    if pixel_size_um is not None:
        results['mask_area_um2'] = mask_area_um2
        results['spot_density_per_um2'] = spot_density_per_um2
        results['pixel_size_um'] = pixel_size_um
    
    # Save filtered results
    if output_file is None:
        # Create output filename automatically
        input_path = Path(csv_file)
        output_file = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"
    
    filtered_df.to_csv(output_file, index=False)
    print(f"\nSaved filtered spots to: {output_file}")
    
    # Save analysis summary
    # Save analysis summary
    summary_file = Path(output_file).parent / f"{Path(output_file).stem}_analysis_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:  # Added encoding
        f.write("Mask-Based Spot Filtering Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input Files:\n  CSV: {csv_file}\n  Mask: {mask_file}\n\n")
        f.write(f"Mask Analysis:\n  Area: {mask_area_pixels} pixels\n")
        f.write(f"  Coverage: {mask_coverage_percent:.2f}%\n")
        if pixel_size_um:
            f.write(f"  Area: {mask_area_um2:.2f} μm²\n")  # Now writes μ correctly
        f.write(f"\nSpot Analysis:\n  Original: {len(df)}\n  Inside mask: {spots_inside_mask}\n")
        f.write(f"  Density: {spot_density_per_pixel:.6f} spots/pixel\n")
        if pixel_size_um:
            f.write(f"  Density: {spot_density_per_um2:.6f} spots/μm²\n")


    
    print(f"Saved analysis summary to: {summary_file}")
    
    return results

def batch_analyze_masks(csv_files, mask_files, output_dir=None, pixel_size_um=None):
    """
    Batch process multiple CSV and mask file pairs.
    
    Parameters:
    -----------
    csv_files : list
        List of CSV file paths
    mask_files : list  
        List of mask file paths (must match csv_files order)
    output_dir : str or Path, optional
        Output directory for results
    pixel_size_um : float, optional
        Pixel size in micrometers
    
    Returns:
    --------
    pandas.DataFrame
        Summary dataframe with metrics for all processed files
    """
    
    if len(csv_files) != len(mask_files):
        raise ValueError("Number of CSV files must match number of mask files")
    
    if output_dir is None:
        output_dir = Path(csv_files[0]).parent / "batch_analysis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    summary_data = []
    
    for i, (csv_file, mask_file) in enumerate(zip(csv_files, mask_files)):
        print(f"\nProcessing pair {i+1}/{len(csv_files)}")
        print(f"CSV: {Path(csv_file).name}")
        print(f"Mask: {Path(mask_file).name}")
        
        # Create individual output file
        output_file = output_dir / f"filtered_{Path(csv_file).stem}.csv"
        
        try:
            results = filter_spots_with_mask(
                csv_file, mask_file, output_file, pixel_size_um
            )
            
            # Add file information to results
            results['csv_file'] = Path(csv_file).name
            results['mask_file'] = Path(mask_file).name
            results['pair_index'] = i + 1
            
            summary_data.append(results)
            
        except Exception as e:
            print(f"Error processing pair {i+1}: {e}")
            continue
    
    # Create summary dataframe
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv = output_dir / "batch_analysis_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nBatch analysis complete. Summary saved to: {summary_csv}")
        
        # Print overall statistics
        print(f"\nBatch Analysis Summary:")
        print(f"  Processed pairs: {len(summary_data)}")
        print(f"  Average mask area: {summary_df['mask_area_pixels'].mean():.0f} pixels")
        print(f"  Average spots inside mask: {summary_df['spots_inside_mask'].mean():.1f}")
        print(f"  Average spot density: {summary_df['spot_density_per_pixel'].mean():.6f} spots/pixel")
        
        return summary_df
    else:
        print("No files were successfully processed.")
        return None

# Example usage
if __name__ == "__main__":
    # Define file paths
    csv_file = r"C:\Users\tyler\OneDrive - Technion\Technion\Journal\9_2025\9_15_2025_qPCR and imaging\Gars Cy5_Tubb3 Cy3_Tubb 488\rep10_tubb_good\rep10_tubb rna_neurite1.csv"
    mask_file = r"C:\Users\tyler\OneDrive - Technion\Technion\Journal\9_2025\9_15_2025_qPCR and imaging\Gars Cy5_Tubb3 Cy3_Tubb 488\rep10_tubb_good\rep10_tubb_neurite1_mask.tif"
    
    # Optional: specify pixel size for area conversion (e.g., 0.1 μm/pixel)
    pixel_size_um = 0.1  # Adjust based on your imaging setup
    
    # Filter spots with enhanced analysis
    results = filter_spots_with_mask(csv_file, mask_file, pixel_size_um=pixel_size_um)
    
    # Access results
    filtered_spots = results['filtered_dataframe']
    print(f"\nKey Results:")
    print(f"Mask area: {results['mask_area_pixels']} pixels")
    print(f"Spots inside mask: {results['spots_inside_mask']}")
    print(f"Spot density: {results['spot_density_per_pixel']:.6f} spots/pixel")
    
    if 'spot_density_per_um2' in results:
        print(f"Spot density: {results['spot_density_per_um2']:.6f} spots/μm²")
    
    # Optional: Display summary statistics of filtered data
    if len(filtered_spots) > 0:
        print("\nFiltered data summary:")
        print(filtered_spots.describe())
