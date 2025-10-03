#!/usr/bin/env python3
"""
Main script for Pokemon palette swapping.
Apply color palette from one Pokemon to another automatically.
"""

import argparse
import numpy as np
from palette_extraction import load_image, save_image
from palette_matching import OptimalPaletteSwap
import matplotlib.pyplot as plt


def visualize_palettes(source_palette, target_palette, permutation, save_path=None):
    """Visualize source and target palettes with matching."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))

    # Source palette
    source_img = np.array([source_palette])
    axes[0].imshow(source_img)
    axes[0].set_title('Source Palette')
    axes[0].axis('off')

    # Target palette
    target_img = np.array([target_palette])
    axes[1].imshow(target_img)
    axes[1].set_title('Target Palette')
    axes[1].axis('off')

    # Matched palette
    matched_palette = target_palette[permutation]
    matched_img = np.array([matched_palette])
    axes[2].imshow(matched_img)
    axes[2].set_title('Matched Palette')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Palette visualization saved to {save_path}")

    plt.close()


def visualize_results(source_image, result_image, save_path=None):
    """Visualize source and result images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(source_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(result_image)
    axes[1].set_title('Recolored Image')
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Result visualization saved to {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Apply color palette from one Pokemon image to another'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to source Pokemon image'
    )
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        help='Path to target Pokemon image (to extract palette from)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='recolored_pokemon.png',
        help='Path to save recolored image'
    )
    parser.add_argument(
        '--num-colors',
        type=int,
        default=5,
        help='Number of colors in palette (default: 5)'
    )
    parser.add_argument(
        '--hue-steps',
        type=int,
        default=8,
        help='Hue transformation steps (default: 8)'
    )
    parser.add_argument(
        '--sat-steps',
        type=int,
        default=3,
        help='Saturation transformation steps (default: 3)'
    )
    parser.add_argument(
        '--val-steps',
        type=int,
        default=3,
        help='Value transformation steps (default: 3)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device for computation (cuda or cpu, default: auto)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization images'
    )

    args = parser.parse_args()

    # Load images
    print(f"Loading source image: {args.source}")
    source_image = load_image(args.source)

    print(f"Loading target image: {args.target}")
    target_image = load_image(args.target)

    # Initialize pipeline
    print(f"\nInitializing pipeline with {args.num_colors} colors...")
    swapper = OptimalPaletteSwap(
        num_colors=args.num_colors,
        hue_steps=args.hue_steps,
        sat_steps=args.sat_steps,
        val_steps=args.val_steps,
        device=args.device
    )

    # Perform palette swap
    print("\nPerforming palette swap...")
    result_image, info = swapper.swap_palette(source_image, target_image)

    # Save result
    save_image(result_image, args.output)
    print(f"\nRecolored image saved to: {args.output}")

    # Print info
    print(f"\nResults:")
    print(f"  Source palette: {info['source_palette'].shape}")
    print(f"  Target palette: {info['target_palette'].shape}")
    print(f"  Optimal permutation: {info['permutation']}")
    print(f"  Distance metric: {info['distance']:.6f}")

    # Visualizations
    if args.visualize:
        palette_vis_path = args.output.replace('.png', '_palettes.png')
        visualize_palettes(
            info['source_palette'],
            info['target_palette'],
            info['permutation'],
            save_path=palette_vis_path
        )

        result_vis_path = args.output.replace('.png', '_comparison.png')
        visualize_results(
            source_image,
            result_image,
            save_path=result_vis_path
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
