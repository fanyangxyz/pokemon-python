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
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s:%(lineno)d - %(message)s')


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
        logging.info(f"Palette visualization saved to {save_path}")

    plt.close()


def visualize_results(source_image, target_image, result_image, save_path=None):
    """Visualize source, target, and result images side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(source_image)
    axes[0].set_title('Source Image\n(Original)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(target_image)
    axes[1].set_title('Target Image\n(Palette Source)', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(result_image)
    axes[2].set_title('Result\n(Source with Target Palette)', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Result visualization saved to {save_path}")

    plt.close()


def visualize_all_permutations(source_image, all_results, best_perm, best_distance, save_path=None):
    """Visualize all permutation results with the best one highlighted."""
    import math

    permutations = all_results['permutations']
    distances = all_results['distances']
    images = all_results['images']

    num_perms = len(permutations)

    # Sort by distance
    sorted_indices = np.argsort(distances)

    # Create grid layout
    ncols = min(6, num_perms)
    nrows = math.ceil(num_perms / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.5))

    if num_perms == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, sorted_idx in enumerate(sorted_indices):
        perm = permutations[sorted_idx]
        dist = distances[sorted_idx]
        img = images[sorted_idx]

        axes[idx].imshow(img)

        # Check if this is the best permutation
        is_best = np.array_equal(perm, best_perm)

        title = f"Perm: {perm}\nDist: {dist:.4f}"
        if is_best:
            title = f"★ BEST ★\n{title}"
            axes[idx].set_title(title, fontweight='bold', color='green', fontsize=10)
            # Add border
            for spine in axes[idx].spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        else:
            axes[idx].set_title(title, fontsize=9)

        axes[idx].axis('off')

    # Hide empty subplots
    for idx in range(num_perms, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'All {num_perms} Permutations (sorted by distance)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"All permutations visualization saved to {save_path}")

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
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers for permutation testing (default: 4)'
    )
    parser.add_argument(
        '--extract-only',
        action='store_true',
        help='Only extract palettes and save them, skip palette matching'
    )
    parser.add_argument(
        '--show-all-permutations',
        action='store_true',
        help='Visualize all permutation results with the optimal one highlighted'
    )
    parser.add_argument(
        '--extraction-method',
        type=str,
        default='kmeans',
        choices=['kmeans', 'blind_separation'],
        help='Palette extraction method: kmeans (fast, default) or blind_separation (gradient descent, slower)'
    )

    args = parser.parse_args()

    # Load images
    logging.info(f"Loading source image: {args.source}")
    source_image = load_image(args.source)

    logging.info(f"Loading target image: {args.target}")
    target_image = load_image(args.target)

    # Initialize pipeline
    logging.info(f"Initializing pipeline with {args.num_colors} colors using {args.extraction_method} extraction...")
    swapper = OptimalPaletteSwap(
        num_colors=args.num_colors,
        hue_steps=args.hue_steps,
        sat_steps=args.sat_steps,
        val_steps=args.val_steps,
        device=args.device,
        extraction_method=args.extraction_method
    )

    if args.extract_only:
        # Only extract palettes and save
        logging.info(f"Extracting palettes only using {args.extraction_method}...")
        from palette_extraction import PaletteExtractor
        from concurrent.futures import ThreadPoolExecutor

        extractor = PaletteExtractor(num_colors=args.num_colors, method=args.extraction_method)

        # Extract both palettes in parallel
        logging.info("Extracting source and target palettes in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_source = executor.submit(extractor.extract_palette, source_image)
            future_target = executor.submit(extractor.extract_palette, target_image)

            source_palette, source_weights = future_source.result()
            logging.info("Source palette extraction complete")

            target_palette, target_weights = future_target.result()
            logging.info("Target palette extraction complete")

        # Save palette information
        import pickle
        palette_data = {
            'source_palette': source_palette,
            'source_weights': source_weights,
            'target_palette': target_palette,
            'target_weights': target_weights
        }
        palette_file = args.output.replace('.png', '_palettes.pkl')
        with open(palette_file, 'wb') as f:
            pickle.dump(palette_data, f)
        logging.info(f"Palette data saved to: {palette_file}")

        logging.info(f"Source palette shape: {source_palette.shape}")
        logging.info(f"Source palette colors:\n{source_palette}")
        logging.info(f"Target palette shape: {target_palette.shape}")
        logging.info(f"Target palette colors:\n{target_palette}")

        # Visualize if requested
        if args.visualize:
            palette_vis_path = args.output.replace('.png', '_palettes.png')
            # Create simple palette visualization
            visualize_palettes(
                source_palette,
                target_palette,
                np.arange(args.num_colors),  # Identity permutation for now
                save_path=palette_vis_path
            )

            # Visualize palette extraction results
            extraction_vis_path = args.output.replace('.png', f'_{args.extraction_method}_source.png')
            extractor.visualize_kmeans_result(
                source_image,
                source_palette,
                source_weights,
                save_path=extraction_vis_path
            )

            extraction_vis_path_target = args.output.replace('.png', f'_{args.extraction_method}_target.png')
            extractor.visualize_kmeans_result(
                target_image,
                target_palette,
                target_weights,
                save_path=extraction_vis_path_target
            )

        logging.info("Palette extraction complete!")
        return

    # Perform palette swap
    logging.info("Performing palette swap...")

    # Enable return_all_results if we need to visualize all permutations
    if args.show_all_permutations:
        swapper._return_all_results = True

    result_image, info = swapper.swap_palette(
        source_image,
        target_image,
        use_parallel=not args.no_parallel,
        num_workers=args.workers
    )

    # Save result
    save_image(result_image, args.output)
    logging.info(f"Recolored image saved to: {args.output}")

    # Print info
    logging.info(f"Results:")
    logging.info(f"  Source palette: {info['source_palette'].shape}")
    logging.info(f"  Target palette: {info['target_palette'].shape}")
    logging.info(f"  Optimal permutation: {info['permutation']}")
    logging.info(f"  Distance metric: {info['distance']:.6f}")

    # Visualizations
    if args.visualize:
        # Visualize palette extraction results
        extraction_vis_path = args.output.replace('.png', f'_{args.extraction_method}_source.png')
        swapper.palette_extractor.visualize_kmeans_result(
            source_image,
            info['source_palette'],
            info['source_weights'],
            save_path=extraction_vis_path
        )

        extraction_vis_path_target = args.output.replace('.png', f'_{args.extraction_method}_target.png')
        target_weights = swapper.palette_extractor.compute_weights(target_image, info['target_palette'])
        swapper.palette_extractor.visualize_kmeans_result(
            target_image,
            info['target_palette'],
            target_weights,
            save_path=extraction_vis_path_target
        )

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
            target_image,
            result_image,
            save_path=result_vis_path
        )

    # Visualize all permutations if requested
    if args.show_all_permutations and info.get('all_results'):
        logging.info("Creating permutation comparison visualization...")
        visualize_all_permutations(
            source_image,
            info['all_results'],
            info['permutation'],
            info['distance'],
            save_path=args.output.replace('.png', '_all_permutations.png')
        )

    logging.info("Done!")


if __name__ == '__main__':
    main()
