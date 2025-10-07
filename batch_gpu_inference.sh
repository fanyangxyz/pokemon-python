#!/bin/bash

# Batch Pokemon recoloring script using GPU with VGG features and Blind Color Separation
# Processes all pairs of Pokemon images in images/pokemons/

# Create output directory
mkdir -p batch-results

# Get all Pokemon images
POKEMON_DIR="images/pokemons"
IMAGES=($(ls "$POKEMON_DIR"/*.png))

echo "Found ${#IMAGES[@]} Pokemon images"
echo "Will process $((${#IMAGES[@]} * (${#IMAGES[@]} - 1))) pairs"

# Counter for progress
TOTAL_PAIRS=$((${#IMAGES[@]} * (${#IMAGES[@]} - 1)))
CURRENT=0

# Iterate through all pairs (source, target)
for source in "${IMAGES[@]}"; do
    source_name=$(basename "$source" .png)

    for target in "${IMAGES[@]}"; do
        # Skip if source == target
        if [ "$source" == "$target" ]; then
            continue
        fi

        target_name=$(basename "$target" .png)
        CURRENT=$((CURRENT + 1))

        echo ""
        echo "[$CURRENT/$TOTAL_PAIRS] Processing: $source_name -> $target_name"

        # Output filename
        OUTPUT_PREFIX="batch-results/${source_name}_to_${target_name}"

        # Skip if output already exists
        if [ -f "${OUTPUT_PREFIX}.png" ]; then
            echo "Skipping (already exists)"
            continue
        fi

        # Run recoloring with GPU, VGG features, blind separation, and visualization
        python3 pokemon_recolor.py \
            --source "$source" \
            --target "$target" \
            --output "${OUTPUT_PREFIX}.png" \
            --extraction-method blind_separation \
            --device cuda \
            --visualize \
            --show-all-permutations \
            --num-colors 5 \
            --workers 32 \
	    --hue-steps 4 \
	    --sat-steps 2 \
	    --val-steps 2

        echo "Completed: $source_name -> $target_name"
        echo "Saved to: ${OUTPUT_PREFIX}.png"
    done
done

echo ""
echo "All done! Processed $TOTAL_PAIRS pairs"
echo "Results saved in batch-results/"
