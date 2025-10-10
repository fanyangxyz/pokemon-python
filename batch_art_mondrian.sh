#!/bin/bash

# Batch art recoloring script - Map Mondrian to all Kelly paintings
# Processes mondrian.png with color palettes from kelly_*.png files

# Create output directory
mkdir -p batch-results-art

# Source image
SOURCE="images/art/mondrian.png"
SOURCE_NAME="mondrian"

# Get all Kelly images
ART_DIR="images/art"
KELLY_IMAGES=($(ls "$ART_DIR"/kelly_*.png | sort))

echo "Source: $SOURCE"
echo "Found ${#KELLY_IMAGES[@]} Kelly paintings"
echo "Will process ${#KELLY_IMAGES[@]} transformations"

# Counter for progress
TOTAL=${#KELLY_IMAGES[@]}
CURRENT=0

# Iterate through all Kelly paintings as targets
for target in "${KELLY_IMAGES[@]}"; do
    target_name=$(basename "$target" .png)
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "[$CURRENT/$TOTAL] Processing: $SOURCE_NAME -> $target_name"

    # Output filename
    OUTPUT_PREFIX="batch-results-art/${SOURCE_NAME}_to_${target_name}"

    # Skip if output already exists
    if [ -f "${OUTPUT_PREFIX}.png" ]; then
        echo "Skipping (already exists)"
        continue
    fi

    # Run recoloring with CPU, blind separation, and visualization
    python3 pokemon_recolor.py \
        --source "$SOURCE" \
        --target "$target" \
        --output "${OUTPUT_PREFIX}.png" \
        --extraction-method blind_separation \
        --device cpu \
        --visualize \
        --show-all-permutations \
        --num-colors 5 \
        --workers 16 \
        --hue-steps 4 \
        --sat-steps 2 \
        --val-steps 2

    echo "Completed: $SOURCE_NAME -> $target_name"
    echo "Saved to: ${OUTPUT_PREFIX}.png"
done

echo ""
echo "All done! Processed $TOTAL transformations"
echo "Results saved in batch-results-art/"
