#!/bin/bash

# Create a montage for each source Pokemon
# Each montage contains all recolorings from that source to all targets

COMPARISON_FILES=($(ls batch-results/*_comparison.png 2>/dev/null | sort))

if [ ${#COMPARISON_FILES[@]} -eq 0 ]; then
    echo "No comparison files found in batch-results/"
    exit 1
fi

echo "Found ${#COMPARISON_FILES[@]} comparison images"

# Extract unique source Pokemon names
SOURCES=()
for file in "${COMPARISON_FILES[@]}"; do
    basename=$(basename "$file" _comparison.png)
    source=$(echo "$basename" | sed 's/_to_.*//')

    # Add to array if not already present
    if [[ ! " ${SOURCES[@]} " =~ " ${source} " ]]; then
        SOURCES+=("$source")
    fi
done

echo "Found ${#SOURCES[@]} source Pokemon"

# Create montage for each source
for source in "${SOURCES[@]}"; do
    echo ""
    echo "Creating montage for: $source"

    # Get all comparison files for this source
    SOURCE_FILES=($(ls batch-results/${source}_to_*_comparison.png 2>/dev/null | sort))

    if [ ${#SOURCE_FILES[@]} -eq 0 ]; then
        echo "  No files found for $source"
        continue
    fi

    echo "  Found ${#SOURCE_FILES[@]} transformations"

    OUTPUT_FILE="batch-results/${source}_all_transformations.png"

    # Stack vertically (1 column)
    montage "${SOURCE_FILES[@]}" \
        -tile 1x \
        -geometry +5+5 \
        -background white \
        -label '%f' \
        "$OUTPUT_FILE"

    echo "  Saved to: $OUTPUT_FILE"
done

echo ""
echo "All montages created!"
