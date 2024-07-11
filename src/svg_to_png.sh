#!/bin/bash

# Directory containing the SVG files
DIRECTORY="./"

# Loop through each SVG file in the directory
for FILE in "$DIRECTORY"/*.svg; do
    # Extract the base name without extension
    BASENAME=$(basename "$FILE" .svg)
    
    # Convert the SVG to PNG with the specified parameters
    inkscape "$FILE" --export-type=png --export-filename="$DIRECTORY/${BASENAME}.png" --export-dpi=600
done

