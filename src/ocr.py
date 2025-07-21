import cv2
import numpy as np
import pytesseract


# ================================================================================
# IMAGE LOADING AND DISPLAY
# ================================================================================
def show_image(image: cv2.typing.MatLike, title: str = "Image"):
    """Display an image in a window."""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1080, 1400)  # Resize to a reasonable size
    cv2.imshow(title, image)

    # Move the window to a known on-screen position
    cv2.moveWindow(title, 100, 100)
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


image = np.array(
    cv2.imread("../images/Screenshot_20250720_131146_Nonogram_galaxy.png")
)
crops_vertical = [300, 900]
crops_horizontal = [0, 0]
image = image[
    crops_vertical[0] : image.shape[0] - crops_vertical[1],
    crops_horizontal[0] : image.shape[1] - crops_horizontal[1],
]
print(f"Image shape after cropping: {image.shape}")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ================================================================================
# EDGE DETECION AND LINE DETECTION
# ================================================================================
low_threshold = 100
high_threshold = 150
edges = cv2.Canny(gray, low_threshold, high_threshold)
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=100,
    minLineLength=100,
    maxLineGap=10,
)
lines = np.squeeze(lines) if lines is not None else None
assert lines is not None, "No lines detected. Check image and parameters!"

# ================================================================================
# SEPARATE HORIZONTAL AND VERTICAL LINES
# ================================================================================

horizontal_lines = []
vertical_lines = []
for index in range(len(lines)):
    line = lines[index]
    assert line.shape == (4,), "Each line must have shape (4,)"
    x1, y1, x2, y2 = map(int, list(line))

    assert (
        isinstance(x1, int)
        and isinstance(y1, int)
        and isinstance(x2, int)
        and isinstance(y2, int)
    ), "Line coordinates must be integers"
    if x1 > x2:
        x1, x2 = x2, x1  # Ensure x1 is always less than x2
    if y1 > y2:
        y1, y2 = y2, y1  # Ensure y1 is always less than y2
    line = np.array([x1, y1, x2, y2])  # Convert to a 2D array for consistency

    # Calculate angle to determine if line is horizontal or vertical
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

    # Lines within 10 degrees of horizontal (0 or 180 degrees)
    if abs(angle) < 10 or abs(angle - 180) < 10:
        horizontal_lines.append(line)
    # Lines within 10 degrees of vertical (90 or -90 degrees)
    elif abs(angle - 90) < 10 or abs(angle + 90) < 10:
        vertical_lines.append(line)
horizontal_lines = np.array(horizontal_lines)
vertical_lines = np.array(vertical_lines)


def line_length(line):
    """Calculate the length of a line segment."""
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def filter_lines(lines, min_distance, by_index=0):
    """Filters horizontal lines if index is 1 and vertical lines if index is 0."""
    filtered_lines = []
    last_position = (
        -min_distance * 2
    )  # Initialize to a negative value to allow the first line
    for line in lines:
        position = line[by_index]  # Get x1 for vertical, y1 for horizontal
        if position - last_position >= min_distance:
            filtered_lines.append(line)
            last_position = position
        else:
            # if the new line is longer, than replace the last one
            if line_length(line) > line_length(filtered_lines[-1]):
                filtered_lines[-1] = line
    filtered_lines = np.array(filtered_lines)
    start_position = np.mean(filtered_lines[5:-5], axis=0, dtype=int)
    filtered_lines[:, 1 - by_index] = 0  # start_position[1-by_index]
    filtered_lines[:, 3 - by_index] = image.shape[
        by_index
    ]  # start_position[3-by_index]
    return filtered_lines


horizontal_lines = sorted(horizontal_lines, key=lambda x: x[1])  # Sort by y1
vertical_lines = sorted(vertical_lines, key=lambda x: x[0])  # Sort by x1
horizontal_lines = filter_lines(horizontal_lines, min_distance=10, by_index=1)
vertical_lines = filter_lines(vertical_lines, min_distance=10, by_index=0)
lines = np.concatenate((horizontal_lines, vertical_lines), axis=0)
print(f"Horizontal lines: {len(horizontal_lines)}")
print(f"Vertical lines: {len(vertical_lines)}")

# ================================================================================
# DISPLAY RESULTS
# ================================================================================
# if lines is not None:
#     print(f"Found {len(lines)} lines")
#     lines_image = image.copy()
#     for line in lines:
#         x1, y1, x2, y2 = line
#         # Draw line in green color with thickness 2
#         cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     # Display the image with lines
#     show_image(lines_image, "Detected Lines")

#     # Create separate visualizations for horizontal and vertical lines
#     horizontal_image = image.copy()
#     # Draw horizontal lines in red
#     for line in horizontal_lines:
#         x1, y1, x2, y2 = line
#         cv2.line(horizontal_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     show_image(horizontal_image, "Horizontal Lines (Red)")

#     vertical_image = image.copy()
#     # Draw vertical lines in blue
#     for line in vertical_lines:
#         x1, y1, x2, y2 = line
#         cv2.line(vertical_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     show_image(vertical_image, "Vertical Lines (Blue)")

# ================================================================================
# OCR PROCESSING
# ================================================================================

def threshold_image(gray, threshold=128):
    """Convert an image to binary using a fixed threshold."""
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

def extract_clue_regions(image, horizontal_lines, vertical_lines):
    """Extract clue regions from the nonogram image."""
    h, w = image.shape[:2]
    c_r = image[:, :, 2]
    c_g = image[:, :, 1]
    c_b = image[:, :, 0]

    # show_image(c_r, "Red Channel")
    # show_image(c_g, "Green Channel")
    # show_image(c_b, "Blue Channel")
    # diffs = {}
    # for i in range(3):
    #     for j in range(i):
    #         if i == j:
    #             continue
    #         diffs[(i, j)] = cv2.absdiff(image[:, :, i], image[:, :, j])

    # for (i, j), diff in diffs.items():
    #     show_image(diff, f"Difference between channel {i} and {j}")

    diff_b_g = cv2.absdiff(c_b, c_g)
    diff_g__b_g = cv2.absdiff(diff_b_g, c_g)
    show_image(diff_b_g, "Difference between Blue and Green Channels")
    show_image(diff_g__b_g, "Difference between Green and Blue-Green Channels")

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # show_image(gray, "Gray Image")
    
    # binary = threshold_image(gray, 128)
    # show_image(binary, "Binary Image")
    binary_small_nums = threshold_image(diff_g__b_g, 120)
    show_image(binary_small_nums, "Binary Small Numbers Image")
    binary_large_nums = threshold_image(diff_b_g, 80)
    show_image(binary_large_nums, "Binary Large Numbers Image")

    exit()
    # Sort lines by position
    horizontal_lines_sorted = sorted(
        horizontal_lines, key=lambda x: x[1]
    )  # Sort by y coordinate
    vertical_lines_sorted = sorted(
        vertical_lines, key=lambda x: x[0]
    )  # Sort by x coordinate

    # Find grid boundaries
    grid_top = (
        horizontal_lines_sorted[0][1] if len(horizontal_lines_sorted) > 0 else 0
    )
    grid_bottom = (
        horizontal_lines_sorted[-1][1]
        if len(horizontal_lines_sorted) > 0
        else h
    )
    grid_left = (
        vertical_lines_sorted[0][0] if len(vertical_lines_sorted) > 0 else 0
    )
    grid_right = (
        vertical_lines_sorted[-1][0] if len(vertical_lines_sorted) > 0 else w
    )

    print(
        f"Grid boundaries: top={grid_top}, bottom={grid_bottom}, left={grid_left}, right={grid_right}"
    )

    # Extract horizontal clue regions (left of the grid, between horizontal lines)
    horizontal_clue_regions = []
    for i in range(len(horizontal_lines_sorted) - 1):
        y1 = horizontal_lines_sorted[i][1]
        y2 = horizontal_lines_sorted[i + 1][1]

        # Crop the region left of the grid
        clue_region = binary[y1:y2, 0:grid_left]
        horizontal_clue_regions.append(
            {
                "region": clue_region,
                "row_index": i,
                "coordinates": (0, y1, grid_left, y2),
            }
        )

    # Extract vertical clue regions (above the grid, between vertical lines)
    vertical_clue_regions = []
    for i in range(len(vertical_lines_sorted) - 1):
        x1 = vertical_lines_sorted[i][0]
        x2 = vertical_lines_sorted[i + 1][0]

        # Crop the region above the grid
        clue_region = binary[0:grid_top, x1:x2]
        vertical_clue_regions.append(
            {
                "region": clue_region,
                "col_index": i,
                "coordinates": (x1, 0, x2, grid_top),
            }
        )

    return horizontal_clue_regions, vertical_clue_regions


def perform_ocr_on_regions(clue_regions, region_type="horizontal"):
    """Perform OCR on clue regions and return extracted text."""
    results = []

    for i, region_data in enumerate(clue_regions):
        region = region_data["region"]

        # Skip if region is too small
        if region.shape[0] < 10 or region.shape[1] < 10:
            print(
                f"Skipping {region_type} region {i}: too small ({region.shape})"
            )
            results.append("")
            continue
        
        show_image(
            region,
            f"{region_type.capitalize()} Clue {i} (before OCR)",
        )
        # Perform OCR with specific config for digits
        custom_config = (
            r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789 "
        )
        text = pytesseract.image_to_string(region, config=custom_config).strip()

        # Clean up the text (remove non-digit characters and extra spaces)
        cleaned_text = "".join(c for c in text if c.isdigit() or c.isspace())
        cleaned_text = " ".join(cleaned_text.split())  # Normalize whitespace

        results.append(cleaned_text)
        print(f"{region_type.capitalize()} clue {i}: '{cleaned_text}'")

        # Optional: Show the processed region for debugging
        if len(cleaned_text) > 0:  # Only show regions with detected text
            show_image(
                region,
                f"{region_type.capitalize()} Clue {i}: '{cleaned_text}'",
            )

    return results


# Extract clue regions
print("Extracting clue regions...")
horizontal_clue_regions, vertical_clue_regions = extract_clue_regions(
    image, horizontal_lines, vertical_lines
)

print(f"Found {len(horizontal_clue_regions)} horizontal clue regions")
print(f"Found {len(vertical_clue_regions)} vertical clue regions")

# for region in horizontal_clue_regions:
#     show_image(region["region"], f"Horizontal Clue {region['row_index']}")

# for region in vertical_clue_regions:
#     show_image(region["region"], f"Vertical Clue {region['col_index']}")

# Perform OCR on horizontal clues (row clues)
print("\nProcessing horizontal clues (row clues):")
horizontal_clues = perform_ocr_on_regions(
    horizontal_clue_regions[5:], "horizontal"
)

# Perform OCR on vertical clues (column clues)
print("\nProcessing vertical clues (column clues):")
vertical_clues = perform_ocr_on_regions(vertical_clue_regions, "vertical")

# Display results
print("\n" + "=" * 50)
print("OCR RESULTS:")
print("=" * 50)
print("Horizontal clues (rows):")
for i, clue in enumerate(horizontal_clues):
    print(f"  Row {i}: {clue}")

print("\nVertical clues (columns):")
for i, clue in enumerate(vertical_clues):
    print(f"  Col {i}: {clue}")

# Save results to a file
with open("nonogram_clues.txt", "w") as f:
    f.write("Nonogram Clues\n")
    f.write("=" * 20 + "\n\n")
    f.write("Horizontal clues (rows):\n")
    for i, clue in enumerate(horizontal_clues):
        f.write(f"Row {i}: {clue}\n")
    f.write("\nVertical clues (columns):\n")
    for i, clue in enumerate(vertical_clues):
        f.write(f"Col {i}: {clue}\n")

print(f"\nResults saved to 'nonogram_clues.txt'")
