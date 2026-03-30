import boto3
import os
from PIL import Image
import io
from dotenv import load_dotenv
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
REGION = os.getenv("REGION", "ap-south-1")

textract_client = boto3.client(
    service_name="textract",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)


# ─── Bounding Box Position Helper ────────────────────────────────────────────

def get_position_label(bounding_box: dict, 
                        h_zones: int = 3, 
                        v_zones: int = 3) -> str:
    """
    Maps a bounding box to a human-readable position label.

    Textract bounding box fields (all are 0.0–1.0 fractions of page size):
        Left   → X distance from left edge
        Top    → Y distance from top edge
        Width  → width of the box
        Height → height of the box

    Args:
        bounding_box : dict from Textract (Left, Top, Width, Height)
        h_zones      : number of horizontal zones (default 3 → Left/Center/Right)
        v_zones      : number of vertical zones   (default 3 → Top/Middle/Bottom)

    Returns:
        e.g. "Top-Left", "Center-Middle", "Bottom-Right"
    """
    left   = bounding_box["Left"]
    top    = bounding_box["Top"]
    width  = bounding_box["Width"]
    height = bounding_box["Height"]

    # Center point of the bounding box
    cx = left + width  / 2
    cy = top  + height / 2

    # ── Horizontal zone ──────────────────────────────────────────────────────
    h_labels = ["Left", "Center", "Right"]
    h_step   = 1.0 / h_zones
    h_idx    = min(int(cx / h_step), h_zones - 1)
    h_label  = h_labels[h_idx] if h_zones <= 3 else f"H-Zone-{h_idx}"

    # ── Vertical zone ────────────────────────────────────────────────────────
    v_labels = ["Top", "Middle", "Bottom"]
    v_step   = 1.0 / v_zones
    v_idx    = min(int(cy / v_step), v_zones - 1)
    v_label  = v_labels[v_idx] if v_zones <= 3 else f"V-Zone-{v_idx}"

    return f"{v_label}-{h_label}"


def get_pixel_coords(bounding_box: dict, 
                     img_width: int, 
                     img_height: int) -> dict:
    """
    Converts Textract's normalized (0–1) bounding box to actual pixel coords.

    Returns:
        {x1, y1, x2, y2, cx, cy}  — top-left, bottom-right, and center pixels
    """
    x1 = int(bounding_box["Left"]  * img_width)
    y1 = int(bounding_box["Top"]   * img_height)
    x2 = int((bounding_box["Left"] + bounding_box["Width"])  * img_width)
    y2 = int((bounding_box["Top"]  + bounding_box["Height"]) * img_height)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2}


# ─── Core Extraction ─────────────────────────────────────────────────────────

def extract_text_with_bounding_boxes(image_path: str) -> list[dict]:
    """
    Runs Textract on a local image file and returns every detected WORD
    with its text, confidence, bounding box (normalized + pixel), and
    position label on the page.

    Returns a list of dicts:
    [
        {
            "text"        : "Invoice",
            "confidence"  : 99.7,
            "bounding_box": {"Left": 0.1, "Top": 0.05, "Width": 0.2, "Height": 0.03},
            "pixels"      : {"x1": 80, "y1": 36, "x2": 240, "y2": 57, "cx": 160, "cy": 46},
            "position"    : "Top-Left",
            "page"        : 1
        },
        ...
    ]
    """
    # Read image bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Get image dimensions for pixel-coord conversion
    img = Image.open(io.BytesIO(image_bytes))
    img_width, img_height = img.size

    # Call Textract
    response = textract_client.detect_document_text(
        Document={"Bytes": image_bytes}
    )

    results = []

    for block in response.get("Blocks", []):
        # We only care about individual WORDs (LINE blocks are also available)
        if block["BlockType"] != "WORD":
            continue

        bb       = block["Geometry"]["BoundingBox"]
        pixels   = get_pixel_coords(bb, img_width, img_height)
        position = get_position_label(bb)

        results.append({
            "text"        : block["Text"],
            "confidence"  : round(block["Confidence"], 2),
            "bounding_box": bb,          # normalized 0–1
            "pixels"      : pixels,      # actual pixel coords
            "position"    : position,    # e.g. "Top-Left"
            "page"        : block.get("Page", 1),
        })

    return results


def extract_lines_with_bounding_boxes(image_path: str) -> list[dict]:
    """
    Same as above but groups words into LINE blocks.
    Useful when you need full sentences/phrases instead of individual words.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    img = Image.open(io.BytesIO(image_bytes))
    img_width, img_height = img.size

    response = textract_client.detect_document_text(
        Document={"Bytes": image_bytes}
    )

    results = []

    for block in response.get("Blocks", []):
        if block["BlockType"] != "LINE":
            continue

        bb       = block["Geometry"]["BoundingBox"]
        pixels   = get_pixel_coords(bb, img_width, img_height)
        position = get_position_label(bb)

        results.append({
            "text"        : block["Text"],
            "confidence"  : round(block["Confidence"], 2),
            "bounding_box": bb,
            "pixels"      : pixels,
            "position"    : position,
            "page"        : block.get("Page", 1),
        })

    return results


# ─── Search by Position ───────────────────────────────────────────────────────

def find_text_in_region(extracted: list[dict], 
                         region: str) -> list[dict]:
    """
    Filter extracted results to only those in a given position region.

    Args:
        extracted : output of extract_text_with_bounding_boxes()
        region    : e.g. "Top-Left", "Middle-Center", "Bottom-Right"

    Returns:
        Filtered list of matching text blocks.
    """
    return [item for item in extracted if item["position"] == region]


def find_text_in_bbox_range(extracted : list[dict],
                             left_min  : float = 0.0,
                             left_max  : float = 1.0,
                             top_min   : float = 0.0,
                             top_max   : float = 1.0) -> list[dict]:
    """
    Filter by raw normalized coordinate range — useful for pinpointing
    a specific column or row in a form/table.

    All values are 0.0–1.0 fractions of the page dimensions.
    """
    results = []
    for item in extracted:
        bb = item["bounding_box"]
        cx = bb["Left"] + bb["Width"]  / 2
        cy = bb["Top"]  + bb["Height"] / 2
        if left_min <= cx <= left_max and top_min <= cy <= top_max:
            results.append(item)
    return results


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    IMAGE_PATH = "sample_invoice.png"   # ← replace with your file

    print("=" * 60)
    print("WORD-LEVEL EXTRACTION")
    print("=" * 60)

    words = extract_text_with_bounding_boxes(IMAGE_PATH)

    for w in words:
        print(
            f"[{w['position']:20s}] "
            f"{w['text']:30s} "
            f"conf={w['confidence']:5.1f}%  "
            f"bbox=({w['bounding_box']['Left']:.3f}, {w['bounding_box']['Top']:.3f})  "
            f"pixels=({w['pixels']['x1']},{w['pixels']['y1']}) → "
            f"({w['pixels']['x2']},{w['pixels']['y2']})"
        )

    # ── Filter: only text in the Top-Left quadrant ────────────────────────────
    print("\n" + "=" * 60)
    print("TEXT FOUND IN 'Top-Left' REGION")
    print("=" * 60)
    top_left_items = find_text_in_region(words, "Top-Left")
    for item in top_left_items:
        print(f"  {item['text']}  (conf={item['confidence']}%)")

    # ── Filter: narrow column (e.g. right-side price column of an invoice) ────
    print("\n" + "=" * 60)
    print("TEXT IN RIGHT COLUMN (Left 0.7–1.0)")
    print("=" * 60)
    right_col = find_text_in_bbox_range(words, left_min=0.70, left_max=1.0)
    for item in right_col:
        print(f"  {item['text']}  @ {item['position']}")
