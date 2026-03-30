from yc_bounding_box import extract_text_with_bounding_boxes
from extract_with_claude import claude_ocr_bedrock, extract_all_fields_and_tables
import json



# ── Field schema ─────────────────────────────────────────────────────────────
# ── Field schema ─────────────────────────────────────────────────────────────
fields = [
    {
        "fieldType": "field",
        "fieldName": "Passport Name",
        "fieldDatatype": "String",
        "fieldDescription": "Extract Passport Given Name",
        "fieldExample": "",
    },
    {
        "fieldType": "field",
        "fieldName": "PassPort Number",
        "fieldDatatype": "String",
        "fieldDescription": "Extract Passport Number",
        "fieldExample": "",
    },
    {
        "fieldType": "field",
        "fieldName": "Surname ",
        "fieldDatatype": "String",
        "fieldDescription": "Extract Surname",
        "fieldExample": "",
    },
    {
        "fieldType": "field",
        "fieldName": "Nationality",
        "fieldDatatype": "String",
        "fieldDescription": "Extract Nationality",
        "fieldExample": "",
    },
    
    
]


def map_fields_to_bboxes(final_result, words):
    mapped_output = {}

    for field, value in final_result.items():
        if not value:
            mapped_output[field] = None
            continue

        value_clean = value.strip().upper()

        matches = []
        for w in words:
            word_text = w["text"].strip().upper()

            # Exact or partial match
            if value_clean == word_text or value_clean in word_text:
                matches.append(w)

        if not matches:
            mapped_output[field] = {
                "value": value,
                "bounding_boxes": []
            }
            continue

        if len(matches) == 1:
            # ✅ Single match — use its bounding box data directly as-is
            m = matches[0]
            mapped_output[field] = {
                "value": value,
                "bounding_box": m["bounding_box"],
                "pixels": m["pixels"],
                "position": m.get("position"),
                "page": m.get("page"),
                "confidence": m["confidence"]
            }
        else:
            # ✅ Multiple matches — merge pixel coords, recalculate normalized bbox
            # Use the first match as a reference for image dimensions
            ref = matches[0]

            x1 = min(m["pixels"]["x1"] for m in matches)
            y1 = min(m["pixels"]["y1"] for m in matches)
            x2 = max(m["pixels"]["x2"] for m in matches)
            y2 = max(m["pixels"]["y2"] for m in matches)

            # Estimate image dimensions from first match's normalized + pixel values
            img_width  = ref["pixels"]["x2"] / ref["bounding_box"]["Width"]  \
                         * ref["bounding_box"]["Width"] + ref["pixels"]["x1"]  \
                         # simplified: use ratio from ref
            img_width  = (ref["pixels"]["x2"] - ref["pixels"]["x1"]) / ref["bounding_box"]["Width"]
            img_height = (ref["pixels"]["y2"] - ref["pixels"]["y1"]) / ref["bounding_box"]["Height"]

            merged_width  = (x2 - x1) / img_width
            merged_height = (y2 - y1) / img_height
            merged_left   = x1 / img_width
            merged_top    = y1 / img_height
            merged_cx     = (x1 + x2) // 2
            merged_cy     = (y1 + y2) // 2

            mapped_output[field] = {
                "value": value,
                "bounding_box": {
                    "Width":  merged_width,
                    "Height": merged_height,
                    "Left":   merged_left,
                    "Top":    merged_top
                },
                "pixels": {
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "cx": merged_cx, "cy": merged_cy
                },
                "position": ref.get("position"),
                "page": ref.get("page"),
                "confidence": max(m["confidence"] for m in matches)
            }

    return mapped_output




IMAGE_PATH = "20260224063720449_1640.jpeg"

print("=" * 60)
print("WORD-LEVEL EXTRACTION")
print("=" * 60)

words = extract_text_with_bounding_boxes(IMAGE_PATH)
print(words)

#======================================================================


file_path = "20260224063720449_1640.jpeg"
print("🔍 Running OCR via Bedrock Claude...")
ocr_text = claude_ocr_bedrock(file_path)
print("OCR result:\n", ocr_text)

print("\n📊 Extracting structured fields...")
final_result = extract_all_fields_and_tables(fields, ocr_text)
print("Final result:\n", json.dumps(final_result, indent=2))

print("\n📍 Mapping fields to bounding boxes...")
mapped = map_fields_to_bboxes(final_result, words)
print("Final mapped result:\n", json.dumps(mapped, indent=2))