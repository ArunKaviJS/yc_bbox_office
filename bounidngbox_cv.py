import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import tempfile

# ============================================================
# Load TrOCR model and processor
# ============================================================

@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

processor, model = load_model()

# ============================================================
# Line segmentation (horizontal projection profile)
# ============================================================

def segment_lines_from_image(image: Image.Image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    dilated = cv2.dilate(closed, kernel_dilate, iterations=1)

    projection = np.sum(dilated, axis=1)
    height = dilated.shape[0]

    lines = []
    in_line = False
    start = 0

    for y in range(height):
        if projection[y] > 0:
            if not in_line:
                start = y
                in_line = True
        else:
            if in_line:
                end = y
                in_line = False
                if end - start > 10:
                    lines.append((start, end))

    if in_line:
        end = height
        if end - start > 10:
            lines.append((start, end))

    segmented_lines = []
    for (top, bottom) in lines:
        margin = 5
        y1 = max(0, top - margin)
        y2 = min(img.shape[0], bottom + margin)
        line_img = img[y1:y2, :]

        max_width = 1000
        h, w = line_img.shape[:2]
        if w > max_width:
            new_w = max_width
            new_h = int(h * (new_w / w))
            line_img = cv2.resize(line_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        segmented_lines.append(Image.fromarray(line_img))

    return segmented_lines

# ============================================================
# Word segmentation (vertical projection on each line)
# ============================================================

def segment_words_from_line(line_image: Image.Image):
    """
    Takes a single line image and splits it into individual word images
    using vertical projection profile.
    """
    img = np.array(line_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to merge letters within a word
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    dilated = cv2.dilate(binary, kernel_dilate, iterations=1)

    # Vertical projection — sum columns
    projection = np.sum(dilated, axis=0)
    width = dilated.shape[1]

    words = []
    in_word = False
    start = 0

    for x in range(width):
        if projection[x] > 0:
            if not in_word:
                start = x
                in_word = True
        else:
            if in_word:
                end = x
                in_word = False
                if end - start > 5:   # ignore tiny noise
                    words.append((start, end))

    if in_word:
        end = width
        if end - start > 5:
            words.append((start, end))

    word_images = []
    for (left, right) in words:
        margin = 4
        x1 = max(0, left - margin)
        x2 = min(img.shape[1], right + margin)
        word_img = img[:, x1:x2]

        # Minimum size check
        h, w = word_img.shape[:2]
        if w < 5 or h < 5:
            continue

        # Resize small word images so TrOCR handles them better
        if h < 32:
            scale = 32 / h
            word_img = cv2.resize(
                word_img,
                (int(w * scale), 32),
                interpolation=cv2.INTER_CUBIC
            )

        word_images.append(Image.fromarray(word_img))

    return word_images

# ============================================================
# OCR for a single image
# ============================================================

def run_ocr(image: Image.Image) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_length=128)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ============================================================
# Helper: update or insert a labeled entry
# ============================================================

def upsert_label(segment_id: str, text: str):
    for entry in st.session_state.labeled_data:
        if entry["segment_id"] == segment_id:
            entry["text"] = text
            return
    st.session_state.labeled_data.append({"segment_id": segment_id, "text": text})

# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(page_title="Handwritten OCR Labeling Tool", layout="wide")
st.title("📄 OCR & Labeling Tool")

# --- Mode selector ---
mode = st.radio(
    "🔍 Segmentation Mode",
    options=["Line by Line", "Word by Word"],
    horizontal=True,
    help="Line mode: OCR each full row. Word mode: OCR each word individually."
)

uploaded_files = st.file_uploader(
    "📤 Upload PDFs or Images",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    # --- Initialise session state ---
    for key, default in [
        ("file_index", 0),
        ("page_index", 0),
        ("labeled_data", []),
        ("ocr_cache", {}),
        ("pdf_images_cache", {}),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Guard against stale indices after re-upload
    st.session_state.file_index = min(st.session_state.file_index, len(uploaded_files) - 1)

    current_file = uploaded_files[st.session_state.file_index]
    file_name = os.path.splitext(current_file.name)[0]

    # --- Convert / cache pages ---
    if current_file.name not in st.session_state.pdf_images_cache:
        if current_file.type == "application/pdf":
            with st.spinner(f"📖 Converting PDF: {current_file.name}"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(current_file.read())
                    tmp_pdf_path = tmp_file.name
                images = convert_from_path(tmp_pdf_path, fmt="png", dpi=300)
        else:
            images = [Image.open(current_file).convert("RGB")]
        st.session_state.pdf_images_cache[current_file.name] = images

    pdf_images = st.session_state.pdf_images_cache[current_file.name]
    total_pages = len(pdf_images)

    st.session_state.page_index = min(st.session_state.page_index, total_pages - 1)
    current_page = st.session_state.page_index

    st.markdown(
        f"### 📄 File {st.session_state.file_index + 1} of {len(uploaded_files)}"
        f" — Page {current_page + 1} of {total_pages}"
    )

    page_img = pdf_images[current_page]
    line_images = segment_lines_from_image(page_img)
    st.info(f"✂️ {len(line_images)} lines found on Page {current_page + 1}")

    # -------------------------------------------------------
    # LINE BY LINE MODE
    # -------------------------------------------------------
    if mode == "Line by Line":
        for line_num, line_img in enumerate(line_images, start=1):
            seg_id = f"{file_name}_page{current_page + 1:03}_line{line_num:03}"

            with st.expander(f"📃 Line {line_num}", expanded=True):
                st.image(line_img, caption=seg_id, use_container_width=True)

                if seg_id not in st.session_state.ocr_cache:
                    with st.spinner("🔍 Running OCR…"):
                        st.session_state.ocr_cache[seg_id] = run_ocr(line_img)

                edited_text = st.text_area(
                    f"✏️ Edit OCR — {seg_id}",
                    value=st.session_state.ocr_cache[seg_id],
                    height=80,
                    key=seg_id,
                )
                upsert_label(seg_id, edited_text)

    # -------------------------------------------------------
    # WORD BY WORD MODE
    # -------------------------------------------------------
    else:
        for line_num, line_img in enumerate(line_images, start=1):
            st.markdown(f"---\n#### 📃 Line {line_num}")
            st.image(line_img, caption=f"Line {line_num} (full)", use_container_width=True)

            word_images = segment_words_from_line(line_img)
            st.caption(f"🔤 {len(word_images)} words detected in Line {line_num}")

            if not word_images:
                st.warning("No words detected in this line.")
                continue

            # Display words in a grid (4 columns)
            num_cols = 4
            rows = [word_images[i:i + num_cols] for i in range(0, len(word_images), num_cols)]

            for row_idx, row in enumerate(rows):
                cols = st.columns(num_cols)
                for col_idx, word_img in enumerate(row):
                    word_num = row_idx * num_cols + col_idx + 1
                    seg_id = (
                        f"{file_name}_page{current_page + 1:03}"
                        f"_line{line_num:03}_word{word_num:03}"
                    )

                    with cols[col_idx]:
                        st.image(word_img, caption=f"Word {word_num}", use_container_width=True)

                        if seg_id not in st.session_state.ocr_cache:
                            with st.spinner("🔍 OCR…"):
                                st.session_state.ocr_cache[seg_id] = run_ocr(word_img)

                        edited_text = st.text_input(
                            f"✏️ {seg_id}",
                            value=st.session_state.ocr_cache[seg_id],
                            key=seg_id,
                        )
                        upsert_label(seg_id, edited_text)

    # -------------------------------------------------------
    # Navigation & Export
    # -------------------------------------------------------
    st.markdown("---")
    col1, col2 = st.columns(2)

    if col1.button("➡️ Continue to Next Page"):
        if st.session_state.page_index + 1 < total_pages:
            st.session_state.page_index += 1
            st.rerun()
        elif st.session_state.file_index + 1 < len(uploaded_files):
            st.session_state.file_index += 1
            st.session_state.page_index = 0
            st.rerun()
        else:
            st.success("✅ All files and pages have been processed!")

            df = pd.DataFrame(st.session_state.labeled_data)
            csv_data = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Labeled CSV",
                csv_data,
                file_name="labeled_data.csv",
                mime="text/csv",
            )

            if st.button("🔁 Reset & Start Over"):
                for key in ["file_index", "page_index", "labeled_data", "ocr_cache", "pdf_images_cache"]:
                    st.session_state.pop(key, None)
                st.rerun()

    if col2.button("🔁 Reset All"):
        for key in ["file_index", "page_index", "labeled_data", "ocr_cache", "pdf_images_cache"]:
            st.session_state.pop(key, None)
        st.rerun()

else:
    st.info("👈 Upload your PDF or image files to begin.")