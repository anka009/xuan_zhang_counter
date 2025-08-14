import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json
import os

PARAM_DB = "zellkern_params.json"

# -------------------- Hilfsfunktionen --------------------
def load_param_db():
    if os.path.exists(PARAM_DB):
        with open(PARAM_DB, "r") as f:
            return json.load(f)
    return []

def save_param_db(db):
    with open(PARAM_DB, "w") as f:
        json.dump(db, f, indent=2)

def get_image_features(img_gray):
    return {
        "contrast": float(img_gray.std()),
        "mean_intensity": float(img_gray.mean()),
        "shape": img_gray.shape
    }

def find_best_params(features, db):
    if not db:
        return None
    best_match = None
    best_score = float("inf")
    for entry in db:
        score = abs(entry["features"]["contrast"] - features["contrast"]) \
              + abs(entry["features"]["mean_intensity"] - features["mean_intensity"]) \
              + abs(entry["features"]["shape"][0] - features["shape"][0]) / 1000 \
              + abs(entry["features"]["shape"][1] - features["shape"][1]) / 1000
        if score < best_score:
            best_score = score
            best_match = entry
    return best_match["params"] if best_match else None

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Interaktiver Zellkern-Zähler (lernend)", layout="wide")
st.title("🧬 Interaktiver Zellkern-Zähler – Stufe 1: Lernfähig")

uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif", "tiff"])
if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = get_image_features(gray)

    # -------------------- Parameter DB laden --------------------
    db = load_param_db()
    auto_params = find_best_params(features, db)

    # -------------------- Parameter Sidebar --------------------
    st.sidebar.header("⚙️ Parameter")
    min_size = st.sidebar.slider("Mindestfläche (Pixel)", 10, 20000, auto_params.get("min_size", 1000) if auto_params else 1000, 10)
    radius = st.sidebar.slider("Kreisradius Markierung", 2, 100, auto_params.get("radius", 8) if auto_params else 8)
    line_thickness = st.sidebar.slider("Liniendicke", 1, 30, auto_params.get("line_thickness", 2) if auto_params else 2)
    color = st.sidebar.color_picker("Farbe der Markierung", auto_params.get("color", "#ff0000") if auto_params else "#ff0000")

    use_manual_contrast = st.sidebar.checkbox("🔧 Manuellen Kontrast verwenden", value=False)
    use_manual_threshold = st.sidebar.checkbox("🔧 Manuellen Threshold verwenden", value=False)

    clip_limit = st.sidebar.slider(
        "CLAHE Clip Limit", 1.0, 10.0,
        auto_params.get("clip_limit", 2.0) if auto_params else 2.0, 0.1
    )

    manual_thresh = st.sidebar.slider(
        "Threshold-Wert", 0, 255,
        int(auto_params.get("threshold", 128)) if auto_params else 128
    )

    rgb_color = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    bgr_color = rgb_color[::-1]

    # -------------------- CLAHE --------------------
    contrast = gray.std()
    if use_manual_contrast:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    else:
        auto_clip = 4.0 if contrast < 40 else 2.0 if contrast < 80 else 1.5
        clahe = cv2.createCLAHE(clipLimit=auto_clip, tileGridSize=(8, 8))

    gray = clahe.apply(gray)

    # -------------------- Thresholding --------------------
    if use_manual_threshold:
        _, mask = cv2.threshold(gray, manual_thresh, 255, cv2.THRESH_BINARY)
        otsu_thresh = manual_thresh
    else:
        otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)

    if np.mean(gray[mask == 255]) > np.mean(gray[mask == 0]):
        mask = cv2.bitwise_not(mask)

    # -------------------- Morphologie --------------------
    kernel_size = max(3, min(image.shape[0], image.shape[1]) // 300)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # -------------------- Konturen --------------------
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_size]
    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    # -------------------- Ausgabe --------------------
    marked = image.copy()
    for (x, y) in centers:
        cv2.circle(marked, (x, y), radius, bgr_color, line_thickness)

    show_original = st.sidebar.checkbox("Originalbild anzeigen", value=True)

    if show_original:
        col1, col2 = st.columns(2)
        col1.image(image, caption="Original", use_container_width=True)
        col2.image(marked, caption=f"Gefundene Kerne: {len(centers)}", use_container_width=True)
    else:
        st.image(marked, caption=f"Gefundene Kerne: {len(centers)}", use_container_width=True)


    # -------------------- Speichern der Parameter --------------------
    if st.button("💾 Aktuelle Parameter als 'Bestes Ergebnis' speichern"):
        new_entry = {
            "features": features,
            "params": {
                "min_size": min_size,
                "radius": radius,
                "line_thickness": line_thickness,
                "color": color,
                "clip_limit": clip_limit if use_manual_contrast else auto_clip,
                "threshold": otsu_thresh
            }
        }
        db.append(new_entry)
        save_param_db(db)
        st.success("Parameter gespeichert – Programm wird daraus lernen!")

    # -------------------- CSV Download --------------------
    df = pd.DataFrame(centers, columns=["X", "Y"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")
