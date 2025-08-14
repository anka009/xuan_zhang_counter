import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Maskenbasierter Fleckenzähler", layout="wide")
st.title("🎯 Bild → Maske → Fleckenzählung")

# -------------------- 1. Bild laden --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif", "tiff"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    st.image(image, caption="Originalbild", use_container_width=True)

    # -------------------- 2. Maske erzeugen --------------------
    st.subheader("Maske erzeugen")
    clip_limit = st.slider("CLAHE Clip Limit", 1.0, 10.0, 2.0, 0.1)
    thresh_val = st.slider("Threshold-Wert", 0, 255, 128)

    # Kontrastverbesserung
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Thresholding
    _, mask = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY)

    # ggf. invertieren
    if np.mean(gray[mask == 255]) > np.mean(gray[mask == 0]):
        mask = cv2.bitwise_not(mask)

    st.image(mask, caption="Maske", use_container_width=True)

    # -------------------- 3. Flecken zählen --------------------
    st.subheader("Fleckenzählung")
    min_size = st.slider("Mindestfläche (Pixel)", 10, 20000, 1000, 10)
    radius = st.slider("Kreisradius", 2, 50, 8)
    line_thickness = st.slider("Liniendicke", 1, 10, 2)
    mark_color = st.color_picker("Markierungsfarbe", "#ff0000")
    bgr_color = tuple(int(mark_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))

    # Morphologische Reinigung
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Konturen finden
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_size]

    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    # Markiertes Bild
    marked = image.copy()
    for (x, y) in centers:
        cv2.circle(marked, (x, y), radius, bgr_color, line_thickness)

    st.image(marked, caption=f"Gefundene Flecken: {len(centers)}", use_container_width=True)

    # CSV Download
    df = pd.DataFrame(centers, columns=["X", "Y"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="flecken.csv", mime="text/csv")
