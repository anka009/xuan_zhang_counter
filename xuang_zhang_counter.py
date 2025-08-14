import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json
import os

PARAM_FILE = "mask_count_params.json"

# ---------------- Hilfsfunktionen ----------------
def save_params(params):
    with open(PARAM_FILE, "w") as f:
        json.dump(params, f, indent=2)

def load_params():
    if os.path.exists(PARAM_FILE):
        with open(PARAM_FILE, "r") as f:
            return json.load(f)
    return None

def cluster_points(points, max_dist):
    """Fasst Punkte zusammen, wenn sie näher als max_dist liegen."""
    clustered = []
    used = set()
    for i, p in enumerate(points):
        if i in used:
            continue
        cluster = [p]
        for j, q in enumerate(points):
            if j != i and j not in used:
                if np.linalg.norm(np.array(p) - np.array(q)) <= max_dist:
                    cluster.append(q)
                    used.add(j)
        used.add(i)
        cx = int(np.mean([pt[0] for pt in cluster]))
        cy = int(np.mean([pt[1] for pt in cluster]))
        clustered.append((cx, cy))
    return clustered

# ---------------- Streamlit ----------------
st.set_page_config(page_title="Maskenbasierter Fleckenzähler", layout="wide")
st.title("🎯 Maske → Organoidenzählung ")

saved_params = load_params()

uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif", "tiff"])
if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # ---------------- Maske erzeugen ----------------
    st.subheader("Maske erzeugen")
    clip_limit = st.slider("CLAHE Clip Limit", 1.0, 10.0, saved_params.get("clip_limit", 2.0) if saved_params else 2.0, 0.1)
    thresh_val = st.slider("Threshold-Wert", 0, 255, saved_params.get("threshold", 128) if saved_params else 128)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    _, mask = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY)
    if np.mean(gray[mask == 255]) > np.mean(gray[mask == 0]):
        mask = cv2.bitwise_not(mask)

    # ---------------- Fleckenzählung ----------------
    st.subheader("Fleckenzählung in Maske")
    min_size = st.slider("Mindestfläche (Pixel)", 10, 20000, saved_params.get("min_size", 1000) if saved_params else 1000, 10)
    mark_radius = st.slider("Anzeigeradius (Pixel)", 1, 200, saved_params.get("mark_radius", 8) if saved_params else 8, 1)
    line_thickness = st.slider("Linienstärke", 1, 50, saved_params.get("line_thickness", 2) if saved_params else 2, 1)
    cluster_dist = st.slider("Cluster-Radius (Pixel)", 1, 500, saved_params.get("cluster_dist", 20) if saved_params else 20, 1)
    mark_color = st.color_picker("Markierungsfarbe", saved_params.get("mark_color", "#ff0000") if saved_params else "#ff0000")
    bgr_color = tuple(int(mark_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))

    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_size]

    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    clustered_centers = cluster_points(centers, cluster_dist)

    # ---------------- Markierte Maske ----------------
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for (x, y) in clustered_centers:
        cv2.circle(mask_colored, (x, y), mark_radius, bgr_color, line_thickness)

    # ---------------- Anzeigeoptionen ----------------
    show_original_img = st.checkbox("Originalbild anzeigen", value=False)
    show_original_mask = st.checkbox("Original-Maske anzeigen", value=False)

    # Dynamische Anzeige
    cols = st.columns(sum([show_original_img, show_original_mask, True]))  # mindestens eine Spalte für markierte Maske
    idx = 0
    if show_original_img:
        cols[idx].image(image, caption="Originalbild", use_container_width=True)
        idx += 1
    if show_original_mask:
        cols[idx].image(mask, caption="Original-Maske", use_container_width=True)
        idx += 1
    cols[idx].image(mask_colored, caption=f"Markierte Maske – Gefundene Strukturen: {len(clustered_centers)}", use_container_width=True)

    # ---------------- CSV Download ----------------
    df = pd.DataFrame(clustered_centers, columns=["X", "Y"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="flecken.csv", mime="text/csv")

    # ---------------- Parameter speichern ----------------
    if st.button("💾 Aktuelle Einstellungen speichern"):
        params = {
            "clip_limit": clip_limit,
            "threshold": thresh_val,
            "min_size": min_size,
            "mark_radius": mark_radius,
            "line_thickness": line_thickness,
            "cluster_dist": cluster_dist,
            "mark_color": mark_color
        }
        save_params(params)
        st.success("✅ Einstellungen gespeichert!")
