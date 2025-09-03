# ======================= Anchored "App features" section (container cover) =======================
st.markdown("<div id='features'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="section">
  <div class="section-cover">
    <div class="eco-emoji">🧭</div>
    <div class="title">Let’s Start Sorting</div>
    <div class="badge">App features</div>
  </div>
  <div class="section-body">
""", unsafe_allow_html=True)

# City / Ward row (left: selector, right: badge)
c1, c2 = st.columns([2, 6])
with c1:
    city_label = st.selectbox("City / Ward", ["Shibuya (Tokyo)"], index=0)
with c2:
    st.markdown("<div class='citybadge badge-align'>More cities coming soon</div>", unsafe_allow_html=True)

city_id = CITY_MAP[city_label]
GUIDE = GUIDE_BY_CITY.get(city_id, {})

# Steps list
st.markdown("""
<ol class="howto">
  <li><strong>Select Upload image</strong> or open your <strong>Camera</strong>.</li>
  <li><strong>Detection runs</strong> and shows results.</li>
  <li>Follow the <strong>custom disposal instructions for your city</strong>.</li>
</ol>
""", unsafe_allow_html=True)

# ---- Inputs (kept inside the section) ----
src = st.radio("Input source", ["Upload image", "Camera"], index=0, horizontal=True)
auto_run = st.toggle("Auto-run detection", value=True, help="Run detection automatically after you choose or take a photo.")

image = None
if src == "Upload image":
    up = st.file_uploader("Choose an image", type=["jpg","jpeg","png"], accept_multiple_files=False,
                          help="Drag and drop file here")
    if not up:
        st.caption("No file chosen")
    else:
        image = Image.open(up).convert("RGB")
    st.caption("Limit 200MB per file • JPG, JPEG, PNG")
else:
    shot = st.camera_input("Open your camera", key="cam1")
    if shot:
        image = Image.open(shot).convert("RGB")

# ---- Advanced settings (still inside the section) ----
_REC_CONF=0.00; _REC_IOU=0.00; _REC_IMGSZ=200
_REC_BOTTLE=0.20; _REC_CAN=0.20; _REC_FOAM=0.20; _REC_AREA_PCT=0.20; _REC_TTA=True

conf=_REC_CONF; iou=_REC_IOU; imgsz=_REC_IMGSZ
bottle_min=_REC_BOTTLE; can_min=_REC_CAN; foam_min=_REC_FOAM
min_area_pct=_REC_AREA_PCT; tta=_REC_TTA

with st.expander("Advanced settings (optional)"):
    preset = st.radio("Preset", ["Minimum filters", "Recommended", "Strict"], index=1, horizontal=True)
    if preset == "Minimum filters":
        conf=0.05; iou=0.10; imgsz=IMGSZ_OPTIONS[0]
        bottle_min=0.00; can_min=0.00; foam_min=0.00; min_area_pct=0.0; tta=False
    elif preset == "Recommended":
        conf=_REC_CONF; iou=_REC_IOU; imgsz=_REC_IMGSZ
        bottle_min=_REC_BOTTLE; can_min=_REC_CAN; foam_min=_REC_FOAM; min_area_pct=_REC_AREA_PCT; tta=_REC_TTA
    elif preset == "Strict":
        conf=0.35; iou=0.50; imgsz=640
        bottle_min=0.70; can_min=0.70; foam_min=0.75; min_area_pct=0.5; tta=False

    conf = st.slider("Base confidence", 0.0, 0.95, float(conf), 0.01)
    iou  = st.slider("IoU",            0.0, 0.90, float(iou),  0.01)
    imgsz = int(st.select_slider("Inference image size", options=IMGSZ_OPTIONS, value=int(imgsz)))
    c1a, c2a, c3a, c4a = st.columns(4)
    bottle_min   = c1a.slider("Min conf: Bottle",    0.0, 1.0, float(bottle_min),   0.01)
    can_min      = c2a.slider("Min conf: Can",       0.0, 1.0, float(can_min),      0.01)
    foam_min     = c3a.slider("Min conf: Styrofoam", 0.0, 1.0, float(foam_min),     0.01)
    min_area_pct = c4a.slider("Min box area (%)",    0.0, 5.0,  float(min_area_pct), 0.1,
                              help="Ignore tiny boxes by percent of image area.")
    tta = st.toggle("Test time augmentation", value=tta, help="Slower. Sometimes reduces false positives.")

st.caption("Model loaded ✅")

# ---- Keep your detection UI/logic inside this same section ----
if image is not None:
    st.image(image, caption="Input", use_container_width=True)
    should_run = auto_run or st.button("Run detection")
    if should_run:
        dets, counts = run_detection(image)
        if dets:
            draw_and_show(image, dets)
            detected_labels = sorted({d["class_name"] for d in dets})
            guide_labels = [lbl for lbl in detected_labels if lbl in GUIDE]
            if guide_labels:
                st.subheader(f"Disposal instructions · {city_label}")
                for lbl in guide_labels:
                    show_guidance_card(lbl, counts.get(lbl, 0), GUIDE=GUIDE)
            else:
                st.caption("No local guidance to show for these detections.")
        else:
            st.info("All detections were filtered by thresholds. Try lowering per class thresholds or min box area.")

# Close the section
st.markdown("</div></div>", unsafe_allow_html=True)
