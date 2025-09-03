import os
import shutil
import hashlib
import time
import base64
from urllib.parse import urlparse

import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import cv2

st.set_page_config(page_title="When AI Sees Litter · Shibuya", page_icon="♻️", layout="wide")

# ======================= Helpers =======================
def data_url(path: str, fallback: str | None = None) -> str:
    """Return a data: URL for a local image if it exists, else the fallback URL (or empty string)."""
    try:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            ext = (os.path.splitext(path)[1].lstrip(".") or "png").lower()
            return f"data:image/{ext};base64,{b64}"
    except Exception:
        pass
    return fallback or ""

def _domain_label(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "link"

# ======================= THEME / NAV / HERO =======================
def apply_theme():
    nav_logo = data_url("logo.png", None)  # None -> hide if missing
    nav_brand_img = f'<img class="brand-logo" src="{nav_logo}" alt="logo"/>' if nav_logo else ""

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;700;900&display=swap');

    :root{{
      /* brand */
      --pri:#79C16D; --pri2:#4FA25A; --hi:#CFEAC0; --bg:#FAFEF6; --card:#FFFFFF;
      --txt:#0F2A1C; --mut:#6F8B7A; --pill:#EEF7E9; --bd:#E5EFE3;
      /* hero */
      --bg2:#f5f7f2; --hero:#6f8f2b; --hero2:#6b8a2c;
    }}
    html, body, [data-testid="stAppViewContainer"]{{
      background:var(--bg2); color:var(--txt); font-family:'Poppins',ui-sans-serif;
    }}
    .main .block-container{{ padding-top:0.6rem !important; max-width:1120px; }}
    html{{ scroll-behavior:smooth; }}

    /* --- navbar --- */
    .nav{{
      position:sticky; top:0; z-index:50; display:flex; align-items:center; justify-content:space-between;
      padding:14px 8px; background:rgba(245,247,242,.9);
      -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);
      border-bottom:1px solid var(--bd);
    }}
    .brand{{ display:flex; align-items:center; gap:10px; font-weight:900; letter-spacing:-.02em; }}
    .brand-logo{{
      height:22px; width:22px; object-fit:contain; border-radius:6px; background:#fff; padding:2px;
      box-shadow:0 0 0 1px rgba(0,0,0,.06) inset;
    }}
    .links{{ display:flex; gap:18px; font-weight:700; }}
    .links a{{ color:inherit !important; text-decoration:none; }}
    .links a:hover{{ text-decoration:underline; }}

    /* --- hero --- */
    .hero{{
      margin:16px 0 24px; padding:28px; border-radius:28px; color:#fff;
      background:linear-gradient(180deg,var(--hero),var(--hero2));
      box-shadow:0 20px 60px rgba(70,90,30,.22);
    }}
    .hero-grid{{ display:grid; grid-template-columns:1.25fr 1fr; gap:24px; align-items:center; }}
    .hero h1{{ font-size:clamp(28px,5vw,62px); line-height:1.05; margin:0 0 8px; font-weight:900; }}
    .sub{{ opacity:.95; font-size:clamp(14px,1.1vw,18px); max-width:560px; }}
    .rule{{ height:1px; background:rgba(255,255,255,.28); margin:14px 0; }}
    .chips{{ display:flex; gap:8px; flex-wrap:wrap; }}
    .chip{{ background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28);
           padding:6px 10px; border-radius:999px; font-weight:700; }}
    .cta{{ display:inline-block; margin-top:16px; padding:12px 18px; border-radius:999px; font-weight:900;
          background:#0e0e0e; color:#fff !important; text-decoration:none; box-shadow:0 10px 24px rgba(0,0,0,.25); }}

    /* hero photo: white circle for transparent logos */
    .photo{{ position:relative; height:320px; }}
    .photo .circle{{
      position:absolute; right:-6%; top:0; height:100%; aspect-ratio:1/1;
      background:#fff !important; border-radius:50%;
      border:10px solid #fff; box-shadow:0 10px 50px rgba(0,0,0,.35);
      display:flex; align-items:center; justify-content:center; overflow:hidden;
    }}
    .photo .circle img{{ width:86%; height:86%; object-fit:contain; background:#fff !important; border:none; }}
    @media (max-width: 900px){{
      .hero-grid{{ grid-template-columns:1fr; }}
      .photo{{ height:220px; }}
      .photo .circle{{ right:auto; left:50%; transform:translateX(-50%); }}
    }}

    /* ===== Section container covers ===== */
    .section{{
      background:var(--card);
      border:1px solid var(--bd);
      border-radius:22px;
      overflow:hidden;
      box-shadow:0 6px 24px rgba(0,0,0,.06);
      margin:14px 0 28px;
    }}
    .section-cover{{
      display:flex; align-items:center; gap:10px;
      padding:14px 18px;
      background:linear-gradient(90deg, var(--pri2), var(--pri));
      color:#fff;
    }}
    .section-cover .eco-emoji{{ font-size:1.2rem; }}
    .section-cover .title{{ font-weight:900; font-size:1.08rem; letter-spacing:-.01em; }}
    .section-cover .badge{{
      margin-left:auto; background:rgba(255,255,255,.18);
      border:1px solid rgba(255,255,255,.28);
      padding:4px 10px; border-radius:999px; font-size:.85rem;
    }}
    .section-body{{ padding:16px 18px; }}

    /* --- original eco UI bits --- */
    .pill{{ display:inline-block; background:var(--pill); padding:2px 10px 4px 10px;
           border-radius:999px; color:var(--pri2); border:1px solid var(--bd); }}
    .eco-links{{ display:flex; gap:10px; margin-top:10px; margin-bottom:22px; flex-wrap:wrap; }}
    .eco-link{{ border-radius:999px; padding:8px 12px; border:1px solid var(--bd);
               background:#fff; text-decoration:none !important; color:var(--pri2) !important; font-weight:700; }}
    .eco-link:hover{{ background:var(--pill); }}
    .citybadge{{ display:inline-block; background:var(--pill); padding:4px 10px;
                border-radius:999px; border:1px solid var(--bd); color:var(--pri2); }}
    .badge-align{{ margin-top:28px; }} @media (max-width:640px){{ .badge-align{{ margin-top:8px; }} }}

    .eco-card{{ background:#fff; border:none; border-radius:22px; padding:18px 16px;
               margin:10px 0 18px 0; box-shadow:0 3px 16px rgba(0,0,0,.04); }}
    .eco-head{{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }}
    .eco-emoji{{ font-size:1.5rem; }}
    .eco-title{{ font-weight:900; font-size:1.28rem; }}
    .eco-badge{{ margin-left:auto; background:var(--pill); color:var(--pri2);
                border:1px solid var(--bd); border-radius:999px; padding:4px 10px; font-size:.85rem; }}

    .eco-section-title-primary{{ font-weight:900; font-size:1.12rem; color:var(--pri2); margin:8px 0 6px 0; }}
    .eco-section-title{{ font-weight:800; margin:8px 0 4px 0; }}
    .eco-list{{ margin:0 0 4px 0; padding-left:18px; }}
    .eco-list li{{ margin:2px 0; }}
    .chip-row{{ display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 2px 0; }}
    .chip{{ background:var(--pill); color:var(--pri2); border:1px solid var(--bd);
           border-radius:999px; padding:4px 10px; font-size:.88rem; }}

    .sdg-caption{{ text-align:center; font-weight:800; margin-top:10px; }}

    /* remove default separators */
    [data-testid="stDivider"], hr, [role="separator"]{{ display:none !important; }}
    [data-testid="stExpander"] details, [data-testid="stExpander"] summary{{
      border:none !important; box-shadow:none !important; background:transparent !important;
    }}
    [data-testid="stHorizontalBlock"], [data-testid="stVerticalBlock"]{{
      border:none !important; box-shadow:none !important; background:transparent !important;
    }}
    [data-testid="stHeader"]{{ background:transparent !important; }}
    [data-testid="stHeader"] div{{ border:none !important; box-shadow:none !important; }}
    </style>

    <div class="nav">
      <div class="brand">{nav_brand_img}<span>When AI Sees Litter</span></div>
      <div class="links">
        <a href="#features">App features</a>
        <a href="#sdgs">Impact &amp; SDGs</a>
        <a href="#about">About us</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

apply_theme()

# ======================= Hero =======================
hero_img = data_url(
    "logo.png",
    "https://images.unsplash.com/photo-1501004318641-b39e6451bec6?q=80&w=1080&auto=format&fit=crop"
)
st.markdown(f"""
<div class="hero">
  <div class="hero-grid">
    <div>
      <h1>Let's Start Sorting</h1>
      <div class="sub">Point your camera at any item and get instant, city-specific disposal guidance across Japan.</div>
      <div class="rule"></div>
      <div class="chips">
        <span class="chip">Recycling</span><span class="chip">Disposal guide</span><span class="chip">AI</span>
      </div>
      <a class="cta" href="#features">Start now</a>
    </div>
    <div class="photo"><div class="circle"><img src="{hero_img}" alt="hero"/></div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ======================= Config & Model =======================
MODEL_URL   = os.getenv("MODEL_URL", "https://raw.githubusercontent.com/Bellzum/streamlit-main/main/new_taco1.pt")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "best.pt")

CACHED_DIR  = "/tmp/models"
def _hash_url(u: str) -> str: return hashlib.sha1(u.encode("utf-8")).hexdigest()[:12]
CACHED_PATH = os.path.join(CACHED_DIR, f"weights_{_hash_url(MODEL_URL)}.pt")

IMGSZ_OPTIONS = [200, 320, 416, 512, 640, 800, 960, 1280]

FORCE_CLASS_NAMES = True
TARGET_NAMES = ["Clear plastic bottle", "Drink can", "Styrofoam piece"]

# Official references & images
SHIBUYA_GUIDE_URL    = "https://www.city.shibuya.tokyo.jp/contents/living-in-shibuya/en/daily/garbage.html"
SHIBUYA_POSTER_EN    = "https://files.city.shibuya.tokyo.jp/assets/12995aba8b194961be709ba879857f70/bfda2f5d763343b5a0b454087299d57f/2024wakedashiEnglish.pdf#page=2"
SHIBUYA_PLASTICS_NOTICE = "https://files.city.shibuya.tokyo.jp/assets/12995aba8b194961be709ba879857f70/0cdf099fdfe8456fbac12bb5ad7927e4/assets_kusei_ShibuyaCityNews2206_e.pdf#page=1"
FUKUOKA_PET_STEPS = [
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph04.png",
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph05.png",
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph06.png",
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph07.png",
]
ICON_PET   = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Recycling_pet.svg/120px-Recycling_pet.svg.png"
ICON_AL    = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Recycling_alumi.svg/120px-Recycling_alumi.svg.png"
ICON_STEEL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Recycling_steel.svg/120px-Recycling_steel.svg.png"
ICON_PLA   = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Recycling_pla.svg/120px-Recycling_pla.svg.png"
LINK_UN_CNP  = "https://unfccc.int/climate-action/united-nations-carbon-offset-platform"
LINK_UN_CNP2 = "https://offset.climateneutralnow.org/"
LINK_WB_MRV  = "https://www.worldbank.org/en/news/feature/2022/07/27/what-you-need-to-know-about-the-measurement-reporting-and-verification-mrv-of-carbon-credits"
LINK_GS      = "https://www.goldstandard.org/"
LINK_VERRA   = "https://verra.org/programs/verified-carbon-standard/"
LINK_JCREDIT = "https://japancredit.go.jp/english/"
HANWA_CAN2CAN = "https://www.hanwa.co.jp/images/csr/business/img_5_01.png"
CCBJI_CAN2CAN = "https://en.ccbji.co.jp/upload/images/20221222-1-1(5).jpg"

# ======================= Guidance content (Shibuya) =======================
GUIDE_SHIBUYA = {
    "Clear plastic bottle": {
        "title": "Shibuya disposal: PET bottle",
        "emoji": "🧴",
        "materials": "Bottle body is PET. Caps and labels are PP or PE.",
        "why_separate": [
            "Caps and labels contaminate the PET stream if left on.",
            "Shibuya asks you to remove caps and labels and sort them with Plastics."
        ],
        "steps": [
            "Remove the cap and label.",
            "Rinse the bottle.",
            "Crush it flat.",
            "Put PET bottles in a transparent bag for PET.",
            "Put caps and labels with Plastics."
        ],
        "recycles_to": ["New PET bottles", "Fibers for clothing and bags", "Sheets and films"],
        "facts": [
            {"text": "Japan’s reported plastic recycling rate includes thermal recovery. Clean PET enables high value bottle to bottle.",
             "url": "https://japan-forward.com/japans-plastic-recycling-the-unseen-reality/"},
        ],
        "images": FUKUOKA_PET_STEPS,
        "icons": [ICON_PET],
        "link": SHIBUYA_GUIDE_URL,
        "poster": SHIBUYA_POSTER_EN,
    },
    "Drink can": {
        "title": "Shibuya disposal: Aluminum or steel can",
        "emoji": "🥫",
        "materials": None,
        "why_separate": [
            "Clean cans keep a high value recycling stream.",
            "Aluminum recycling saves major energy compared with producing new metal."
        ],
        "steps": [
            "Rinse the can.",
            "Optional: Lightly crush or squeeze to save space if your building allows.",
            "Put cans in a transparent bag for cans."
        ],
        "recycles_to": [
            "New beverage cans",
            "Automotive and construction parts",
            "Remelt scrap ingots"
        ],
        "facts": [
            {"text": "Coca Cola Bottlers Japan promotes can to can with recycled aluminum bodies.",
             "url": "https://en.ccbji.co.jp/news/detail.php?id=1347"},
            {"text": "Hanwa explains used aluminum cans are cleaned, melted and supplied as remelt scrap ingots then used again as cans.",
             "url": HANWA_CAN2CAN},
        ],
        "images": [HANWA_CAN2CAN, CCBJI_CAN2CAN],
        "icons": [ICON_AL, ICON_STEEL],
        "link": SHIBUYA_GUIDE_URL,
        "poster": SHIBUYA_POSTER_EN,
    },
    "Styrofoam piece": {
        "title": "Shibuya disposal: Styrofoam piece",
        "emoji": "🧊",
        "materials": "Expanded polystyrene foam.",
        "why_separate": [
            "Clean pieces can go to Plastic items when marked as packaging.",
            "Keeping plastics clean improves material recovery quality."
        ],
        "steps": [
            "Remove food residue and wipe or quick rinse if needed.",
            "Break large pieces down to fit bags.",
            "Put Styrofoam with Plastic items in a clear or semi clear bag following building day."
        ],
        "recycles_to": ["Foam trays and molded parts", "Pellets for plastic goods", "Sometimes thermal recovery"],
        "facts": [
            {"text": "Plastic sorting rules vary by municipality. See Shibuya’s plastics notice for details.",
             "url": SHIBUYA_PLASTICS_NOTICE},
        ],
        "images": ["https://www.fpco.jp/dcms_media/image/appeal_img01_b.jpg"],
        "icons": [ICON_PLA],
        "link": SHIBUYA_GUIDE_URL,
        "poster": SHIBUYA_PLASTICS_NOTICE,
    },
}
GUIDE_BY_CITY = {"shibuya": GUIDE_SHIBUYA}
CITY_MAP = {"Shibuya (Tokyo)": "shibuya"}

# ======================= Download / load model =======================
def _download_file(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        import requests
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for ch in r.iter_content(chunk_size=8192):
                    if ch: f.write(ch)
    except Exception as e_req:
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
        except Exception as e_url:
            st.error(
                f"Failed to download model from URL:\n{url}\n\n"
                f"requests error: {e_req}\nurllib error: {e_url}\n\n"
                "If private or rate limited, make the file public or commit it to this repo."
            )
            st.stop()

def _ensure_model_path() -> str:
    if MODEL_URL.strip().startswith("http"):
        CACHED = os.path.exists(CACHED_PATH)
        if not CACHED:
            _download_file(MODEL_URL.strip(), CACHED_PATH)
        return CACHED_PATH
    if not os.path.exists(LOCAL_MODEL):
        st.error("Model file not found. Provide MODEL_URL or place best.pt next to this file.")
        st.stop()
    return LOCAL_MODEL

def _cache_key_for(path: str) -> str:
    try:
        return f"{path}:{os.path.getmtime(path)}:{os.path.getsize(path)}"
    except Exception:
        return path

@st.cache_resource(show_spinner=True)
def _load_model_cached(path: str, key: str):
    m = YOLO(path)
    if FORCE_CLASS_NAMES:
        try:
            m.names = {i: n for i, n in enumerate(TARGET_NAMES)}
        except Exception:
            pass
    return m

def load_model():
    path = _ensure_model_path()
    return _load_model_cached(path, _cache_key_for(path))

GLOBAL_MODEL = load_model()

# ======================= Guidance rendering helpers =======================
def _get_names_map(pred, model):
    if FORCE_CLASS_NAMES:
        return {i: n for i, n in enumerate(TARGET_NAMES)}
    if hasattr(pred, "names") and isinstance(pred.names, dict):  return pred.names
    if hasattr(model, "names") and isinstance(model.names, dict): return model.names
    if hasattr(model, "names") and isinstance(model.names, list): return {i:n for i,n in enumerate(model.names)}
    return {0:"Clear plastic bottle", 1:"Drink can", 2:"Styrofoam piece"}

def _guide_link(url: str, label: str):
    st.markdown(f'<a class="eco-link" href="{url}" target="_blank" rel="noopener">{label}</a>', unsafe_allow_html=True)

def _guidance_text(info: dict):
    st.markdown('<div class="eco-section-title-primary">How to put out</div>', unsafe_allow_html=True)
    st.markdown('<ul class="eco-list">', unsafe_allow_html=True)
    for step in info["steps"]:
        st.markdown(f'<li>{step}</li>', unsafe_allow_html=True)
    st.markdown('</ul>', unsafe_allow_html=True)

    if info.get("why_separate"):
        st.markdown('<div class="eco-section-title">How to manage</div>', unsafe_allow_html=True)
        st.markdown('<ul class="eco-list">', unsafe_allow_html=True)
        for reason in info["why_separate"]:
            st.markdown(f'<li>{reason}</li>', unsafe_allow_html=True)
        st.markdown('</ul>', unsafe_allow_html=True)

    if info.get("recycles_to"):
        st.markdown('<div class="eco-section-title">Commonly recycled into</div>', unsafe_allow_html=True)
        st.markdown('<div class="chip-row">', unsafe_allow_html=True)
        for item in info["recycles_to"]:
            st.markdown(f'<div class="chip">{item}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    facts = info.get("facts", [])
    if facts:
        st.markdown('<div class="eco-section-title">Did you know?</div>', unsafe_allow_html=True)
        st.markdown('<ul class="eco-list">', unsafe_allow_html=True)
        for fact in facts:
            st.markdown(f'<li>{fact["text"]}</li>', unsafe_allow_html=True)
        st.markdown('</ul>', unsafe_allow_html=True)
        st.markdown('<div class="eco-links">', unsafe_allow_html=True)
        for fact in facts:
            dom = _domain_label(fact["url"])
            _guide_link(fact["url"], f"Learn more · {dom}")
        st.markdown('</div>', unsafe_allow_html=True)

def show_guidance_card(label: str, count: int = 0, GUIDE=None):
    info = GUIDE.get(label) if GUIDE else None
    if not info: return
    st.markdown('<div class="eco-card">', unsafe_allow_html=True)
    st.markdown(f"""
      <div class="eco-head">
        <div class="eco-emoji">{info['emoji']}</div>
        <div class="eco-title">{info['title']}</div>
        <div class="eco-badge">Detected: {count}</div>
      </div>
    """, unsafe_allow_html=True)
    if info.get("icons"):
        st.image(info["icons"], width=48, caption=[""]*len(info["icons"]))
    imgs = info.get("images") or []
    if imgs:
        left, right = st.columns([1, 2])
        with left:
            if len(imgs) == 1:
                st.image(imgs[0], use_container_width=True)
            elif len(imgs) <= 3:
                for im in imgs: st.image(im, use_container_width=True)
            else:
                st.image(imgs, width=160, caption=[""]*len(imgs))
        with right:
            _guidance_text(info)
    else:
        _guidance_text(info)
    st.markdown('<div class="eco-links">', unsafe_allow_html=True)
    if info.get("poster"):
        _guide_link(info["poster"], f"📄 Poster (PDF) · {_domain_label(info['poster'])}")
    _guide_link(info["link"], f"🌐 Official guidance · {_domain_label(info['link'])}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================= Detection helpers =======================
def run_detection(image_pil: Image.Image, conf, iou, imgsz, tta, bottle_min, can_min, foam_min, min_area_pct):
    model = GLOBAL_MODEL
    bgr = np.array(image_pil.convert("RGB"))[:, :, ::-1]
    results = model.predict(bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False, augment=tta)
    pred = results[0]
    if pred.boxes is None or len(pred.boxes) == 0:
        st.info("No detections")
        return [], {}
    boxes = pred.boxes.xyxy.cpu().numpy()
    scores = pred.boxes.conf.cpu().numpy()
    clsi   = pred.boxes.cls.cpu().numpy().astype(int)

    names_map = {i:n for i,n in enumerate(TARGET_NAMES)} if FORCE_CLASS_NAMES else _get_names_map(pred, model)
    per_class_min = {
        "Clear plastic bottle": bottle_min,
        "Drink can":            can_min,
        "Styrofoam piece":      foam_min,
    }

    H, W = bgr.shape[:2]
    min_area = (min_area_pct / 100.0) * (H * W)

    dets, counts = [], {}
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].tolist()
        w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
        area = w * h
        c = int(clsi[i])
        name = names_map.get(c, str(c))
        s = float(scores[i])
        if s < per_class_min.get(name, conf): continue
        if area < min_area: continue
        dets.append({"xyxy":[x1,y1,x2,y2], "class_id":c, "class_name":name, "score":s})
        counts[name] = counts.get(name, 0) + 1
    return dets, counts

# Color levels (no level text in label)
def _level_for(s: float) -> str:
    if s >= 0.80: return "High"
    if s >= 0.60: return "Moderate"
    if s >= 0.40: return "Low"
    return "Very Low"

def _level_color(level: str) -> tuple[int,int,int]:
    return {
        "High": (35,110,65),
        "Moderate": (70,150,100),
        "Low": (130,190,150),
        "Very Low": (195,225,205),
    }.get(level, (28,160,78))

def draw_and_show(image_pil: Image.Image, dets):
    bgr = np.array(image_pil.convert("RGB"))[:, :, ::-1]
    out = bgr.copy()
    H, W = out.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        lvl = _level_for(float(d["score"]))          # for color only
        color = _level_color(lvl)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f'{d["class_name"]} {d["score"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = y1 - 4
        if y_text - th - 4 < 0: y_text = min(y1 + th + 6, H - 2)
        x_text = max(0, min(x1, W - tw - 6))
        cv2.rectangle(out, (x_text, max(0, y_text - th - 4)),
                           (min(x_text + tw + 6, W - 1), min(y_text + 2, H - 1)), color, -1)
        cv2.putText(out, label, (x_text + 3, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    st.image(Image.fromarray(out[:, :, ::-1]), caption="Detections", use_container_width=True)

# ======================= “Let’s Start Sorting” (everything inside the section) =======================
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

# ---- Inputs (inside the section) ----
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

# ---- Advanced settings (inside the section) ----
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

# ---- Detection flow (inside the same section) ----
if image is not None:
    st.image(image, caption="Input", use_container_width=True)
    should_run = auto_run or st.button("Run detection")
    if should_run:
        dets, counts = run_detection(
            image, conf, iou, imgsz, tta,
            bottle_min=bottle_min, can_min=can_min, foam_min=foam_min, min_area_pct=min_area_pct
        )
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

# ======================= Impact & SDGs (container) =======================
st.markdown("<div id='sdgs'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="section">
  <div class="section-cover">
    <div class="eco-emoji">🌏</div>
    <div class="title">Impact &amp; SDGs</div>
  </div>
  <div class="section-body">
""", unsafe_allow_html=True)

st.markdown("""
- **Carbon credits (what they are):** A carbon credit represents **1 tonne of CO₂ equivalent** reduced or removed. Credits exist only when a **registered project** follows an **approved methodology** and passes **MRV**. They are then **issued on a registry** such as Gold Standard, Verra, or Japan’s J-Credit.
- **This app does not issue credits.** It helps people sort properly. Educational CO₂e-avoided estimates are okay, but they are **not credits**.
""", unsafe_allow_html=True)

st.markdown(
    f"""
<div class="eco-links">
  <a class="eco-link" href="{LINK_UN_CNP}"  target="_blank" rel="noopener">UN Carbon Offset Platform</a>
  <a class="eco-link" href="{LINK_UN_CNP2}" target="_blank" rel="noopener">Climate Neutral Now</a>
  <a class="eco-link" href="{LINK_WB_MRV}"  target="_blank" rel="noopener">World Bank: MRV</a>
  <a class="eco-link" href="{LINK_GS}"      target="_blank" rel="noopener">Gold Standard</a>
  <a class="eco-link" href="{LINK_VERRA}"   target="_blank" rel="noopener">Verra VCS</a>
  <a class="eco-link" href="{LINK_JCREDIT}" target="_blank" rel="noopener">Japan J-Credit</a>
</div>
""", unsafe_allow_html=True)

# SDG tiles (caption optional; hide SDG11 text)
col1, col2, col3 = st.columns(3)
def sdg_tile(col, path, label=None):
    with col:
        if os.path.exists(path):
            st.image(path, width=180)
        else:
            st.warning(f"Missing {path}")
        if label:
            st.markdown(f"<div class='sdg-caption'>{label}</div>", unsafe_allow_html=True)

sdg_tile(col1, "sdg12.png", "12 Responsible Consumption and Production")
sdg_tile(col2, "sdg11.png", None)  # no caption text
sdg_tile(col3, "sdg13.png", "13 Climate Action")

st.markdown("</div></div>", unsafe_allow_html=True)

# ======================= About us (container, below SDGs) =======================
st.markdown("<div id='about'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="section">
  <div class="section-cover">
    <div class="eco-emoji">👋</div>
    <div class="title">About us</div>
  </div>
  <div class="section-body">
    <p>“When AI Sees Litter” is a community project that helps people sort waste correctly using computer vision and local rules.
    Shibuya is the first city we support. More cities are on the way.</p>
  </div>
</div>
""", unsafe_allow_html=True)
