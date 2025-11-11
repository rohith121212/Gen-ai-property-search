# app.py
# Bengaluru Property Search Assistant — 100k dataset (NO IMAGES)
# Includes:
# - Exact Area filter (with tolerance %)
# - Google Maps links (map popups, cards, results table)
# - Filters: keyword/location/type/price/area/builder/amenities/BHK
# - Builder & project details, amenities list
# - Favorites (persistent JSON)
# - Map clustering + performance sampling

import json
import warnings
import urllib.parse
from pathlib import Path

import streamlit as st
import pandas as pd

# Dependencies: streamlit-folium + folium for maps
try:
    from streamlit_folium import st_folium
    import folium
    from folium.plugins import MarkerCluster
except ModuleNotFoundError:
    st.error(
        "Missing dependency: `streamlit-folium`.\n\n"
        "Install with:\n\n"
        "    python -m pip install -U streamlit streamlit-folium folium pandas\n"
    )
    st.stop()

# --- CONFIG ---
st.set_page_config(page_title="🏡 Property Search Assistant - Bengaluru (100k, no images)", layout="wide")
warnings.filterwarnings("ignore", message=".*infer vegalite type from 'empty'.*")

# --- PATHS ---
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Point to your 100k dataset (without using images)
DATA_PATH = DATA_DIR / "Bengaluru_House_Data_100k_with_images_builders_amenities.json"
# Fallback to earlier file if needed
if not DATA_PATH.exists():
    fallback = DATA_DIR / "Bengaluru_House_Data_with_coords.json"
    if fallback.exists():
        DATA_PATH = fallback

FAV_PATH = DATA_DIR / "favorites.json"
if not FAV_PATH.exists():
    FAV_PATH.write_text("[]", encoding="utf-8")

# --- UTILS: Favorites ---
def load_favorites() -> list:
    try:
        return json.loads(FAV_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_favorites(favs: list):
    FAV_PATH.write_text(json.dumps(favs, indent=2, ensure_ascii=False), encoding="utf-8")

# --- UTILS: Google Maps link ---
def make_gmaps_link(row: pd.Series) -> str:
    lat, lng = row.get("lat"), row.get("lng")
    title = str(row.get("title") or "")
    loc = str(row.get("location") or "")
    if pd.notna(lat) and pd.notna(lng):
        return f"https://www.google.com/maps?q={lat},{lng}"
    q = urllib.parse.quote_plus(f"{title} {loc} Bengaluru")
    return f"https://www.google.com/maps/search/?api=1&query={q}"

# --- LOAD DATA ---
@st.cache_data(show_spinner=True)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.warning(f"Data file not found at `{path}`.")
        return pd.DataFrame(columns=[
            "id","title","description","price","type","area","location","lat","lng",
            "builder","project","amenities"
        ])
    df = pd.read_json(path)

    # Ensure expected columns exist (ignore images entirely)
    expected = ["id","title","description","price","type","area","location","lat","lng","builder","project","amenities"]
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA

    # Types & cleaning
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["area"]  = pd.to_numeric(df["area"], errors="coerce")

    # Derive BHK (plots -> 0)
    if "bhk" not in df.columns:
        def bhk_from_row(r):
            if str(r.get("type")).lower() == "plot":
                return 0
            title = str(r.get("title") or "")
            for n in range(1, 7):
                if f"{n} bhk" in title.lower():
                    return n
            area = pd.to_numeric(r.get("area"), errors="coerce")
            return int(max(1, min(5, (area or 450)//450)))
        df["bhk"] = df.apply(bhk_from_row, axis=1)

    # Normalize amenities to list
    def ensure_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x) or x is None:
            return []
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else [str(v)]
        except Exception:
            return [str(x)]
    df["amenities"] = df["amenities"].apply(ensure_list)

    # Clean strings for UI
    df["location"] = df["location"].astype(str)
    df["builder"]  = df["builder"].astype(str)
    df["type"]     = df["type"].astype(str)

    return df

df = load_data(DATA_PATH)

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Search & Filters")

# Keyword + location
q = st.sidebar.text_input("Keyword (title/description/type/project/builder)")
location = st.sidebar.text_input("Location (e.g., Whitefield, HSR Layout)")

# Dynamic ranges
price_min = int(pd.to_numeric(df["price"], errors="coerce").min() or 0)
price_max = int(pd.to_numeric(df["price"], errors="coerce").max() or 10_00_00_000)
area_min = int(pd.to_numeric(df["area"], errors="coerce").min() or 0)
area_max = int(pd.to_numeric(df["area"], errors="coerce").max() or 10_000)

sel_price = st.sidebar.slider("Price range (₹)", min_value=price_min, max_value=price_max,
                              value=(price_min, price_max), step=10000)
sel_area = st.sidebar.slider("Area (sqft)", min_value=area_min, max_value=area_max,
                             value=(area_min, area_max), step=50)

# 🎯 Exact area filter (optional)
st.sidebar.markdown("#### 🎯 Exact Area Filter (optional)")
target_area = st.sidebar.number_input("Target area (sqft)", min_value=0, value=0, step=50, help="Leave 0 to disable")
tolerance_pct = st.sidebar.slider("Tolerance (%)", min_value=0, max_value=30, value=5, help="±% around target area")

# Type / BHK / Builder
all_types = ["All"] + sorted([t for t in df["type"].dropna().unique().tolist() if t])
ptype = st.sidebar.selectbox("Property Type", all_types)
bhk_options = ["All", 1, 2, 3, 4, 5]
bhk_sel = st.sidebar.selectbox("BHK", bhk_options, index=0)
builders = ["All"] + sorted([b for b in df["builder"].dropna().unique().tolist() if b])
builder_sel = st.sidebar.selectbox("Builder", builders)

# Amenities (must include all)
amen_counts = {}
for ams in df["amenities"]:
    for a in ams:
        amen_counts[a] = amen_counts.get(a, 0) + 1
top_amenities = [a for a, _ in sorted(amen_counts.items(), key=lambda x: x[1], reverse=True)][:30]
amenities_sel = st.sidebar.multiselect("Amenities (must include all)", top_amenities)

# Extras
actions = st.sidebar.multiselect("Extras", [
    "Show average price by locality",
    "Show top localities by listing count",
])

# Performance knobs
st.sidebar.subheader("⚡ Performance")
max_map_points = st.sidebar.number_input("Max markers on map", min_value=200, max_value=5000, value=1200, step=100)
max_table_rows = st.sidebar.number_input("Max rows in table", min_value=500, max_value=10000, value=3000, step=100)
random_state = 42

# --- APPLY FILTERS ---
filtered = df.copy()

def contains(text, needles):
    t = (text or "").lower()
    return any(n in t for n in needles)

# Keyword
if q:
    ql = q.lower().strip()
    tokens = [ql]
    filtered = filtered[filtered.apply(
        lambda r: contains(str(r.get("title","")), tokens) or
                  contains(str(r.get("description","")), tokens) or
                  contains(str(r.get("type","")), tokens) or
                  contains(str(r.get("project","")), tokens) or
                  contains(str(r.get("builder","")), tokens),
        axis=1
    )]

# Location (substring)
if location:
    filtered = filtered[filtered["location"].astype(str).str.contains(location, case=False, na=False)]

# Type
if ptype != "All":
    filtered = filtered[filtered["type"] == ptype]

# Price / Area (range)
filtered = filtered[
    (pd.to_numeric(filtered["price"], errors="coerce") >= sel_price[0]) &
    (pd.to_numeric(filtered["price"], errors="coerce") <= sel_price[1]) &
    (pd.to_numeric(filtered["area"], errors="coerce")  >= sel_area[0]) &
    (pd.to_numeric(filtered["area"], errors="coerce")  <= sel_area[1])
]

# 🎯 Exact area filter
if target_area > 0:
    tol = target_area * (tolerance_pct / 100.0)
    filtered = filtered[
        (pd.to_numeric(filtered["area"], errors="coerce") >= (target_area - tol)) &
        (pd.to_numeric(filtered["area"], errors="coerce") <= (target_area + tol))
    ]

# BHK (ignore plots)
if bhk_sel != "All":
    filtered = filtered[(filtered["bhk"] == int(bhk_sel)) | (filtered["type"].str.lower() == "plot")]

# Builder
if builder_sel != "All":
    filtered = filtered[filtered["builder"] == builder_sel]

# Amenities — require all selected
if amenities_sel:
    aset = set(amenities_sel)
    filtered = filtered[filtered["amenities"].apply(lambda L: aset.issubset(set(L)))]

# --- PAGE HEADER ---
st.title("🏡 Bengaluru Property Search Assistant (100k, no images)")
st.caption("Explore properties, view them on a cluster map, open Google Maps, and save your favorites. Optimized for large datasets.")

col1, col2 = st.columns([2, 1], gap="large")

# --- FAVORITES PANEL ---
with col2:
    st.markdown("### 🔖 Favorites")
    favs = load_favorites()
    if not favs:
        st.info("No favorites saved yet. Click ★ on any property card to add.")
    else:
        for f in favs[:25]:
            price_val = f.get("price")
            price_disp = f"₹{int(price_val):,}" if pd.notna(price_val) else "₹—"
            st.write(f"**{f.get('title','(no title)')}** — {price_disp} — {f.get('location','—')}")

# --- MAP + RESULTS ---
with col1:
    st.markdown(f"### 🔎 Results: {len(filtered):,} properties found")

    # Build map with clustering; sample for performance
    if len(filtered) > max_map_points:
        st.info(f"Showing a random sample of **{max_map_points}** on the map for performance. Refine filters to see more.")
        map_df = filtered.sample(max_map_points, random_state=random_state)
    else:
        map_df = filtered

    # Create folium map
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=11, tiles="OpenStreetMap")
    cluster = MarkerCluster().add_to(m)

    for _, row in map_df.iterrows():
        lat, lng = row.get("lat"), row.get("lng")
        if pd.notna(lat) and pd.notna(lng):
            price_val = row.get("price")
            price_disp = f"₹{int(price_val):,}" if pd.notna(price_val) else "₹—"
            gmaps_url = make_gmaps_link(row)
            popup_html = (
                f"<b>{row.get('title','(no title)')}</b><br/>"
                f"{price_disp} · {row.get('area','—')} sqft<br/>"
                f"{row.get('location','—')}<br/>"
                f"<i>{row.get('type','—')}</i> · {row.get('bhk','—')} BHK<br/>"
                f"Builder: {row.get('builder','—')}<br/>"
                f"Project: {row.get('project','—')}<br/>"
                f"<a href='{gmaps_url}' target='_blank'>Open in Google Maps</a>"
            )
            folium.Marker([float(lat), float(lng)], popup=popup_html).add_to(cluster)
    st_folium(m, width=950, height=520)

    # Results Table (sampled if too big) with clickable Google Maps link
    st.markdown("#### 📋 Results Table")
    show_table = st.checkbox("Show results table", value=False)
    if show_table:
        table_df = filtered[["title","location","price","area","type","bhk","builder","project"]].copy()
        table_df["Google Maps"] = filtered.apply(make_gmaps_link, axis=1)

        if len(table_df) > max_table_rows:
            st.info(f"Showing a random sample of **{max_table_rows}** rows for performance.")
            table_df = table_df.sample(max_table_rows, random_state=random_state)

        st.data_editor(
            table_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Google Maps": st.column_config.LinkColumn("Google Maps", display_text="Open")
            }
        )

    # Property Cards (limit number to keep UI snappy)
    st.markdown("### 🏠 Property Details")
    max_cards = 300
    if len(filtered) > max_cards:
        st.info(f"Showing the first **{max_cards}** property cards. Use filters to narrow down.")
    cards_df = filtered.head(max_cards).reset_index(drop=True)

    for idx, row in cards_df.iterrows():
        rid = row.get("id")
        key_base = str(rid) if pd.notna(rid) else str(idx)
        price_val = row.get("price")
        price_disp = f"₹{int(price_val):,}" if pd.notna(price_val) else "₹—"
        header = f"{row.get('title','(no title)')} — {price_disp}"
        with st.expander(header):
            # Top meta
            st.write(
                f"**Location:** {row.get('location','—')}\n\n"
                f"**Type:** {row.get('type','—')} — **Area:** {row.get('area') or '—'} sqft — **BHK:** {row.get('bhk','—')}\n\n"
                f"**Builder:** {row.get('builder','—')} — **Project:** {row.get('project','—')}"
            )
            # Description
            st.write(row.get("description") or "No additional description available.")

            # Amenities
            amenities = row.get("amenities", [])
            if amenities:
                st.write("**Amenities:** " + ", ".join(amenities[:20]))

            # Google Maps link
            gmaps_url = make_gmaps_link(row)
            st.markdown(f"[🗺️ Open in Google Maps]({gmaps_url})")

            # Favorite actions
            bcol = st.columns([1, 1, 6])
            if bcol[0].button("★ Add to favorites", key=f"fav_add_{key_base}"):
                favs = load_favorites()
                if not any(str(f.get("id")) == str(rid) for f in favs):
                    record = {k: row.get(k) for k in ["id","title","price","location","type","area","bhk","builder","project"]}
                    favs.insert(0, record)
                    favs = favs[:200]  # cap
                    save_favorites(favs)
                    st.success("Added to favorites ✅")
                else:
                    st.info("Already in favorites.")
            if bcol[1].button("Remove", key=f"fav_remove_{key_base}"):
                favs = load_favorites()
                favs = [f for f in favs if str(f.get("id")) != str(rid)]
                save_favorites(favs)
                st.info("Removed from favorites ❌")

# --- CHARTS ---
if "Show average price by locality" in actions:
    st.markdown("### 📈 Average Price by Locality (Top 20)")
    grp = filtered.groupby("location", dropna=True)["price"].mean().dropna().sort_values(ascending=False).head(20)
    if not grp.empty:
        st.bar_chart(grp)
    else:
        st.info("No data available for chart — adjust your filters.")

if "Show top localities by listing count" in actions:
    st.markdown("### 📍 Top Localities by Listing Count")
    grp2 = filtered["location"].dropna().value_counts().head(25)
    if not grp2.empty:
        st.bar_chart(grp2)
    else:
        st.info("No data available for chart — adjust your filters.")

# --- FOOTER TIP ---
st.caption("Tip: Use the exact area filter with a small tolerance to pinpoint floor plans. Open listings directly in Google Maps from the cards or the table.")
