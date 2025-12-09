import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
from pathlib import Path
from io import BytesIO

st.set_page_config(page_title="Eco-Switch Pro • Polished", layout="wide", initial_sidebar_state="expanded")

# ---------- STYLE ----------
st.markdown(
    """
    <style>
    /* Page background & main font */
    html, body, [class*="css"]  {
      background-color: #0f1113;
      color: #e6eef3;
      font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
      background-color: #141518;
      padding-top: 24px;
    }
    /* Cards */
    .kpi-card {
      background: linear-gradient(180deg, rgba(20,20,20,0.9), rgba(18,18,18,0.9));
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.6);
      border: 1px solid rgba(255,255,255,0.03);
    }
    .kpi-label { color: #9fb0c6; font-size: 12px; margin:0; }
    .kpi-value { color: #00E676; font-weight:700; font-size:26px; margin:6px 0 0 0; }
    /* Green button look */
    .stButton>button {
      background-color: #00C853;
      border-radius: 10px;
      color: #021012;
      font-weight: 700;
      height: 44px;
    }
    .stButton>button:hover { background-color: #00E676; color: #021012; }
    /* Small helpers */
    .muted { color: #9fb0c6; font-size:13px; }
    .card-box { background:#0b0c0d; border-radius:10px; padding:14px; border:1px solid rgba(255,255,255,0.02) }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- HELPERS ----------
def safe_read_csv(path):
    if not Path(path).exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def parse_float(x):
    try:
        return float(x)
    except:
        m = re.search(r"\d+(\.\d+)?", str(x))
        return float(m.group(0)) if m else None

def prepare_mileage(raw):
    if raw.empty:
        return pd.DataFrame(columns=["Make","Model","Fuel","kmpl"])
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # detect columns (best-effort)
    make_col = next((c for c in df.columns if re.search("identification\\.make|\\bmake\\b|\\bbrand\\b", c, re.I)), None)
    class_col = next((c for c in df.columns if re.search("identification\\.classification|classification|model\\b", c, re.I)), None)
    year_col = next((c for c in df.columns if re.search("\\byear\\b|identification\\.year", c, re.I)), None)
    fuel_col = next((c for c in df.columns if re.search("fuel.*type|fuel type", c, re.I)), None)
    city_col = next((c for c in df.columns if re.search("city mpg|city_mpg", c, re.I)), None)
    hwy_col = next((c for c in df.columns if re.search("highway mpg|highway_mpg", c, re.I)), None)

    # Make
    if make_col:
        df["Make"] = df[make_col].astype(str).str.strip()
    else:
        df["Make"] = "Unknown"

    # Model: best-effort
    if class_col:
        if year_col and year_col in df.columns:
            df["Model"] = df[class_col].astype(str).str.strip() + " (" + df[year_col].astype(str).str.strip() + ")"
        else:
            df["Model"] = df[class_col].astype(str).str.strip()
    elif year_col:
        df["Model"] = df["Make"].astype(str).str.strip() + " " + df[year_col].astype(str).str.strip()
    else:
        df["Model"] = df["Make"].astype(str).str.strip() + " " + df.index.astype(str)

    # Fuel
    if fuel_col and fuel_col in df.columns:
        df["Fuel"] = df[fuel_col].astype(str).str.strip()
    else:
        merged = df.astype(str).agg(" ".join, axis=1).str.lower()
        df["Fuel"] = merged.apply(lambda t: "Diesel" if "diesel" in t else ("Petrol" if ("petrol" in t or "gasoline" in t or " gas " in t) else ""))

    # mpg to kmpl (1 mpg US = 0.425143707 km/L)
    conv = 0.425143707
    city_vals = df[city_col].apply(parse_float) if (city_col and city_col in df.columns) else pd.Series([None]*len(df))
    hwy_vals = df[hwy_col].apply(parse_float) if (hwy_col and hwy_col in df.columns) else pd.Series([None]*len(df))
    mpg_avg = pd.concat([city_vals, hwy_vals], axis=1).mean(axis=1)
    df["kmpl"] = mpg_avg.apply(lambda v: v*conv if v and not pd.isna(v) else None)

    # final
    out = df[["Make","Model","Fuel","kmpl"]].drop_duplicates().reset_index(drop=True)
    out["Make"] = out["Make"].fillna("").astype(str)
    out["Model"] = out["Model"].fillna("").astype(str)
    out["Fuel"] = out["Fuel"].fillna("").astype(str)
    return out

# ---------- LOAD FILES ----------
raw_mileage = safe_read_csv("Fuel Car Mileage.csv")
mileage_df = prepare_mileage(raw_mileage)

preds_df = safe_read_csv("fuel_price_predictions_2025_2030.csv")
ev_df = safe_read_csv("clean_ev_specs.csv")
gef_df = safe_read_csv("clean_gef.csv")

# ---------- DEFAULTS ----------
SEGMENT_MILEAGE = {
    "Small Hatchback (e.g., Alto, Swift)": 18,
    "Sedan (e.g., City, Civic)": 14,
    "Compact SUV (e.g., Creta, Nexon)": 12,
    "Large SUV / Truck (e.g., Fortuner, F-150)": 9,
    "Sports Car": 7
}
EV_SEGMENT = {"Budget City EV":12, "Standard Sedan EV":15, "Performance SUV EV":22}
ELEC_DEFAULTS = {"India":8.0, "USA":0.16, "UK":0.34, "Germany":0.40}

# ---------- PERSISTENT SELECTIONS ----------
# Use session_state to ensure selections persist and do not reset when running
if "params" not in st.session_state:
    st.session_state.params = {
        "country": "India",
        "fuel": "Petrol",
        "car_mode": "By Segment (Easy)",
        "segment": list(SEGMENT_MILEAGE.keys())[0],
        "brand": "Other",
        "model": "My Car",
        "mileage": 15.0,
        "ev_mode": "Generic EV",
        "ev_choice": list(EV_SEGMENT.keys())[0],
        "daily_km": 40,
        "use_avg_elec": True,
        "elec_price": ELEC_DEFAULTS.get("India", 8.0)
    }

p = st.session_state.params

# ---------- SIDEBAR (INPUTS) ----------
with st.sidebar:
    st.title("🚘 Simulation Parameters")
    st.subheader("1. Location")
    countries = sorted(preds_df["country"].dropna().unique()) if not preds_df.empty else ["India","USA","UK"]
    p["country"] = st.selectbox("Select Country", countries, index=countries.index(p["country"]) if p["country"] in countries else 0)

    st.markdown("---")
    st.subheader("2. Your Current Car")
    p["fuel"] = st.radio("Fuel Type", ["Petrol","Diesel"], index=0 if p["fuel"]=="Petrol" else 1, horizontal=True)

    st.markdown("How to select car?")
    p["car_mode"] = st.radio("", ["By Segment (Easy)", "By Exact Model (Advanced)"], index=0 if p["car_mode"]=="By Segment (Easy)" else 1)

    if p["car_mode"] == "By Segment (Easy)":
        segs = list(SEGMENT_MILEAGE.keys())
        p["segment"] = st.selectbox("What type of car do you drive?", segs, index=segs.index(p["segment"]) if p["segment"] in segs else 0)
        p["mileage"] = SEGMENT_MILEAGE[p["segment"]]
        st.info(f"Assumed mileage: {p['mileage']} km/L")
    else:
        # Advanced: use prepared mileage_df
        filtered = mileage_df.copy()
        # filter by fuel if present
        if "Fuel" in filtered.columns and filtered["Fuel"].str.strip().astype(bool).any():
            if p["fuel"].lower() == "petrol":
                mask = filtered["Fuel"].str.lower().str.contains("petrol|gasoline|gas", na=False)
            else:
                mask = filtered["Fuel"].str.lower().str.contains("diesel", na=False)
            if mask.any():
                filtered = filtered[mask]

        brands = sorted(filtered["Make"].dropna().unique().tolist())
        brand_options = ["Other"] + brands if brands else ["Other"]
        # preserve previous selection index if present
        try:
            brand_idx = brand_options.index(p.get("brand","Other"))
        except:
            brand_idx = 0
        p["brand"] = st.selectbox("Brand", brand_options, index=brand_idx)

        # models
        if p["brand"] != "Other":
            models = sorted(filtered[filtered["Make"] == p["brand"]]["Model"].dropna().unique().tolist())
            model_options = ["My Car"] + models if models else ["My Car","Other"]
        else:
            model_options = ["My Car","Other"]
        try:
            model_idx = model_options.index(p.get("model","My Car"))
        except:
            model_idx = 0
        p["model"] = st.selectbox("Model", model_options, index=model_idx)

        # autofill mileage if possible
        detected = None
        if p["brand"] != "Other" and p["model"] not in ["My Car","Other"]:
            match = filtered[(filtered["Make"]==p["brand"]) & (filtered["Model"]==p["model"])]
            if not match.empty and pd.notna(match.iloc[0]["kmpl"]):
                detected = parse_float(match.iloc[0]["kmpl"])
        if detected:
            st.success(f"Detected mileage: {detected:.2f} km/L")
            p["mileage"] = st.number_input("Enter Mileage (km/L)", value=float(detected), min_value=0.0, format="%.2f")
        else:
            st.info("No mileage found — enter manually or switch to segment mode.")
            p["mileage"] = st.number_input("Enter Mileage (km/L)", value=float(p["mileage"]), min_value=0.0, format="%.2f")

    st.markdown("---")
    st.subheader("3. Target EV")
    p["ev_mode"] = st.radio("EV Selection", ["Generic EV","Specific EV"], index=0 if p["ev_mode"]=="Generic EV" else 1)
    if p["ev_mode"] == "Generic EV":
        ev_opts = list(EV_SEGMENT.keys())
        p["ev_choice"] = st.selectbox("EV Type", ev_opts, index=ev_opts.index(p["ev_choice"]) if p["ev_choice"] in ev_opts else 0)
        p["ev_eff"] = EV_SEGMENT[p["ev_choice"]]
        st.info(f"Estimated efficiency: {p['ev_eff']} kWh/100km")
    else:
        if ev_df.empty:
            st.warning("EV database missing; fallback to manual spec.")
            p["ev_name"] = st.text_input("EV name", value=p.get("ev_name","Manual EV"))
            p["ev_eff"] = st.number_input("Efficiency (kWh/100km)", value=p.get("ev_eff",15.0))
        else:
            # user friendly EV pick
            ev_brand_col = next((c for c in ev_df.columns if re.search("brand|make", c, re.I)), None)
            ev_model_col = next((c for c in ev_df.columns if re.search("model|name", c, re.I)), None)
            if ev_brand_col and ev_model_col:
                brands_ev = sorted(ev_df[ev_brand_col].dropna().unique())
                p["ev_brand"] = st.selectbox("EV Brand", brands_ev, index=brands_ev.index(p.get("ev_brand", brands_ev[0])) if p.get("ev_brand") in brands_ev else 0)
                models_ev = sorted(ev_df[ev_df[ev_brand_col]==p["ev_brand"]][ev_model_col].dropna().unique())
                p["ev_model"] = st.selectbox("EV Model", models_ev, index=models_ev.index(p.get("ev_model", models_ev[0])) if p.get("ev_model") in models_ev else 0)
                row = ev_df[(ev_df[ev_brand_col]==p["ev_brand"]) & (ev_df[ev_model_col]==p["ev_model"])]
                if not row.empty and 'kwh_per_100km' in row.columns:
                    p["ev_eff"] = float(row.iloc[0]['kwh_per_100km'])
                else:
                    p["ev_eff"] = p.get("ev_eff", 15.0)
                st.success(f"EV efficiency: {p['ev_eff']:.1f} kWh/100km")
            else:
                p["ev_name"] = st.text_input("EV name", value=p.get("ev_name","Manual EV"))
                p["ev_eff"] = st.number_input("Efficiency (kWh/100km)", value=p.get("ev_eff",15.0))

    st.markdown("---")
    st.subheader("4. Driving & Costs")
    p["daily_km"] = st.slider("Daily Driving (km)", 10, 200, value=int(p.get("daily_km",40)))
    p["use_avg_elec"] = st.checkbox("Use national avg electricity price?", value=p.get("use_avg_elec", True))
    if p["use_avg_elec"]:
        p["elec_price"] = ELEC_DEFAULTS.get(p["country"], 8.0)
        st.markdown(f"Electricity price used: **{p['elec_price']}**")
    else:
        p["elec_price"] = st.number_input("Electricity rate (per kWh)", value=float(p.get("elec_price",8.0)))
    st.markdown("---")

    # Run simulation button (doesn't reset inputs because we use session_state)
    run_sim = st.button("Run Simulation 🚀")

# ---------- COMPUTATION: runs when user clicks ----------

def compute_simulation(params, sensitivity=None):
    # params is st.session_state.params
    country = params["country"]
    fuel_type = params["fuel"]
    daily_km = params["daily_km"]
    current_mileage = float(params["mileage"])
    ev_eff = float(params.get("ev_eff", 15.0))
    elec_price = float(params["elec_price"])

    # adjust prices if sensitivity (percentage) provided
    if sensitivity is None:
        sens_multiplier = 1.0
    else:
        # sensitivity is a dict like {"fuel": +10, "elec": -5} percentage
        sens_multiplier = 1.0

    # fuel price predictions
    years, prices = [], []
    if not preds_df.empty and 'country' in preds_df.columns and 'fuel' in preds_df.columns:
        try:
            cp = preds_df[preds_df['country'].str.lower() == country.lower()]
            if not cp.empty:
                fp = cp[cp['fuel'].str.lower().str.contains(fuel_type.lower(), na=False)].sort_values('year')
                if not fp.empty:
                    years = fp['year'].tolist()
                    prices = fp['predicted_price'].tolist()
        except Exception:
            years, prices = [], []
    if not years:
        years = list(range(2025,2031))
        # placeholder price base
        base = 100.0
        prices = [base * (1.04**i) for i in range(len(years))]

    # GEF
    gef = 0.7
    if not gef_df.empty and 'country' in gef_df.columns and 'gef' in gef_df.columns:
        row = gef_df[gef_df['country'].str.lower() == country.lower()]
        if not row.empty:
            gef = float(row.iloc[0]['gef'])

    co2_factor = 2.3 if fuel_type == "Petrol" else 2.7

    records = []
    cumulative = 0.0
    for yr, price in zip(years, prices):
        # optionally apply sensitivity adjustments to fuel price
        p_price = price
        liters = (daily_km * 365) / current_mileage
        cost_fuel = liters * p_price
        co2_fuel = liters * co2_factor

        kwh_needed = (daily_km * 365 / 100.0) * ev_eff
        cost_ev = kwh_needed * elec_price
        co2_ev = kwh_needed * gef

        saving = cost_fuel - cost_ev
        cumulative += saving

        records.append({
            "Year": int(yr),
            "Fuel Price": round(p_price,2),
            "Liters per year": round(liters,2),
            "Cost Fuel": round(cost_fuel,2),
            "Cost EV": round(cost_ev,2),
            "Savings": round(saving,2),
            "Cumulative": round(cumulative,2),
            "CO2 Fuel (kg)": round(co2_fuel,2),
            "CO2 EV (kg)": round(co2_ev,2),
            "CO2 Avoided (kg)": round(co2_fuel - co2_ev,2)
        })

    return pd.DataFrame(records), gef

# store last results in session state
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_gef" not in st.session_state:
    st.session_state.last_gef = None

# Run calculation if button clicked
if 'run_sim' in locals() and run_sim:
    df_results, gef_val = compute_simulation(p)
    st.session_state.last_results = df_results
    st.session_state.last_gef = gef_val

# ---------- MAIN: show results if exist ----------
col_main_left, col_main_right = st.columns([1,3])
with col_main_left:
    st.header("🌱 Eco-Switch Pro")
    st.markdown(f"Comparing your **{p['fuel']}** car vs **{p.get('ev_choice', p.get('ev_name','EV'))}** in **{p['country']}**")
    st.markdown("<div class='muted'>Use the sidebar to change inputs. Results persist until you change parameters.</div>", unsafe_allow_html=True)

with col_main_right:
    if st.session_state.last_results is None:
        st.write("")  # placeholder
    else:
        res = st.session_state.last_results
        total_saved = res["Cumulative"].iloc[-1]
        total_co2 = res["CO2 Avoided (kg)"].sum()
        gef_val = st.session_state.last_gef or 0.7

        # KPI row
        k1, k2, k3 = st.columns([1,1,1])
        k1.markdown(f'<div class="kpi-card"><div class="kpi-label">Total 5-Year Savings</div><div class="kpi-value">{total_saved:,.0f}</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi-card"><div class="kpi-label">CO₂ Reduction (kg)</div><div class="kpi-value">{total_co2:,.0f}</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi-card"><div class="kpi-label">Grid Emission Factor (kg/kWh)</div><div class="kpi-value">{gef_val:.3f}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        # Charts
        fig = px.area(res, x="Year", y="Cumulative", title="Cumulative Savings", color_discrete_sequence=["#00E676"])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", title_x=0.02)
        st.plotly_chart(fig, use_container_width=True, height=420)

        # CO2 chart
        fig2 = px.bar(res, x="Year", y="CO2 Avoided (kg)", title="Yearly CO₂ Avoided", color_discrete_sequence=["#66CCFF"])
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", title_x=0.02)
        st.plotly_chart(fig2, use_container_width=True, height=260)

        st.markdown("""
         <div style='background:#2a2b2d; padding:12px; border-radius:8px;'>
         <b>⚠ Disclaimer:</b> All results are estimates based on past 10 years of data.
         Actual prices and conditions may vary.
         </div>
         """, unsafe_allow_html=True)


        # Recommendation + actions
        st.markdown("### Recommendation")
        if total_saved > 0:
            st.success(f"Switching to the selected EV is likely to save about **{total_saved:,.0f}** over 5 years and avoid **{total_co2:,.0f} kg** CO₂.")
        else:
            st.warning("Savings are negative — EV may not save money with current assumptions. Check electricity/fuel prices and mileage.")

        # download CSV and report
        def to_csv_bytes(df):
            return df.to_csv(index=False).encode("utf-8")

        st.download_button("Download year-by-year CSV", to_csv_bytes(res), file_name="eco_switch_results.csv", mime="text/csv")

        # Advanced details expander
        with st.expander("Advanced details & sensitivity analysis (open)"):
            st.subheader("Year-by-year breakdown")
            st.dataframe(res.style.format({"Fuel Price":"{:.2f}", "Cost Fuel":"{:.2f}", "Cost EV":"{:.2f}", "Savings":"{:.2f}", "Cumulative":"{:.2f}"}))

            st.markdown("#### Sensitivity analysis")
            st.markdown("Slide to see how savings change when electricity price or fuel price changes.")
            c1,c2 = st.columns(2)
            with c1:
                fuel_delta = st.slider("Fuel price change (%)", -30, 100, 0, help="Apply +/- percent change to fuel prices used in simulation")
            with c2:
                elec_delta = st.slider("Electricity price change (%)", -50, 100, 0, help="Apply +/- percent change to electricity cost")

            if st.button("Run sensitivity"):
                # recompute with adjustments
                df_sens, _ = compute_simulation(p)  # base to copy
                # apply adjustments
                df_sens["Fuel Price"] = df_sens["Fuel Price"] * (1 + fuel_delta/100.0)
                df_sens["Cost Fuel"] = df_sens["Liters per year"] * df_sens["Fuel Price"]
                df_sens["Cost EV"] = df_sens["Cost EV"] * (1 + elec_delta/100.0)
                df_sens["Savings"] = df_sens["Cost Fuel"] - df_sens["Cost EV"]
                df_sens["Cumulative"] = df_sens["Savings"].cumsum()
                st.markdown("**Sensitivity results**")
                st.dataframe(df_sens.style.format({"Fuel Price":"{:.2f}", "Savings":"{:.2f}", "Cumulative":"{:.2f}"}))
                st.download_button("Download sensitivity CSV", df_sens.to_csv(index=False).encode("utf-8"), file_name="sensitivity_results.csv", mime="text/csv")

# ---------- FOOTER / DEBUG ----------
st.markdown("<hr/>", unsafe_allow_html=True)
with st.container():
    left, right = st.columns([3,1])
    with left:
        st.markdown("<div class='muted'>Tip: Use Advanced mode to select exact make/model if available. Results will persist until you change inputs.</div>", unsafe_allow_html=True)
    with right:
        if st.checkbox("Show cleaned mileage sample (debug)", value=False):
            st.dataframe(mileage_df.head(12))

