# app.py  -- Robust backend for Eco-Switch AI
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- CONFIG: filenames (keep these exact or change here) ----------
BASE = Path(".")
FUEL_CLEAN = BASE / "clean_fuel_prices.csv"
FUEL_PRED = BASE / "fuel_price_predictions_2025_2030.csv"
EV_SPECS = BASE / "clean_ev_specs.csv"
GEF = BASE / "clean_gef.csv"
MILEAGE = BASE / "clean_mileage.csv"
CO2_FILE = BASE / "clean_co2.csv"
OUT_SUMMARY = BASE / "switch_summary.json"

# ---------- Helpers ----------
def try_read(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_excel(path)
        except Exception:
            return pd.DataFrame()

def normalize(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

# ---------- Loading ----------
def load_all():
    return {
        "clean_fuel": try_read(FUEL_CLEAN),
        "preds": try_read(FUEL_PRED),
        "ev": try_read(EV_SPECS),
        "gef": try_read(GEF),
        "mileage": try_read(MILEAGE),
        "co2": try_read(CO2_FILE)
    }

# ---------- Matching & Fallback logic ----------
def find_country_fuel_preds(preds_df, country, fuel_type):
    """Return predictions DataFrame for a country & fuel (robust matching)."""
    if preds_df.empty or 'country' not in preds_df.columns or 'fuel' not in preds_df.columns:
        return pd.DataFrame()

    df = preds_df.copy()
    df['country_n'] = df['country'].apply(normalize)
    df['fuel_n'] = df['fuel'].apply(normalize)

    country_n = normalize(country)
    fuel_n = normalize(fuel_type)

    # candidate tokens
    petrol_tokens = {'petrol','petrol_price','gasoline','gasoline_price','gas'}
    diesel_tokens = {'diesel','diesel_price'}

    # choose tokens set
    tokens = petrol_tokens if ('petrol' in fuel_n or 'gasoline' in fuel_n) else diesel_tokens

    # find country rows: exact then contains
    mask = df['country_n'] == country_n
    if mask.sum() == 0:
        mask = df['country_n'].str.contains(country_n, na=False)
    sub = df[mask].copy()
    if sub.empty:
        return sub

    # match fuel by checking if any token is in fuel_n
    sub['fuel_match'] = sub['fuel_n'].apply(lambda x: any(tok in x for tok in tokens))
    matched = sub[sub['fuel_match']].sort_values('year')
    if not matched.empty:
        return matched[['country','fuel','year','predicted_price']]
    # as a last resort return any predictions for the country
    return sub.sort_values('year')[['country','fuel','year','predicted_price']]

def fallback_using_historical(clean_fuel_df, country, fuel_type):
    """If no preds found, create constant-year predictions (2025-2030) from last known historical price."""
    if clean_fuel_df.empty or 'country' not in clean_fuel_df.columns:
        return pd.DataFrame()
    df = clean_fuel_df.copy()
    df['country_n'] = df['country'].apply(normalize)
    country_n = normalize(country)
    mask = df['country_n'] == country_n
    sub = df[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    # find column for petrol/diesel
    target_col = None
    if 'petrol' in fuel_type.lower():
        for c in sub.columns:
            if 'petrol' in c.lower():
                target_col = c; break
    else:
        for c in sub.columns:
            if 'diesel' in c.lower():
                target_col = c; break
    if target_col is None:
        return pd.DataFrame()

    # last known price
    valid = pd.to_numeric(sub[target_col], errors='coerce').dropna()
    if valid.empty:
        return pd.DataFrame()
    last_price = float(valid.iloc[-1])
    years = list(range(2025, 2031))
    out = pd.DataFrame({
        'country': [country]*len(years),
        'fuel': [target_col]*len(years),
        'year': years,
        'predicted_price': [last_price]*len(years)
    })
    return out[['country','fuel','year','predicted_price']]

# ---------- Calculations ----------
def annual_fuel_litres(daily_km, mileage_kmpl):
    return (daily_km / mileage_kmpl) * 365.0

def compute_yearly_costs(preds_table, daily_km, mileage_kmpl):
    """Given preds_table with columns year,predicted_price -> return table with annual litres & cost."""
    preds_table = preds_table.copy()
    annual_l = annual_fuel_litres(daily_km, mileage_kmpl)
    preds_table['annual_litres'] = annual_l
    preds_table['annual_cost'] = preds_table['predicted_price'] * preds_table['annual_litres']
    return preds_table[['year','predicted_price','annual_litres','annual_cost']]

# ---------- Summarize function ----------
def summarize_switch(country,
                     fuel_type,
                     daily_km,
                     mileage_kmpl=None,
                     car_model=None,
                     ev_model=None,
                     electricity_price_per_kwh=10.0,
                     save_json=True):
    # load data
    data = load_all()
    clean_fuel = data['clean_fuel']
    preds = data['preds']
    ev = data['ev']
    gef = data['gef']
    mileage = data['mileage']
    co2 = data['co2']

    # resolve mileage_kmpl
    if mileage_kmpl is None:
        # try to find median in mileage file
        if not mileage.empty:
            if 'km_per_litre' in mileage.columns:
                mileage_kmpl = float(mileage['km_per_litre'].dropna().median())
            else:
                # try to find mpg and convert
                mpg_cols = [c for c in mileage.columns if 'mpg' in c.lower()]
                if mpg_cols:
                    mpg_vals = pd.to_numeric(mileage[mpg_cols[0]], errors='coerce').dropna()
                    if not mpg_vals.empty:
                        mileage_kmpl = float(mpg_vals.median() * 0.425144)
    if mileage_kmpl is None:
        raise ValueError("Mileage unknown. Provide mileage_kmpl or ensure clean_mileage.csv has usable mpg/km_per_litre.")

    # find predictions (robust)
    matched = find_country_fuel_preds(preds, country, fuel_type)
    if matched.empty:
        # fallback to historical
        matched = fallback_using_historical(clean_fuel, country, fuel_type)
        if matched.empty:
            raise ValueError(f"No predictions or historical fallback found for {country}, fuel {fuel_type}")

    # get yearly cost table
    costs_df = compute_yearly_costs(matched, daily_km, mileage_kmpl)
    years_count = costs_df['year'].nunique()
    total_fuel_cost_5yr = float(costs_df['annual_cost'].sum())
    total_fuel_litres_5yr = float(costs_df['annual_litres'].sum())

    # CO2 per litre
    co2_map = {}
    if not co2.empty and {'fuel_type','co2_per_litre'}.issubset(set(c.lower() for c in co2.columns)):
        # attempt to standardize lower-case column names
        cols = {c.lower():c for c in co2.columns}
        ft_col = cols.get('fuel_type')
        cp_col = cols.get('co2_per_litre')
        for _, r in co2.iterrows():
            co2_map[str(r[ft_col]).strip().lower()] = float(r[cp_col])
    else:
        co2_map = {'petrol':2.31, 'diesel':2.68}
    co2_per_l = co2_map.get(fuel_type.lower(), list(co2_map.values())[0])

    yearly_fuel_co2_kg = costs_df['annual_litres'] * co2_per_l
    total_fuel_co2_kg_5yr = float(yearly_fuel_co2_kg.sum())

    # EV efficiency pick
    ev_kwh100 = None
    if ev_model and not ev.empty:
        # try matching model column names
        if 'model' in ev.columns and 'kwh_per_100km' in ev.columns:
            row = ev[ev['model'].astype(str).str.lower().str.contains(ev_model.lower(), na=False)]
            if not row.empty:
                ev_kwh100 = float(row.iloc[0]['kwh_per_100km'])
        if ev_kwh100 is None and 'battery_kwh' in ev.columns and 'range_km' in ev.columns:
            row = ev[ev['model'].astype(str).str.lower().str.contains(ev_model.lower(), na=False)] if 'model' in ev.columns else ev.iloc[[0]]
            if not row.empty:
                ev_kwh100 = float(row.iloc[0]['battery_kwh']) / (float(row.iloc[0]['range_km'])/100.0)
    if ev_kwh100 is None:
        if not ev.empty and 'kwh_per_100km' in ev.columns:
            ev_kwh100 = float(ev['kwh_per_100km'].dropna().median())
        else:
            ev_kwh100 = 15.0  # default if empty

    daily_kwh = (daily_km/100.0) * ev_kwh100
    annual_kwh = daily_kwh * 365.0
    annual_ev_cost = annual_kwh * electricity_price_per_kwh
    total_ev_cost_5yr = float(annual_ev_cost * years_count)

    # GEF
    gef_val = None
    if not gef.empty:
        # try common column names
        cols = {c.lower():c for c in gef.columns}
        country_col = None
        gef_col = None
        for key in ['country','country name','country_name']:
            if key in cols: country_col = cols[key]; break
        for key in ['gef','grid emission intensity (kg co2/kwh)','grid emission intensity']:
            if key in cols: gef_col = cols[key]; break
        if country_col and gef_col:
            lookup = {str(r[country_col]).strip().lower(): float(r[gef_col]) for _, r in gef.iterrows() if pd.notna(r[gef_col])}
            gef_val = lookup.get(country.lower(), np.median(list(lookup.values())) if lookup else 0.7)
    if gef_val is None:
        gef_val = 0.7

    annual_ev_co2_kg = annual_kwh * gef_val
    total_ev_co2_kg_5yr = float(annual_ev_co2_kg * years_count)

    money_saved_5yr = total_fuel_cost_5yr - total_ev_cost_5yr
    co2_avoided_kg_5yr = total_fuel_co2_kg_5yr - total_ev_co2_kg_5yr

    out = {
        "country": country,
        "fuel_type": fuel_type,
        "daily_km": daily_km,
        "mileage_kmpl_used": mileage_kmpl,
        "years_breakdown": costs_df.to_dict(orient="records"),
        "total_fuel_cost_5yr": total_fuel_cost_5yr,
        "total_ev_cost_5yr": total_ev_cost_5yr,
        "money_saved_5yr": money_saved_5yr,
        "total_fuel_co2_tons_5yr": total_fuel_co2_kg_5yr/1000.0,
        "total_ev_co2_tons_5yr": total_ev_co2_kg_5yr/1000.0,
        "co2_avoided_tons_5yr": co2_avoided_kg_5yr/1000.0,
        "ev_kwh_per_100km_used": ev_kwh100,
        "annual_ev_kwh": annual_kwh,
        "annual_ev_cost": annual_ev_cost,
        "gef_used_kg_co2_per_kwh": gef_val
    }

    if save_json:
        try:
            with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            print("Warning: could not save summary JSON:", e)

    return out

# ---------- Main: simple interactive example run ----------
if __name__ == "__main__":
    # EDIT these lines to test a scenario
    test_country = "India"
    test_fuel = "petrol"
    test_daily_km = 60
    test_elec_price = 8.0   # local currency per kWh (update to local value)

    try:
        summary = summarize_switch(test_country, test_fuel, daily_km=test_daily_km, electricity_price_per_kwh=test_elec_price)
        print("=== Summary (printed) ===")
        print(json.dumps(summary, indent=2))
        print(f"\nSaved summary to {OUT_SUMMARY}")
    except Exception as exc:
        print("Error:", exc)
        print("Make sure the following files exist in this folder with correct names:")
        for p in [FUEL_CLEAN, FUEL_PRED, EV_SPECS, GEF, MILEAGE, CO2_FILE]:
            print(" ", p.name, "->", p.exists())
