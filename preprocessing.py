"""
preprocessing.py — Data loading, preprocessing for incidents, inventory, app mapping, insights.
"""
import pandas as pd
import re


def load_data(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path, low_memory=False)
    return pd.read_excel(file_path)


def preprocess_data(df):
    required_columns = [
        "category", "closed_dttm", "resolution", "incident_code_id", "priority_df",
        "open_dttm", "autogen", "hostname", "resolution_code", "sso_ticket",
        "mttr_excl_hold", "business_application", "suggested_automata", "closure_code",
        "description", "reassignments", "assignment_grp_parent", "ownergroup", "label",
        "ostype",
    ]
    df = df.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip().str.lower()
    keep = [c for c in df.columns if c in required_columns]
    df = df.loc[:, keep].copy()

    if "open_dttm" in df.columns:
        df["open_dttm"] = pd.to_datetime(df["open_dttm"], errors="coerce", utc=True)
    if "closed_dttm" in df.columns:
        df["closed_dttm"] = pd.to_datetime(df["closed_dttm"], errors="coerce", utc=True)
    if "priority_df" in df.columns:
        df["priority_df"] = pd.to_numeric(df["priority_df"], errors="coerce")
    if "mttr_excl_hold" in df.columns:
        df["mttr_excl_hold"] = pd.to_numeric(df["mttr_excl_hold"], errors="coerce")
    if "reassignments" in df.columns:
        df["reassignments"] = pd.to_numeric(df["reassignments"], errors="coerce")
    return df


def preprocess_inventory(df):
    df = df.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip().str.lower()
    for col in list(df.columns):
        stripped = col.replace(' ', '').replace('_', '')
        if stripped == 'hostname' or stripped == 'hostnames':
            df.rename(columns={col: 'hostname'}, inplace=True)
    keep_cols = ["hostname", "osname", "eol_date", "eos_date", "end_of_extended_support_date", "eoes_status"]
    keep = [c for c in df.columns if c in keep_cols]
    df = df.loc[:, keep].copy()
    return df


def preprocess_app_mapping(df):
    df = df.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip().str.lower()
    keep_cols = ["hostname", "product_name"]
    keep = [c for c in df.columns if c in keep_cols]
    df = df.loc[:, keep].copy()
    return df


def preprocess_actionable_insights(df):
    df = df.copy()
    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip().str.lower()
    for col in list(df.columns):
        stripped = col.replace(' ', '').replace('_', '')
        if stripped == 'insightid':
            df.rename(columns={col: 'insight_id'}, inplace=True)
        if stripped == 'hostname':
            df.rename(columns={col: 'hostname'}, inplace=True)
        if stripped == 'insighttitle':
            df.rename(columns={col: 'insight_title'}, inplace=True)
        if stripped == 'observation':
            df.rename(columns={col: 'observation'}, inplace=True)
        if stripped == 'recommendation':
            df.rename(columns={col: 'recommendation'}, inplace=True)
        if stripped == 'category':
            df.rename(columns={col: 'insight_category'}, inplace=True)
        if stripped == 'insightcategory':
            df.rename(columns={col: 'insight_category'}, inplace=True)
        if stripped == 'impacttype':
            df.rename(columns={col: 'impact_type'}, inplace=True)
        if stripped == 'impactvalue':
            df.rename(columns={col: 'impact_value'}, inplace=True)
    # Keep all relevant columns for insights analysis
    keep_cols = ['hostname', 'insight_id', 'insight_title', 'action', 'observation', 
                 'recommendation', 'insight_category', 'impact_type', 'impact_value']
    keep = [c for c in df.columns if c in keep_cols]
    df = df.loc[:, keep].copy()
    if 'insight_id' in df.columns:
        df['insight_id'] = pd.to_numeric(df['insight_id'], errors='coerce')
    return df


# =========================
# Utility helpers
# =========================

def get_months_in_data(df):
    if "open_dttm" not in df.columns:
        return 1
    valid = df["open_dttm"].dropna()
    if len(valid) == 0:
        return 1
    min_date = valid.min()
    max_date = valid.max()
    months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1
    return max(1, int(months))


def add_month_col(df_in):
    df_out = df_in.copy()
    try:
        df_out["month"] = df_out["open_dttm"].dt.tz_convert(None).dt.to_period("M")
    except Exception:
        df_out["month"] = df_out["open_dttm"].dt.to_period("M")
    return df_out


def calc_projected_yearly_hours(sub_df):
    if "mttr_excl_hold" not in sub_df.columns or "open_dttm" not in sub_df.columns or len(sub_df) == 0:
        return 0.0
    valid = sub_df.dropna(subset=["open_dttm", "mttr_excl_hold"])
    if len(valid) == 0:
        return 0.0
    tmp = add_month_col(valid)
    monthly_avg = tmp.groupby("month")["mttr_excl_hold"].mean()
    avg_of_monthly = float(monthly_avg.mean())
    return round(avg_of_monthly * 12, 2)


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions
