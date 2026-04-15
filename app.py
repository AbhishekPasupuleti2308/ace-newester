"""
app.py — Kyndryl Bridge Intelligence: Unified Data Platform
Combines the incident-analysis-dashboard with automated data extraction
from frontend (cookie APIs) and backend (Elasticsearch).
"""
from flask import Flask, render_template, request, jsonify, Response, send_file
import pandas as pd
import numpy as np
import json
import math
import os
import re
import io
import csv
import time
import tempfile
import threading
import uuid
import traceback
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import requests as http_requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config import (
    AZURE_CONFIGURED, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
    INDUSTRY_DOWNTIME_COSTS, EUR_TO_USD, NOISE_COST_PER_HOUR, convert_currency,
)
from preprocessing import (
    load_data, preprocess_data, preprocess_inventory, preprocess_app_mapping,
    preprocess_actionable_insights, allowed_file, get_months_in_data, add_month_col,
    calc_projected_yearly_hours,
)
from analysis_core import (
    perform_analysis_on_dataframe, split_managed_nonmanaged, build_comparison_data,
)
from analysis_vesa import perform_vesa_analysis
from analysis_insights import analyse_insights_file
from visualizations import create_visualizations, create_comparison_visualizations
from llm_helpers import (
    generate_with_azure, try_answer_deterministically,
    SYSTEM_PROMPT_GENERAL, SYSTEM_PROMPT_EXEC_SUMMARY,
)


def clean_for_json(obj):
    """Recursively clean an object for JSON serialization, replacing NaN/Inf with None."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif pd.isna(obj):
        return None
    else:
        return obj

# =========================
# Flask app config
# =========================
app = Flask(__name__)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
app.config["UPLOAD_FOLDER"] = os.path.join(_APP_DIR, "uploads")
app.config["OUTPUT_FOLDER"] = os.path.join(_APP_DIR, "outputs")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
app.config["ALLOWED_EXTENSIONS"] = {"csv", "xlsx", "xls"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# Global state
analysis_data = {}
calculation_methods = {}
supplementary_data = {
    "inventory": None,
    "app_mapping": None,
    "incidents_filepath": None,
    "actionable_insights": None,
    "insights_combined_filepath": None,
    "change_filepath": None,
    "service_requests_filepath": None,
}
global_llm_context = {}
latest_exec_summary = {}
vesa_results = {}  # Store VESA analysis results
change_data = {}   # Store Change Management analysis results
service_request_data = {}  # Store Service Request analysis results
insights_v2_data = {}  # Store new insights analysis (actionable/growth/threatcon)


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"error": "File too large", "max_bytes": app.config.get("MAX_CONTENT_LENGTH")}), 413


# =========================
# Global LLM Context Builder
# =========================
def _rebuild_global_llm_context():
    global global_llm_context
    ctx = {}

    if analysis_data and "complete_analysis" in analysis_data:
        ca = analysis_data["complete_analysis"]
        meta = ca.get("metadata", {})
        sa = ca.get("severity_analysis", {}) or {}
        ao = ca.get("automation_opportunity", {}) or {}
        ci = ca.get("common_issues", {}) or {}
        llm_ctx = ca.get("llm_context", {}) or {}

        ctx["incident_analysis"] = {
            "total_tickets": int(meta.get("total_tickets", 0)),
            "months_of_data": meta.get("months_of_data", 0),
            "date_range": meta.get("date_range", {}),
            "industry": meta.get("industry"),
            "currency": meta.get("currency", "USD"),
            "high_priority_tickets_sev1_2": int(sa.get("high_priority_count", 0)),
            "high_priority_percentage": float(sa.get("high_priority_percentage", 0)),
            "severity_counts": sa.get("severity_counts", {}),
            "mttr_by_severity": sa.get("mttr_by_severity", {}),
            "automation": {
                "automation_opportunity_percentage": float(ao.get("automation_opportunity_percentage", 0)),
                "automation_opportunity_count": int(ao.get("automation_opportunity_count", 0)),
                "total_tickets_analyzed": int(ao.get("total_tickets_analyzed", 0)),
                "tickets_after_closure_exclusion": int(ao.get("tickets_after_closure_exclusion", 0)),
                "tickets_with_suggested_automata": int(ao.get("tickets_with_suggested_automata", 0)),
                "potential_time_savings_hours": float(ao.get("potential_time_savings_hours", 0)),
                "opportunities_by_automata": ao.get("opportunities_by_automata", {}),
                "opportunities_by_category": ao.get("opportunities_by_category", {}),
                "automation_by_severity": ao.get("automation_by_severity", {}),
                "exclusions": ao.get("exclusions", {}),
                "label_mappings_applied": int(ao.get("label_mappings_applied", 0)),
                "label_mapping_details": ao.get("label_mapping_details", {}),
                "analysis_method": ao.get("analysis_method", "unknown"),
                "analysis_note": ao.get("analysis_note", ""),
            },
            "common_issues": {
                "top_descriptions": ci.get("top_descriptions", [])[:10],
                "top_resolutions": ci.get("top_resolutions", [])[:10],
                "top_categories": ci.get("top_categories", [])[:10],
                "description_keywords": ci.get("description_keywords", [])[:20],
                "resolution_keywords": ci.get("resolution_keywords", [])[:20],
                "common_issue_phrases": ci.get("common_issue_phrases", [])[:15],
                "sev1_top_descriptions": ci.get("sev1_top_descriptions", [])[:10],
                "sev2_top_descriptions": ci.get("sev2_top_descriptions", [])[:10],
                "issues_by_top_app": ci.get("issues_by_top_app", {}),
            },
            **llm_ctx,
        }

        top_apps = ca.get("top_applications", {}) or {}
        all_apps_costs = ca.get("all_applications_costs", {}) or {}
        ctx["incident_analysis"]["top_apps_analysis"] = []
        for app_name, app_data_item in list(top_apps.items())[:10]:
            d = app_data_item.get("downtime", {})
            n = app_data_item.get("noise", {})
            ctx["incident_analysis"]["top_apps_analysis"].append({
                "application": app_name,
                "total_tickets": app_data_item.get("total_tickets", 0),
                "downtime_ticket_count": d.get("ticket_count", 0),
                "downtime_projected_yearly_hours": d.get("projected_yearly_hours", 0),
                "downtime_projected_yearly_cost": d.get("projected_yearly_cost"),
                "noise_ticket_count": n.get("ticket_count", 0),
                "noise_projected_yearly_hours": n.get("projected_yearly_hours", 0),
                "noise_projected_yearly_cost": n.get("projected_yearly_cost"),
            })

        ctx["incident_analysis"]["all_apps_costs_top20"] = [
            {
                "application": k,
                "total_tickets": v.get("total_tickets", 0),
                "downtime_sev12_tickets": v.get("downtime", {}).get("ticket_count", 0),
                "downtime_projected_yearly_hours": v.get("downtime", {}).get("projected_yearly_hours", 0),
                "downtime_projected_yearly_cost": v.get("downtime", {}).get("projected_yearly_cost"),
                "noise_sev34_tickets": v.get("noise", {}).get("ticket_count", 0),
                "noise_projected_yearly_hours": v.get("noise", {}).get("projected_yearly_hours", 0),
                "noise_projected_yearly_cost": v.get("noise", {}).get("projected_yearly_cost"),
            }
            for k, v in list(all_apps_costs.items())[:20]
        ]

        ctx["comparison_data"] = analysis_data.get("comparison_data", {})

    # VESA analytics context
    if vesa_results:
        vesa_ctx = {}
        tc = vesa_results.get("topic_clusters", {})
        if tc and not tc.get("error"):
            vesa_ctx["topic_clusters"] = {
                "topics_found": tc.get("topics_found", 0),
                "topics": [
                    {
                        "label": t["label"], "ticket_count": t["ticket_count"],
                        "pct": t["pct_of_total"],
                        "trend_direction": t.get("trend_direction", "unknown"),
                        "high_severity_pct": t.get("high_severity_pct", 0),
                        "avg_mttr": t.get("avg_mttr", 0),
                    }
                    for t in tc.get("topics", [])[:10]
                ],
            }
        st = vesa_results.get("similar_tickets", {})
        if st and not st.get("error"):
            vesa_ctx["similar_tickets"] = {
                "groups_found": st.get("similar_groups_found", 0),
                "problem_candidates": st.get("problem_candidates", 0),
                "grouped_pct": st.get("grouped_pct", 0),
                "top_groups": [
                    {
                        "ticket_count": g["ticket_count"],
                        "description": g.get("representative_description", "")[:100],
                        "is_problem_candidate": g.get("is_problem_candidate", False),
                        "total_hours": g.get("total_hours_spent", 0),
                    }
                    for g in st.get("groups", [])[:10]
                ],
            }
        mp = vesa_results.get("mttr_prediction", {})
        if mp and not mp.get("error"):
            vesa_ctx["mttr_prediction"] = {
                "overall_trend": mp.get("mttr_overall_trend", "unknown"),
                "trend_change_pct": mp.get("mttr_trend_change_pct", 0),
                "predictions_by_severity": mp.get("mttr_predictions_by_severity", []),
                "outlier_count": mp.get("outlier_count", 0),
            }
        ctx["vesa_analytics"] = vesa_ctx

    if "initial_analysis_result" in global_llm_context:
        ctx["eol_eos_analysis"] = global_llm_context["initial_analysis_result"]
    if "insights_analysis_result" in global_llm_context:
        ctx["business_insights"] = global_llm_context["insights_analysis_result"]
    for key in ["initial_analysis_result", "insights_analysis_result"]:
        if key in global_llm_context:
            ctx[key] = global_llm_context[key]

    global_llm_context = ctx


# =========================
# Full Analysis Pipeline
# =========================
def perform_full_analysis(file_path, industry=None, currency="USD"):
    global analysis_data, calculation_methods, vesa_results
    calculation_methods = {}

    df = load_data(file_path)
    df = preprocess_data(df)

    complete_analysis = perform_analysis_on_dataframe(df, "complete", industry, currency)

    managed_df, non_managed_df = split_managed_nonmanaged(df)

    def _safe_analysis(sub_df, atype):
        if len(sub_df) > 0:
            return perform_analysis_on_dataframe(sub_df, atype, industry, currency)
        return {"metadata": {"total_tickets": 0, "analysis_type": atype, "industry": industry, "currency": currency, "message": f"No {atype} tickets found"}}

    managed_analysis = _safe_analysis(managed_df, "managed")
    non_managed_analysis = _safe_analysis(non_managed_df, "non_managed")
    comparison_data = build_comparison_data(managed_df, non_managed_df)

    # Run VESA analytics
    try:
        vesa_results = perform_vesa_analysis(df)
    except Exception as e:
        print(f"VESA analysis error (non-fatal): {e}")
        vesa_results = {}

    analysis_data = {
        "complete_analysis": complete_analysis,
        "managed_analysis": managed_analysis,
        "non_managed_analysis": non_managed_analysis,
        "comparison_data": comparison_data,
        "dataframe": df,
        "managed_df": managed_df,
        "non_managed_df": non_managed_df,
        "industry": industry,
        "currency": currency,
    }

    _rebuild_global_llm_context()
    return analysis_data


# =========================
# Routes
# =========================

@app.route("/")
def index():
    here = os.path.dirname(os.path.abspath(__file__))
    accounts_path = os.path.join(here, "ou_list_stripped.json")
    accounts = []
    if os.path.exists(accounts_path):
        with open(accounts_path, encoding="utf-8") as f:
            accounts = json.load(f)
    return render_template("index.html", accounts=accounts)


@app.route("/industries", methods=["GET"])
def get_industries():
    industries = [
        {"id": k, "name": k.replace("_", " ").title(), "downtime_cost_per_hour_usd": v, "downtime_cost_per_hour_eur": round(v / EUR_TO_USD, 0)}
        for k, v in INDUSTRY_DOWNTIME_COSTS.items()
    ]
    return jsonify({"industries": industries})


@app.route("/upload_inventory", methods=["POST"])
def upload_inventory():
    global supplementary_data
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
        return jsonify({"error": "Invalid file type. Please upload CSV, XLSX, or XLS."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"inventory_{filename}")
    file.save(filepath)

    try:
        df_raw = load_data(filepath)
        cols_normalized = [c.replace("\ufeff", "").strip().lower().replace(' ', '').replace('_', '') for c in df_raw.columns]
        if not any(c in ['hostname', 'hostnames'] for c in cols_normalized):
            return jsonify({"error": "Invalid inventory file. Required column 'hostname' not found.", "columns_found": list(df_raw.columns)}), 400

        total_rows = len(df_raw)
        df_processed = preprocess_inventory(df_raw)
        supplementary_data["inventory"] = df_processed
        unique_hosts = int(df_processed["hostname"].nunique()) if "hostname" in df_processed.columns else 0
        return jsonify({
            "success": True,
            "message": f"{unique_hosts:,} unique servers identified",
            "row_count": total_rows,
            "unique_hosts": unique_hosts,
            "columns_kept": list(df_processed.columns),
        })
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": "Exception during inventory processing", "detail": str(e)}), 500


@app.route("/upload_app_mapping", methods=["POST"])
def upload_app_mapping():
    global supplementary_data
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
        return jsonify({"error": "Invalid file type."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"appmapping_{filename}")
    file.save(filepath)

    try:
        df_raw = load_data(filepath)
        cols_normalized = [c.replace("\ufeff", "").strip().lower() for c in df_raw.columns]
        if "product_name" not in cols_normalized:
            return jsonify({"error": "Invalid file. Required column 'product_name' not found.", "columns_found": list(df_raw.columns)}), 400

        total_rows = len(df_raw)
        df_processed = preprocess_app_mapping(df_raw)
        supplementary_data["app_mapping"] = df_processed
        return jsonify({
            "success": True,
            "message": f"{total_rows:,} business application mappings identified",
            "row_count": total_rows,
            "columns_kept": list(df_processed.columns),
        })
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": "Exception during app mapping processing", "detail": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        try:
            df_check = load_data(filepath)
            cols_normalized = [c.replace("\ufeff", "").strip().lower() for c in df_check.columns]
            if "incident_code_id" not in cols_normalized:
                return jsonify({"error": "Invalid incidents file. Required column 'incident_code_id' not found.", "columns_found": list(df_check.columns)}), 400

            total_rows = len(df_check)
            supplementary_data["incidents_filepath"] = filepath
            return jsonify({
                "success": True,
                "message": f"{total_rows:,} incidents ready",
                "row_count": total_rows,
            })
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            return jsonify({"error": "Exception during processing", "detail": str(e), "traceback": tb}), 500
    return jsonify({"error": "Invalid file type"}), 400


@app.route("/upload_actionable_insights", methods=["POST"])
def upload_actionable_insights():
    global supplementary_data, insights_v2_data
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
        return jsonify({"error": "Invalid file type"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"actionable_insights_{filename}")
    file.save(filepath)
    try:
        # Auto-detect: if Excel with multiple sheets matching A-/G- pattern → combined format
        is_combined = False
        if filepath.endswith((".xlsx", ".xls")):
            try:
                xls = pd.ExcelFile(filepath)
                sheet_names = xls.sheet_names
                ag_sheets = [s for s in sheet_names if re.match(r"[AG]-\d+", s)]
                if len(ag_sheets) >= 3:
                    is_combined = True
            except Exception:
                pass

        if is_combined:
            # New combined-insights format (summary + detail sheets per insight)
            supplementary_data["insights_combined_filepath"] = filepath
            # Auto-run the v2 analysis
            insights_v2_data = analyse_insights_file(filepath)
            summary = insights_v2_data.get("summary", {})
            total = summary.get("total_insights", 0)
            act_count = summary.get("actionable_count", 0)
            gro_count = summary.get("growth_count", 0)
            thr_count = summary.get("threatcon_count", 0)
            broader_cats = len(insights_v2_data.get("broader_categories", []))
            return jsonify({
                "success": True,
                "message": f"{total} insights loaded ({act_count} actionable · {gro_count} growth · {thr_count} threat) in {broader_cats} categories",
                "is_combined_format": True,
                "row_count": total,
                "unique_hosts": 0,
                "unique_insights": total,
            })
        else:
            # Legacy flat format
            df_raw = load_data(filepath)
            cols_norm = [c.replace("\ufeff", "").strip().lower() for c in df_raw.columns]
            has_insight = any("insight" in c and "id" in c for c in cols_norm)
            has_host = any("host" in c for c in cols_norm)
            if not has_insight or not has_host:
                return jsonify({"error": "Required columns (hostname, insight_id) not found", "columns_found": list(df_raw.columns)}), 400
            total_rows = len(df_raw)
            df_proc = preprocess_actionable_insights(df_raw)
            supplementary_data["actionable_insights"] = df_proc
            unique_hosts = int(df_proc["hostname"].nunique()) if "hostname" in df_proc.columns else 0
            unique_insights = int(df_proc["insight_id"].nunique()) if "insight_id" in df_proc.columns else 0
            return jsonify({
                "success": True,
                "message": f"{unique_hosts:,} devices · {unique_insights:,} unique insights loaded",
                "is_combined_format": False,
                "row_count": total_rows,
                "unique_hosts": unique_hosts,
                "unique_insights": unique_insights,
            })
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": "Exception during processing", "detail": str(e)}), 500


@app.route("/upload_insights_combined", methods=["POST"])
def upload_insights_combined():
    """Upload the new combined-insights Excel (summary + detail per insight)."""
    global supplementary_data
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
        return jsonify({"error": "Invalid file type"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"insights_combined_{filename}")
    file.save(filepath)
    try:
        xls = pd.ExcelFile(filepath)
        sheet_count = len(xls.sheet_names)
        supplementary_data["insights_combined_filepath"] = filepath
        # Also load into legacy format for backward compat
        return jsonify({
            "success": True,
            "message": f"{sheet_count} sheets loaded from combined insights file",
            "sheet_count": sheet_count,
        })
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": "Exception during processing", "detail": str(e)}), 500


@app.route("/insights_v2_analysis", methods=["POST"])
def run_insights_v2_analysis():
    """Run the new per-insight analysis engine."""
    global insights_v2_data
    filepath = supplementary_data.get("insights_combined_filepath")
    if not filepath:
        return jsonify({"error": "No combined insights file uploaded. Please upload the insights file first."}), 400
    try:
        insights_v2_data = analyse_insights_file(filepath)
        return jsonify(clean_for_json(insights_v2_data))
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": "Analysis failed", "detail": str(e)}), 500


@app.route("/insights_v2_results", methods=["GET"])
def get_insights_v2_results():
    """Return cached insights v2 data."""
    if not insights_v2_data:
        return jsonify({"error": "No insights analysis available. Run analysis first."}), 400
    return jsonify(clean_for_json(insights_v2_data))


@app.route("/analyse", methods=["POST"])
def run_analysis():
    filepath = supplementary_data.get("incidents_filepath")
    if not filepath:
        return jsonify({"error": "No incidents file uploaded."}), 400
    try:
        data = request.json or {}
        industry = data.get("industry")
        currency = data.get("currency", "USD")
        perform_full_analysis(filepath, industry=industry, currency=currency)
        response_data = {
            "complete_analysis": {k: v for k, v in analysis_data["complete_analysis"].items() if k != "dataframe"},
            "managed_analysis": {k: v for k, v in analysis_data["managed_analysis"].items() if k != "dataframe"},
            "non_managed_analysis": {k: v for k, v in analysis_data["non_managed_analysis"].items() if k != "dataframe"},
            "comparison_data": analysis_data["comparison_data"],
        }
        # Clean NaN values before JSON serialization
        response_data = clean_for_json(response_data)
        total = analysis_data["complete_analysis"]["metadata"]["total_tickets"]
        return jsonify({
            "success": True,
            "message": f"{total:,} incidents analysed",
            "analysis": response_data,
        })
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": "Exception during analysis", "detail": str(e), "traceback": tb}), 500


# =========================
# VESA Analytics Route
# =========================
@app.route("/vesa_analysis", methods=["GET"])
def get_vesa_analysis():
    """Return VESA analytics results (topic clusters, similar tickets, MTTR prediction)."""
    if not vesa_results:
        return jsonify({"error": "No VESA analysis available. Run incident analysis first."}), 400
    return jsonify({"success": True, "vesa": clean_for_json(vesa_results)})


# =========================
# Initial Analysis (EOL/EOS) — imported from original
# =========================
@app.route("/initial_analysis", methods=["POST"])
def initial_analysis():
    global global_llm_context

    inv_df = supplementary_data.get("inventory")
    app_df = supplementary_data.get("app_mapping")
    inc_filepath = supplementary_data.get("incidents_filepath")

    if inv_df is None:
        return jsonify({"error": "Inventory file not uploaded."}), 400

    result = {}
    inv = inv_df.copy()
    for col in ["eol_date", "eos_date"]:
        if col in inv.columns:
            parsed = pd.to_datetime(inv[col], errors="coerce", utc=True)
            inv[col] = parsed.dt.tz_convert(None)

    today = pd.Timestamp.now(tz=None).normalize()
    total_devices = len(inv)

    eol_devices = inv[inv["eol_date"].notna() & (inv["eol_date"] < today)] if "eol_date" in inv.columns else inv.iloc[0:0]
    eos_devices = inv[inv["eos_date"].notna() & (inv["eos_date"] < today)] if "eos_date" in inv.columns else inv.iloc[0:0]

    # Extended Support & EOES: Only devices that have ALREADY reached EOS can be in extended support.
    # Extended Support = devices past EOS but still within extended support window (EOES<1Y + EOES>1Y)
    # EOES = devices PAST extended support (status = "EOES")
    extended_support_devices = inv.iloc[0:0]  # empty default
    eoes_devices = inv.iloc[0:0]  # empty default (past extended support)
    if "eoes_status" in inv.columns:
        eoes_status_upper = inv["eoes_status"].fillna("").astype(str).str.upper()
        # Build EOS mask: device must have already reached EOS to qualify for extended support
        eos_mask_for_ext = pd.Series([False] * len(inv), index=inv.index)
        if "eos_date" in inv.columns:
            eos_mask_for_ext = inv["eos_date"].notna() & (inv["eos_date"] < today)
        # Extended Support = approaching extended support end AND already past EOS
        extended_support_devices = inv[eoes_status_upper.isin(["EOES<1Y", "EOES>1Y"]) & eos_mask_for_ext]
        # EOES = PAST extended support (status = "EOES") AND already past EOS
        eoes_devices = inv[(eoes_status_upper == "EOES") & eos_mask_for_ext]
    elif "end_of_extended_support_date" in inv.columns:
        parsed_eoes = pd.to_datetime(inv["end_of_extended_support_date"], errors="coerce", utc=True)
        inv["end_of_extended_support_date"] = parsed_eoes.dt.tz_convert(None) if hasattr(parsed_eoes.dt, 'tz_convert') else parsed_eoes
        one_year_from_now = today + pd.DateOffset(years=1)
        # Build EOS mask: device must have already reached EOS
        eos_mask_for_ext = pd.Series([False] * len(inv), index=inv.index)
        if "eos_date" in inv.columns:
            eos_mask_for_ext = inv["eos_date"].notna() & (inv["eos_date"] < today)
        # Extended Support = approaching (within 1 year) AND already past EOS
        extended_support_devices = inv[inv["end_of_extended_support_date"].notna() & (inv["end_of_extended_support_date"] < one_year_from_now) & (inv["end_of_extended_support_date"] >= today) & eos_mask_for_ext]
        # EOES = past extended support AND already past EOS
        eoes_devices = inv[inv["end_of_extended_support_date"].notna() & (inv["end_of_extended_support_date"] < today) & eos_mask_for_ext]

    eol_pct = round(len(eol_devices) / total_devices * 100, 1) if total_devices > 0 else 0
    eos_pct = round(len(eos_devices) / total_devices * 100, 1) if total_devices > 0 else 0
    extended_support_pct = round(len(extended_support_devices) / total_devices * 100, 1) if total_devices > 0 else 0
    eoes_pct = round(len(eoes_devices) / total_devices * 100, 1) if total_devices > 0 else 0

    result["total_devices"] = total_devices
    result["eol_count"] = len(eol_devices)
    result["eos_count"] = len(eos_devices)
    result["extended_support_count"] = len(extended_support_devices)
    result["eoes_count"] = len(eoes_devices)
    result["eol_percentage"] = eol_pct
    result["eos_percentage"] = eos_pct
    result["extended_support_percentage"] = extended_support_pct
    result["eoes_percentage"] = eoes_pct

    # Extended Support OS breakdown
    if len(extended_support_devices) > 0 and "osname" in extended_support_devices.columns:
        ext_supp_os = extended_support_devices["osname"].fillna("(unknown)").value_counts().head(10)
        result["extended_support_by_os"] = [{"os": str(k), "count": int(v)} for k, v in ext_supp_os.items()]
    else:
        result["extended_support_by_os"] = []

    # EOES OS breakdown
    if len(eoes_devices) > 0 and "osname" in eoes_devices.columns:
        eoes_os = eoes_devices["osname"].fillna("(unknown)").value_counts().head(10)
        result["eoes_by_os"] = [{"os": str(k), "count": int(v)} for k, v in eoes_os.items()]
    else:
        result["eoes_by_os"] = []

    one_year_ago = today - pd.DateOffset(years=1)
    three_years_ago = today - pd.DateOffset(years=3)

    eol_gt1yr = inv[inv["eol_date"].notna() & (inv["eol_date"] < one_year_ago)] if "eol_date" in inv.columns else inv.iloc[0:0]
    eol_gt3yr = inv[inv["eol_date"].notna() & (inv["eol_date"] < three_years_ago)] if "eol_date" in inv.columns else inv.iloc[0:0]

    eol_gt1yr_hostnames = set(eol_gt1yr["hostname"].dropna().str.lower()) if "hostname" in eol_gt1yr.columns else set()
    eol_gt3yr_hostnames = set(eol_gt3yr["hostname"].dropna().str.lower()) if "hostname" in eol_gt3yr.columns else set()
    eol_hostnames = set(eol_devices["hostname"].dropna().str.lower()) if "hostname" in eol_devices.columns else set()
    eos_hostnames = set(eos_devices["hostname"].dropna().str.lower()) if "hostname" in eos_devices.columns else set()
    extended_support_hostnames = set(extended_support_devices["hostname"].dropna().str.lower()) if "hostname" in extended_support_devices.columns else set()
    eoes_hostnames = set(eoes_devices["hostname"].dropna().str.lower()) if "hostname" in eoes_devices.columns else set()

    app_analysis = {}
    if app_df is not None:
        app_m = app_df.copy()
        if "hostname" in app_m.columns and "product_name" in app_m.columns:
            app_m["hostname_lower"] = app_m["hostname"].str.lower()
            for prod, grp in app_m.groupby("product_name"):
                prod_hosts = set(grp["hostname_lower"].dropna())
                total_prod = len(prod_hosts)
                if total_prod == 0:
                    continue
                eol_c = len(prod_hosts & eol_hostnames)
                eos_c = len(prod_hosts & eos_hostnames)
                app_analysis[str(prod)] = {
                    "total_servers": total_prod,
                    "eol_count": eol_c,
                    "eos_count": eos_c,
                    "eol_percentage": round(eol_c / total_prod * 100, 1),
                    "eos_percentage": round(eos_c / total_prod * 100, 1),
                    "eol_gt1yr_count": len(prod_hosts & eol_gt1yr_hostnames),
                    "eol_gt3yr_count": len(prod_hosts & eol_gt3yr_hostnames),
                }

    def top_n(d, key, n=10):
        return sorted(d.items(), key=lambda x: x[1][key], reverse=True)[:n]

    result["apps_by_eol_count"] = [{"app": k, **v} for k, v in top_n(app_analysis, "eol_count", 10)] if app_analysis else []
    result["apps_over_30pct_eol_or_eos"] = [
        {"app": k, **v} for k, v in app_analysis.items()
        if v["eol_percentage"] >= 30 or v["eos_percentage"] >= 30
    ] if app_analysis else []
    result["apps_eol_gt1yr"] = [{"app": k, **v} for k, v in top_n(app_analysis, "eol_gt1yr_count", 10) if v["eol_gt1yr_count"] > 0] if app_analysis else []
    result["apps_eol_gt3yr"] = [{"app": k, **v} for k, v in top_n(app_analysis, "eol_gt3yr_count", 10) if v["eol_gt3yr_count"] > 0] if app_analysis else []

    has_incidents = False
    if inc_filepath:
        try:
            inc_df_raw = load_data(inc_filepath)
            inc_df = preprocess_data(inc_df_raw)
            if "hostname" in inc_df.columns:
                has_incidents = True
                total_incidents = len(inc_df)
                inc_df["hostname_lower"] = inc_df["hostname"].str.lower()
                host_counts = inc_df["hostname_lower"].value_counts()
                threshold_10pct = total_incidents * 0.10
                result["high_volume_hosts"] = [
                    {"hostname": h, "incident_count": int(c), "pct": round(c / total_incidents * 100, 1)}
                    for h, c in host_counts.items() if c >= threshold_10pct
                ]
                if "priority_df" in inc_df.columns:
                    sev12_df = inc_df[inc_df["priority_df"].isin([1, 2])]
                    sev12_counts = sev12_df["hostname_lower"].value_counts().head(10)
                    result["top_sev12_hosts"] = [{"hostname": h, "sev12_count": int(c)} for h, c in sev12_counts.items()]
                else:
                    result["top_sev12_hosts"] = []

                if "mttr_excl_hold" in inc_df.columns:
                    grp = inc_df.dropna(subset=["hostname_lower", "mttr_excl_hold"]).groupby("hostname_lower")["mttr_excl_hold"].agg(["mean", "count"])
                    grp = grp[grp["count"] >= 3]
                    peer_avg = float(grp["mean"].mean()) if len(grp) > 0 else 0
                    above_avg = grp[grp["mean"] > peer_avg].sort_values("mean", ascending=False).head(10)
                    result["high_mttr_hosts"] = [
                        {"hostname": h, "avg_mttr": round(float(r["mean"]), 2), "incident_count": int(r["count"]),
                         "pct_above_avg": round((float(r["mean"]) - peer_avg) / peer_avg * 100, 1) if peer_avg > 0 else 0}
                        for h, r in above_avg.iterrows()
                    ]
                    result["peer_avg_mttr"] = round(peer_avg, 2)
                else:
                    result["high_mttr_hosts"] = []
                    result["peer_avg_mttr"] = 0

                result["total_incidents"] = total_incidents

                # EOL/EOS MTTR Comparison
                eol_or_eos_hostnames = eol_hostnames | eos_hostnames
                all_inv_hostnames = set(inv["hostname"].dropna().str.lower()) if "hostname" in inv.columns else set()
                healthy_hostnames = all_inv_hostnames - eol_or_eos_hostnames - extended_support_hostnames - eoes_hostnames

                if "mttr_excl_hold" in inc_df.columns and len(eol_or_eos_hostnames) > 0:
                    inc_with_mttr = inc_df.dropna(subset=["hostname_lower", "mttr_excl_hold"]).copy()
                    eol_incidents = inc_with_mttr[inc_with_mttr["hostname_lower"].isin(eol_or_eos_hostnames)]
                    healthy_incidents = inc_with_mttr[inc_with_mttr["hostname_lower"].isin(healthy_hostnames)]

                    # Individual MTTR for EOL-only, EOS-only, Extended Support, EOES
                    eol_only_incidents = inc_with_mttr[inc_with_mttr["hostname_lower"].isin(eol_hostnames)]
                    eos_only_incidents = inc_with_mttr[inc_with_mttr["hostname_lower"].isin(eos_hostnames)]
                    extended_support_incidents = inc_with_mttr[inc_with_mttr["hostname_lower"].isin(extended_support_hostnames)]
                    eoes_only_incidents = inc_with_mttr[inc_with_mttr["hostname_lower"].isin(eoes_hostnames)]

                    eol_only_avg_mttr = float(eol_only_incidents["mttr_excl_hold"].mean()) if len(eol_only_incidents) > 0 else 0
                    eos_only_avg_mttr = float(eos_only_incidents["mttr_excl_hold"].mean()) if len(eos_only_incidents) > 0 else 0
                    extended_support_avg_mttr = float(extended_support_incidents["mttr_excl_hold"].mean()) if len(extended_support_incidents) > 0 else 0
                    eoes_only_avg_mttr = float(eoes_only_incidents["mttr_excl_hold"].mean()) if len(eoes_only_incidents) > 0 else 0

                    eol_avg_mttr = float(eol_incidents["mttr_excl_hold"].mean()) if len(eol_incidents) > 0 else 0
                    healthy_avg_mttr = float(healthy_incidents["mttr_excl_hold"].mean()) if len(healthy_incidents) > 0 else 0
                    overall_avg_mttr = float(inc_with_mttr["mttr_excl_hold"].mean()) if len(inc_with_mttr) > 0 else 0

                    mttr_pct_higher = round(((eol_avg_mttr - healthy_avg_mttr) / healthy_avg_mttr * 100), 1) if healthy_avg_mttr > 0 else 0
                    eol_mttr_is_worse = eol_avg_mttr > healthy_avg_mttr
                    eol_incident_count = len(eol_incidents)
                    healthy_incident_count = len(healthy_incidents)
                    eol_total_hours = float(eol_incidents["mttr_excl_hold"].sum())
                    healthy_total_hours = float(healthy_incidents["mttr_excl_hold"].sum())
                    excess_hours = max(0, eol_total_hours - (eol_incident_count * healthy_avg_mttr)) if healthy_avg_mttr > 0 else 0

                    months = get_months_in_data(inc_df)
                    projected_yearly_eol_hours = round(eol_total_hours / months * 12, 2) if months > 0 else 0
                    projected_yearly_excess_hours = round(excess_hours / months * 12, 2) if months > 0 else 0

                    eol_sev12 = eol_incidents[eol_incidents["priority_df"].isin([1, 2])] if "priority_df" in eol_incidents.columns else eol_incidents.iloc[0:0]
                    eol_sev34 = eol_incidents[eol_incidents["priority_df"].isin([3, 4])] if "priority_df" in eol_incidents.columns else eol_incidents.iloc[0:0]
                    eol_sev12_hours = float(eol_sev12["mttr_excl_hold"].sum()) if len(eol_sev12) > 0 else 0
                    eol_sev34_hours = float(eol_sev34["mttr_excl_hold"].sum()) if len(eol_sev34) > 0 else 0
                    projected_yearly_eol_sev12_hours = round(eol_sev12_hours / months * 12, 2) if months > 0 else 0
                    projected_yearly_eol_sev34_hours = round(eol_sev34_hours / months * 12, 2) if months > 0 else 0

                    current_industry = analysis_data.get("industry") if analysis_data else None
                    current_currency = analysis_data.get("currency", "USD") if analysis_data else "USD"
                    industry_rate = convert_currency(INDUSTRY_DOWNTIME_COSTS.get(current_industry, 0), current_currency) if current_industry else 0
                    noise_rate = convert_currency(NOISE_COST_PER_HOUR, current_currency)

                    eol_downtime_cost = round(projected_yearly_eol_sev12_hours * industry_rate, 0) if industry_rate else None
                    eol_noise_cost = round(projected_yearly_eol_sev34_hours * noise_rate, 0)
                    eol_total_cost = round((eol_downtime_cost or 0) + eol_noise_cost, 0)
                    modernization_savings = round(eol_total_cost * 0.40, 0) if eol_total_cost else None
                    excess_downtime_cost = round(projected_yearly_excess_hours * industry_rate * 0.6, 0) if industry_rate and projected_yearly_excess_hours else None
                    excess_noise_cost = round(projected_yearly_excess_hours * noise_rate * 0.4, 0)

                    # LLM summary for EOL risk
                    eol_eos_llm_summary = ""
                    if AZURE_CONFIGURED and eol_incident_count > 0:
                        try:
                            desc_col = "description" if "description" in eol_incidents.columns else None
                            desc_sample = []
                            if desc_col:
                                desc_sample = eol_incidents[desc_col].dropna().value_counts().head(10).to_dict()
                                desc_sample = [{"description": str(k)[:120], "count": int(v)} for k, v in desc_sample.items()]

                            eol_sev_dist_str = ""
                            if "priority_df" in eol_incidents.columns:
                                sd = eol_incidents["priority_df"].value_counts().sort_index()
                                eol_sev_dist_str = ", ".join([f"Sev {int(k)}: {int(v)}" for k, v in sd.items() if not pd.isna(k)])

                            llm_prompt = (
                                f"You are an IT infrastructure analyst. Analyze these incident patterns from EOL/EOS servers.\n\n"
                                f"Total EOL/EOS incidents: {eol_incident_count}\n"
                                f"Severity distribution: {eol_sev_dist_str}\n"
                                f"EOL/EOS Avg MTTR: {round(eol_avg_mttr, 2)}h vs Healthy Avg MTTR: {round(healthy_avg_mttr, 2)}h\n"
                                f"Total EOL/EOS hours lost: {round(eol_total_hours, 1)}h\n\n"
                                f"Top incident descriptions on EOL/EOS devices:\n{json.dumps(desc_sample, indent=1)}\n\n"
                                "Provide a detailed analysis (5-7 sentences): main issue types, why more likely on EOL infrastructure, "
                                "which could be prevented with modernization, and operational risk of not modernizing."
                            )
                            eol_eos_llm_summary = generate_with_azure(llm_prompt, max_output_tokens=800, temperature=0.2)
                        except Exception as e:
                            print(f"EOL/EOS LLM summary error: {e}")

                    # Build MTTR chart data for all categories
                    mttr_chart_data = {
                        "categories": [],
                        "avg_mttr": [],
                        "incident_count": [],
                        "total_hours": [],
                        "worse_than_healthy": []
                    }
                    
                    # Always include Healthy baseline
                    mttr_chart_data["categories"].append("Healthy")
                    mttr_chart_data["avg_mttr"].append(round(healthy_avg_mttr, 2))
                    mttr_chart_data["incident_count"].append(healthy_incident_count)
                    mttr_chart_data["total_hours"].append(round(healthy_total_hours, 2))
                    mttr_chart_data["worse_than_healthy"].append(False)
                    
                    # Add EOL
                    if len(eol_only_incidents) > 0:
                        mttr_chart_data["categories"].append("EOL")
                        mttr_chart_data["avg_mttr"].append(round(eol_only_avg_mttr, 2))
                        mttr_chart_data["incident_count"].append(len(eol_only_incidents))
                        mttr_chart_data["total_hours"].append(round(float(eol_only_incidents["mttr_excl_hold"].sum()), 2))
                        mttr_chart_data["worse_than_healthy"].append(eol_only_avg_mttr > healthy_avg_mttr)
                    
                    # Add EOS
                    if len(eos_only_incidents) > 0:
                        mttr_chart_data["categories"].append("EOS")
                        mttr_chart_data["avg_mttr"].append(round(eos_only_avg_mttr, 2))
                        mttr_chart_data["incident_count"].append(len(eos_only_incidents))
                        mttr_chart_data["total_hours"].append(round(float(eos_only_incidents["mttr_excl_hold"].sum()), 2))
                        mttr_chart_data["worse_than_healthy"].append(eos_only_avg_mttr > healthy_avg_mttr)
                    
                    # Add Extended Support
                    if len(extended_support_incidents) > 0:
                        mttr_chart_data["categories"].append("Extended Support")
                        mttr_chart_data["avg_mttr"].append(round(extended_support_avg_mttr, 2))
                        mttr_chart_data["incident_count"].append(len(extended_support_incidents))
                        mttr_chart_data["total_hours"].append(round(float(extended_support_incidents["mttr_excl_hold"].sum()), 2))
                        mttr_chart_data["worse_than_healthy"].append(extended_support_avg_mttr > healthy_avg_mttr)
                    
                    # Add EOES (past extended support)
                    if len(eoes_only_incidents) > 0:
                        mttr_chart_data["categories"].append("EOES")
                        mttr_chart_data["avg_mttr"].append(round(eoes_only_avg_mttr, 2))
                        mttr_chart_data["incident_count"].append(len(eoes_only_incidents))
                        mttr_chart_data["total_hours"].append(round(float(eoes_only_incidents["mttr_excl_hold"].sum()), 2))
                        mttr_chart_data["worse_than_healthy"].append(eoes_only_avg_mttr > healthy_avg_mttr)

                    result["eol_eos_risk_impact"] = {
                        "eol_eos_avg_mttr": round(eol_avg_mttr, 2),
                        "healthy_avg_mttr": round(healthy_avg_mttr, 2),
                        "overall_avg_mttr": round(overall_avg_mttr, 2),
                        "mttr_pct_higher": mttr_pct_higher,
                        "eol_eos_incident_count": eol_incident_count,
                        "healthy_incident_count": healthy_incident_count,
                        "eol_eos_total_hours": round(eol_total_hours, 2),
                        "healthy_total_hours": round(healthy_total_hours, 2),
                        "excess_hours_due_to_eol": round(excess_hours, 2),
                        "projected_yearly_eol_hours": projected_yearly_eol_hours,
                        "projected_yearly_excess_hours": projected_yearly_excess_hours,
                        "eol_sev12_incidents": len(eol_sev12),
                        "eol_sev34_incidents": len(eol_sev34),
                        "projected_yearly_eol_sev12_hours": projected_yearly_eol_sev12_hours,
                        "projected_yearly_eol_sev34_hours": projected_yearly_eol_sev34_hours,
                        "eol_downtime_cost": eol_downtime_cost,
                        "eol_noise_cost": eol_noise_cost,
                        "eol_total_cost": eol_total_cost,
                        "modernization_savings": modernization_savings,
                        "excess_downtime_cost": excess_downtime_cost,
                        "excess_noise_cost": excess_noise_cost,
                        "eol_mttr_is_worse": eol_mttr_is_worse,
                        "eol_eos_llm_summary": eol_eos_llm_summary,
                        # Individual MTTR by lifecycle status
                        "eol_only_avg_mttr": round(eol_only_avg_mttr, 2),
                        "eos_only_avg_mttr": round(eos_only_avg_mttr, 2),
                        "extended_support_avg_mttr": round(extended_support_avg_mttr, 2),
                        "eoes_only_avg_mttr": round(eoes_only_avg_mttr, 2),
                        "eol_only_incident_count": len(eol_only_incidents),
                        "eos_only_incident_count": len(eos_only_incidents),
                        "extended_support_incident_count": len(extended_support_incidents),
                        "eoes_only_incident_count": len(eoes_only_incidents),
                        "eol_only_total_hours": round(float(eol_only_incidents["mttr_excl_hold"].sum()), 2) if len(eol_only_incidents) > 0 else 0,
                        "eos_only_total_hours": round(float(eos_only_incidents["mttr_excl_hold"].sum()), 2) if len(eos_only_incidents) > 0 else 0,
                        "extended_support_total_hours": round(float(extended_support_incidents["mttr_excl_hold"].sum()), 2) if len(extended_support_incidents) > 0 else 0,
                        "eoes_only_total_hours": round(float(eoes_only_incidents["mttr_excl_hold"].sum()), 2) if len(eoes_only_incidents) > 0 else 0,
                        # MTTR chart data for frontend graphs
                        "mttr_chart_data": mttr_chart_data,
                    }
        except Exception as e:
            print(f"Incident correlation error: {e}")

    result["has_incidents"] = has_incidents
    result["has_app_mapping"] = app_df is not None

    eol_eos_for_llm = {
        "total_devices": total_devices,
        "eol_count": len(eol_devices), "eos_count": len(eos_devices), 
        "extended_support_count": len(extended_support_devices), "eoes_count": len(eoes_devices),
        "eol_percentage": eol_pct, "eos_percentage": eos_pct, 
        "extended_support_percentage": extended_support_pct, "eoes_percentage": eoes_pct,
        "eol_gt1yr_count": len(eol_gt1yr), "eol_gt3yr_count": len(eol_gt3yr),
        "apps_by_eol_count": result.get("apps_by_eol_count", [])[:10],
        "apps_over_30pct_eol_or_eos": result.get("apps_over_30pct_eol_or_eos", [])[:10],
        "high_volume_hosts": result.get("high_volume_hosts", []),
        "top_sev12_hosts": result.get("top_sev12_hosts", []),
        "high_mttr_hosts": result.get("high_mttr_hosts", []),
        "peer_avg_mttr": result.get("peer_avg_mttr", 0),
        "total_incidents": result.get("total_incidents", 0),
        "has_incidents": has_incidents,
        "has_app_mapping": app_df is not None,
        "eol_eos_risk_impact": result.get("eol_eos_risk_impact", {}),
    }
    global_llm_context["initial_analysis_result"] = eol_eos_for_llm
    _rebuild_global_llm_context()

    # Add OS breakdown data for frontend charts
    # Use _process_inventory_for_eol_eos to get broader_categories and detailed_os_data
    try:
        inv_eol_data = _process_inventory_for_eol_eos(inv_df)
        result["broader_categories"] = inv_eol_data.get("broader_categories", {})
        result["detailed_os_data"] = inv_eol_data.get("detailed_os_data", [])
        result["extended_support_by_osname"] = inv_eol_data.get("extended_support_by_osname", [])
    except Exception as e:
        print(f"Error getting OS breakdown: {e}")
        result["broader_categories"] = {}
        result["detailed_os_data"] = []
        result["extended_support_by_osname"] = []

    return jsonify({"success": True, "result": clean_for_json(result)})


# =========================
# Insights Analysis
# =========================
@app.route("/insights_analysis", methods=["POST"])
def insights_analysis():
    global global_llm_context

    # If combined v2 data is already available (from auto-detection at upload), use it directly
    if insights_v2_data and insights_v2_data.get("summary", {}).get("total_insights", 0) > 0:
        # Return the v2 data directly — frontend will detect 'is_v2' flag
        v2_result = clean_for_json(insights_v2_data)
        v2_result["is_v2"] = True
        v2_result["success"] = True
        return jsonify({"success": True, "result": v2_result, "is_v2": True})

    # Otherwise fall back to legacy insights analysis
    insights_df = supplementary_data.get("actionable_insights")
    app_df = supplementary_data.get("app_mapping")
    inv_df = supplementary_data.get("inventory")
    inc_filepath = supplementary_data.get("incidents_filepath")
    
    if insights_df is None:
        return jsonify({"error": "Actionable Insights file must be uploaded."}), 400

    result = {}
    summary_lookup = {}
    
    # Load incidents data if available for correlation
    inc_df = None
    if inc_filepath:
        try:
            inc_df_raw = load_data(inc_filepath)
            inc_df = preprocess_data(inc_df_raw)
            if "hostname" in inc_df.columns:
                inc_df["hostname_lower"] = inc_df["hostname"].str.lower()
        except Exception as e:
            print(f"Could not load incidents for correlation: {e}")
    
    # Build EOL/EOS host sets for correlation
    eol_hostnames = set()
    eos_hostnames = set()
    if inv_df is not None and "hostname" in inv_df.columns:
        inv = inv_df.copy()
        for col in ["eol_date", "eos_date"]:
            if col in inv.columns:
                parsed = pd.to_datetime(inv[col], errors="coerce", utc=True)
                inv[col] = parsed.dt.tz_convert(None) if hasattr(parsed.dt, 'tz_convert') else parsed
        today = pd.Timestamp.now(tz=None).normalize()
        if "eol_date" in inv.columns:
            eol_devices = inv[inv["eol_date"].notna() & (inv["eol_date"] < today)]
            eol_hostnames = set(eol_devices["hostname"].dropna().str.lower())
        if "eos_date" in inv.columns:
            eos_devices = inv[inv["eos_date"].notna() & (inv["eos_date"] < today)]
            eos_hostnames = set(eos_devices["hostname"].dropna().str.lower())
    
    # Process insights
    if "insight_id" in insights_df.columns:
        for _, row in insights_df.drop_duplicates(subset=["insight_id"]).iterrows():
            iid = row.get("insight_id")
            if pd.notna(iid):
                observation_text = str(row.get("observation", "")) if pd.notna(row.get("observation")) else ""
                recommendation_text = str(row.get("recommendation", "")) if pd.notna(row.get("recommendation")) else ""
                action_text = str(row.get("action", "")) if pd.notna(row.get("action")) else ""
                insight_category = str(row.get("insight_category", "")) if pd.notna(row.get("insight_category")) else ""
                insight_title = str(row.get("insight_title", "")) if pd.notna(row.get("insight_title")) else ""
                
                # Parse rec_sections from recommendation text
                rec_sections = {}
                for section_name in ["Situation", "Problem", "Recommendation"]:
                    pattern = rf'{section_name}\s+(.*?)(?=(?:Situation|Problem|Recommendation|Recommended Actions|APM Solutions|$))'
                    match = re.search(pattern, recommendation_text, re.DOTALL | re.IGNORECASE)
                    if match:
                        rec_sections[section_name.lower()] = match.group(1).strip()[:500]
                
                # Extract key metrics from observation
                obs_numbers = re.findall(r'(\d+(?:,\d+)*)\s+(?:tickets?|servers?|devices?|incidents?)', observation_text, re.IGNORECASE)
                obs_key_metrics = [int(n.replace(',', '')) for n in obs_numbers[:10]]
                
                # Extract top items mentioned
                obs_top_items = re.findall(r'([A-Za-z][A-Za-z0-9\s\(\)\-\.\/]+?)\s*\((\d+(?:,\d+)*)\s+tickets?\)', observation_text)
                obs_top_items_list = [{"name": name.strip(), "tickets": int(count.replace(',', ''))} for name, count in obs_top_items[:20]]
                
                # Extract symptoms
                obs_symptoms = re.findall(r'Top Symptoms?\s*:?\s*(.*?)(?=Top|Observations|$)', observation_text, re.DOTALL | re.IGNORECASE)
                symptoms_list = []
                if obs_symptoms:
                    symptom_matches = re.findall(r'([a-zA-Z\s]+?)\s*\((\d+)\s+tickets?\)', obs_symptoms[0])
                    symptoms_list = [{"symptom": s.strip(), "tickets": int(c)} for s, c in symptom_matches]

                # Determine insight type category for correlation
                insight_type_tags = []
                title_lower = insight_title.lower()
                obs_lower = observation_text.lower()
                rec_lower = recommendation_text.lower()
                
                if any(x in title_lower or x in obs_lower for x in ['eol', 'end of life', 'end-of-life', 'lifecycle']):
                    insight_type_tags.append('eol_related')
                if any(x in title_lower or x in obs_lower for x in ['eos', 'end of service', 'end-of-service', 'support']):
                    insight_type_tags.append('eos_related')
                if any(x in title_lower or x in obs_lower for x in ['incident', 'ticket', 'outage', 'failure', 'mttr']):
                    insight_type_tags.append('incident_related')
                if any(x in title_lower or x in obs_lower for x in ['change', 'deployment', 'patch', 'update']):
                    insight_type_tags.append('change_related')
                if any(x in title_lower or x in obs_lower for x in ['security', 'vulnerab', 'risk', 'compliance']):
                    insight_type_tags.append('security_related')
                if any(x in title_lower or x in obs_lower for x in ['performance', 'cpu', 'memory', 'disk', 'latency']):
                    insight_type_tags.append('performance_related')
                if any(x in title_lower or x in obs_lower for x in ['capacity', 'storage', 'growth', 'utilization']):
                    insight_type_tags.append('capacity_related')
                if any(x in title_lower or x in obs_lower for x in ['monitor', 'alert', 'observ']):
                    insight_type_tags.append('monitoring_related')
                
                summary_lookup[int(iid)] = {
                    "insight_id": int(iid),
                    "insight_title": insight_title,
                    "action": action_text,
                    "is_growth": int(iid) < 1000,
                    "observation_summary": observation_text[:1000],
                    "recommendation_summary": recommendation_text[:1000],
                    "rec_sections": rec_sections,
                    "key_metrics": obs_key_metrics,
                    "top_items_mentioned": obs_top_items_list,
                    "symptoms": symptoms_list,
                    "insight_category": insight_category,
                    "insight_type_tags": insight_type_tags,
                }

    growth_count = sum(1 for v in summary_lookup.values() if v["is_growth"])
    actionable_count = sum(1 for v in summary_lookup.values() if not v["is_growth"])
    result["growth_insight_count"] = growth_count
    result["actionable_insight_count"] = actionable_count
    result["total_insight_types"] = len(summary_lookup)

    # Build host insights with correlation data
    host_insights = {}
    if "hostname" in insights_df.columns and "insight_id" in insights_df.columns:
        dev_clean = insights_df.dropna(subset=["hostname", "insight_id"]).copy()
        dev_clean["hostname_lower"] = dev_clean["hostname"].str.lower()
        dev_clean["insight_id"] = dev_clean["insight_id"].astype(int)
        
        for host, grp in dev_clean.groupby("hostname_lower"):
            unique_ids = sorted(grp["insight_id"].unique().tolist())
            insights_detail = [summary_lookup[iid] for iid in unique_ids if iid in summary_lookup]
            
            # Calculate correlation data
            is_eol = host in eol_hostnames
            is_eos = host in eos_hostnames
            incident_count = 0
            sev12_count = 0
            total_mttr = 0
            
            if inc_df is not None and "hostname_lower" in inc_df.columns:
                host_incidents = inc_df[inc_df["hostname_lower"] == host]
                incident_count = len(host_incidents)
                if "priority_df" in host_incidents.columns:
                    sev12_count = len(host_incidents[host_incidents["priority_df"].isin([1, 2])])
                if "mttr_excl_hold" in host_incidents.columns:
                    total_mttr = float(host_incidents["mttr_excl_hold"].sum())
            
            host_insights[host] = {
                "hostname": host,
                "insight_ids": unique_ids,
                "insight_count": len(unique_ids),
                "growth_count": sum(1 for i in unique_ids if i < 1000),
                "actionable_count": sum(1 for i in unique_ids if i >= 1000),
                "insights": insights_detail,
                "is_eol": is_eol,
                "is_eos": is_eos,
                "incident_count": incident_count,
                "sev12_count": sev12_count,
                "total_mttr_hours": round(total_mttr, 2),
            }

    sorted_hosts = sorted(host_insights.values(), key=lambda x: x["insight_count"], reverse=True)
    result["hosts_with_insights"] = sorted_hosts
    result["total_hosts_with_insights"] = len(sorted_hosts)
    
    # Calculate aggregate correlation stats
    eol_hosts_with_insights = [h for h in sorted_hosts if h["is_eol"]]
    eos_hosts_with_insights = [h for h in sorted_hosts if h["is_eos"]]
    high_incident_hosts = [h for h in sorted_hosts if h["incident_count"] >= 10]
    
    result["correlation_summary"] = {
        "eol_hosts_with_insights": len(eol_hosts_with_insights),
        "eos_hosts_with_insights": len(eos_hosts_with_insights),
        "high_incident_hosts_with_insights": len(high_incident_hosts),
        "total_incidents_on_insight_hosts": sum(h["incident_count"] for h in sorted_hosts),
        "total_sev12_on_insight_hosts": sum(h["sev12_count"] for h in sorted_hosts),
        "total_mttr_hours_on_insight_hosts": round(sum(h["total_mttr_hours"] for h in sorted_hosts), 2),
    }

    # App insights with correlation
    result["has_app_mapping"] = app_df is not None
    app_insights = {}
    if app_df is not None and "hostname" in app_df.columns and "product_name" in app_df.columns:
        app_map_lower = app_df.copy()
        app_map_lower["hostname_lower"] = app_map_lower["hostname"].str.lower()
        
        for prod, grp in app_map_lower.groupby("product_name"):
            prod_hosts = set(grp["hostname_lower"].dropna())
            matched_hosts = prod_hosts & set(host_insights.keys())
            if not matched_hosts:
                continue
            
            all_ids = set()
            growth_ids = set()
            actionable_ids = set()
            eol_count = 0
            eos_count = 0
            total_incidents = 0
            total_sev12 = 0
            total_mttr = 0
            
            for h in matched_hosts:
                host_data = host_insights[h]
                for iid in host_data["insight_ids"]:
                    all_ids.add(iid)
                    if iid < 1000:
                        growth_ids.add(iid)
                    else:
                        actionable_ids.add(iid)
                if host_data["is_eol"]:
                    eol_count += 1
                if host_data["is_eos"]:
                    eos_count += 1
                total_incidents += host_data["incident_count"]
                total_sev12 += host_data["sev12_count"]
                total_mttr += host_data["total_mttr_hours"]
            
            insights_detail = [summary_lookup[i] for i in sorted(all_ids) if i in summary_lookup]
            app_insights[str(prod)] = {
                "app": str(prod),
                "total_hosts": len(prod_hosts),
                "hosts_with_insights": len(matched_hosts),
                "unique_insight_count": len(all_ids),
                "growth_insight_count": len(growth_ids),
                "actionable_insight_count": len(actionable_ids),
                "insights": insights_detail,
                "eol_hosts": eol_count,
                "eos_hosts": eos_count,
                "total_incidents": total_incidents,
                "total_sev12": total_sev12,
                "total_mttr_hours": round(total_mttr, 2),
            }

    sorted_apps = sorted(app_insights.values(), key=lambda x: x["unique_insight_count"], reverse=True)
    result["apps_with_insights"] = sorted_apps
    result["total_apps_with_insights"] = len(sorted_apps)

    # Category counts
    all_assigned_ids = set()
    for h in host_insights.values():
        all_assigned_ids.update(h["insight_ids"])
    cat_counts = {}
    for iid in all_assigned_ids:
        if iid in summary_lookup:
            cat = summary_lookup[iid].get("insight_title", "Unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    result["insights_by_category"] = sorted(
        [{"category": k, "count": v} for k, v in cat_counts.items()],
        key=lambda x: x["count"], reverse=True
    )

    # Build enriched insight cards with impact calculation
    enriched_insights = []
    current_industry = analysis_data.get("industry") if analysis_data else None
    current_currency = analysis_data.get("currency", "USD") if analysis_data else "USD"
    industry_rate = convert_currency(INDUSTRY_DOWNTIME_COSTS.get(current_industry, 0), current_currency) if current_industry else 0
    noise_rate = convert_currency(NOISE_COST_PER_HOUR, current_currency)
    
    for iid, insight_data in summary_lookup.items():
        # Find all hosts affected by this insight
        affected_hosts = [h for h in sorted_hosts if iid in h["insight_ids"]]
        host_count = len(affected_hosts)
        
        # Calculate impact metrics
        total_incidents_affected = sum(h["incident_count"] for h in affected_hosts)
        total_sev12_affected = sum(h["sev12_count"] for h in affected_hosts)
        total_mttr_affected = sum(h["total_mttr_hours"] for h in affected_hosts)
        eol_count = sum(1 for h in affected_hosts if h["is_eol"])
        eos_count = sum(1 for h in affected_hosts if h["is_eos"])
        
        # Calculate estimated cost impact
        # Hard facts: based on actual data
        hard_downtime_hours = 0
        hard_noise_hours = 0
        if inc_df is not None:
            for h in affected_hosts:
                host_incidents = inc_df[inc_df["hostname_lower"] == h["hostname"]] if "hostname_lower" in inc_df.columns else pd.DataFrame()
                if "priority_df" in host_incidents.columns and "mttr_excl_hold" in host_incidents.columns:
                    sev12_mttr = host_incidents[host_incidents["priority_df"].isin([1, 2])]["mttr_excl_hold"].sum()
                    sev34_mttr = host_incidents[host_incidents["priority_df"].isin([3, 4])]["mttr_excl_hold"].sum()
                    hard_downtime_hours += float(sev12_mttr) if pd.notna(sev12_mttr) else 0
                    hard_noise_hours += float(sev34_mttr) if pd.notna(sev34_mttr) else 0
        
        hard_downtime_cost = round(hard_downtime_hours * industry_rate, 0) if industry_rate else None
        hard_noise_cost = round(hard_noise_hours * noise_rate, 0)
        hard_total_cost = (hard_downtime_cost or 0) + hard_noise_cost
        
        # LLM-induced estimates (conservative projections based on patterns)
        # Assume addressing insight could reduce incidents by 20-40%
        llm_estimated_savings_low = round(hard_total_cost * 0.20, 0) if hard_total_cost > 0 else None
        llm_estimated_savings_high = round(hard_total_cost * 0.40, 0) if hard_total_cost > 0 else None
        
        enriched_insight = {
            **insight_data,
            "affected_host_count": host_count,
            "affected_hosts_sample": [h["hostname"] for h in affected_hosts[:10]],
            "total_incidents_affected": total_incidents_affected,
            "total_sev12_affected": total_sev12_affected,
            "total_mttr_hours_affected": round(total_mttr_affected, 2),
            "eol_hosts_affected": eol_count,
            "eos_hosts_affected": eos_count,
            "impact_metrics": {
                "hard_facts": {
                    "downtime_hours": round(hard_downtime_hours, 2),
                    "noise_hours": round(hard_noise_hours, 2),
                    "downtime_cost": hard_downtime_cost,
                    "noise_cost": hard_noise_cost,
                    "total_cost": hard_total_cost,
                    "source": "calculated_from_incident_data",
                },
                "llm_induced": {
                    "estimated_savings_low": llm_estimated_savings_low,
                    "estimated_savings_high": llm_estimated_savings_high,
                    "savings_assumption": "20-40% reduction if insight addressed",
                    "source": "projected_estimate",
                },
            },
        }
        enriched_insights.append(enriched_insight)
    
    # Sort enriched insights by impact
    enriched_insights.sort(key=lambda x: x["impact_metrics"]["hard_facts"]["total_cost"] or 0, reverse=True)
    result["enriched_insights"] = enriched_insights

    # Build LLM context
    insights_for_llm = {
        "growth_insight_count": result["growth_insight_count"],
        "actionable_insight_count": result["actionable_insight_count"],
        "total_insight_types": result["total_insight_types"],
        "total_hosts_with_insights": result["total_hosts_with_insights"],
        "total_apps_with_insights": result["total_apps_with_insights"],
        "correlation_summary": result["correlation_summary"],
        "insights_by_category": result["insights_by_category"][:15],
        "top_hosts_by_insight_count": [
            {
                "hostname": h["hostname"], "insight_count": h["insight_count"],
                "growth": h["growth_count"], "actionable": h["actionable_count"],
                "is_eol": h["is_eol"], "is_eos": h["is_eos"],
                "incident_count": h["incident_count"], "sev12_count": h["sev12_count"],
            }
            for h in sorted_hosts[:20]
        ],
        "top_apps_by_insight_count": [
            {
                "app": a["app"], "total_hosts": a["total_hosts"],
                "hosts_with_insights": a["hosts_with_insights"],
                "unique_insights": a["unique_insight_count"],
                "growth": a["growth_insight_count"], "actionable": a["actionable_insight_count"],
                "eol_hosts": a["eol_hosts"], "eos_hosts": a["eos_hosts"],
                "total_incidents": a["total_incidents"],
            }
            for a in sorted_apps[:20]
        ],
        "enriched_insights_top": [
            {
                "insight_id": e["insight_id"], "insight_title": e["insight_title"],
                "action": e["action"], "is_growth": e["is_growth"],
                "affected_host_count": e["affected_host_count"],
                "total_incidents_affected": e["total_incidents_affected"],
                "impact_metrics": e["impact_metrics"],
                "insight_type_tags": e.get("insight_type_tags", []),
                "observation_summary": e.get("observation_summary", "")[:400],
                "rec_sections": e.get("rec_sections", {}),
            }
            for e in enriched_insights[:20]
        ],
    }
    
    all_symptoms = {}
    for v in summary_lookup.values():
        for s in v.get("symptoms", []):
            all_symptoms[s["symptom"]] = all_symptoms.get(s["symptom"], 0) + s["tickets"]
    insights_for_llm["all_symptoms_aggregated"] = sorted(
        [{"symptom": k, "tickets": v} for k, v in all_symptoms.items()], key=lambda x: x["tickets"], reverse=True
    )

    global_llm_context["insights_analysis_result"] = insights_for_llm
    _rebuild_global_llm_context()

    return jsonify({"success": True, "result": result})


# =========================
# Recalculate, Visualizations, Ask, Executive Summary, Export
# =========================

@app.route("/recalculate", methods=["POST"])
def recalculate_costs():
    if not analysis_data or "dataframe" not in analysis_data:
        return jsonify({"error": "No data available."}), 400
    data = request.json or {}
    industry = data.get("industry")
    currency = data.get("currency", "USD")
    if industry and industry not in INDUSTRY_DOWNTIME_COSTS:
        return jsonify({"error": f"Invalid industry: {industry}"}), 400
    if currency not in ["USD", "EUR"]:
        return jsonify({"error": "Currency must be USD or EUR"}), 400
    try:
        df = analysis_data["dataframe"]
        managed_df = analysis_data.get("managed_df", pd.DataFrame())
        non_managed_df = analysis_data.get("non_managed_df", pd.DataFrame())
        complete_analysis = perform_analysis_on_dataframe(df, "complete", industry, currency)
        def _safe(sub_df, atype):
            if len(sub_df) > 0:
                return perform_analysis_on_dataframe(sub_df, atype, industry, currency)
            return {"metadata": {"total_tickets": 0, "analysis_type": atype}}
        managed_analysis = _safe(managed_df, "managed")
        non_managed_analysis = _safe(non_managed_df, "non_managed")
        analysis_data["complete_analysis"] = complete_analysis
        analysis_data["managed_analysis"] = managed_analysis
        analysis_data["non_managed_analysis"] = non_managed_analysis
        analysis_data["industry"] = industry
        analysis_data["currency"] = currency
        _rebuild_global_llm_context()
        response_data = {
            "complete_analysis": {k: v for k, v in complete_analysis.items() if k != "dataframe"},
            "managed_analysis": {k: v for k, v in managed_analysis.items() if k != "dataframe"},
            "non_managed_analysis": {k: v for k, v in non_managed_analysis.items() if k != "dataframe"},
            "comparison_data": analysis_data["comparison_data"],
        }
        industry_name = industry.replace('_', ' ').title() if industry else "None"
        return jsonify({"success": True, "message": f"Costs recalculated for {industry_name} in {currency}", "industry": industry, "currency": currency, "analysis": response_data})
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": "Exception during recalculation", "detail": str(e), "traceback": tb}), 500


@app.route("/visualizations")
def get_visualizations():
    if not analysis_data:
        return jsonify({"error": "No data analyzed yet"}), 400
    analysis_type = request.args.get("type", "complete")
    return jsonify(create_visualizations(analysis_data, analysis_type))


@app.route("/comparison_visualizations")
def get_comparison_visualizations():
    if not analysis_data:
        return jsonify({"error": "No data analyzed yet"}), 400
    return jsonify(create_comparison_visualizations(analysis_data))


@app.route("/ask", methods=["POST"])
def ask_llm():
    data = request.json or {}
    question = (data.get("question") or "").strip()
    analysis_type = (data.get("analysis_type") or "complete").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not global_llm_context and (not analysis_data or "dataframe" not in analysis_data):
        return jsonify({"error": "No data available."}), 400
    try:
        if analysis_data and "dataframe" in analysis_data:
            df = analysis_data["dataframe"]
            det = try_answer_deterministically(question, df, analysis_type)
            if det is not None:
                return jsonify({"success": True, "answer": det, "provider": "deterministic", "analysis_type": analysis_type})

        current_industry = analysis_data.get("industry") if analysis_data else None
        current_currency = analysis_data.get("currency", "USD") if analysis_data else "USD"
        currency_symbol = "€" if current_currency == "EUR" else "$"
        context_obj = dict(global_llm_context)
        context_obj["analysis_scope"] = analysis_type
        context_obj["currency"] = current_currency
        context_obj["currency_symbol"] = currency_symbol
        context_obj["industry"] = current_industry

        if analysis_data and analysis_type in ["managed", "non_managed"]:
            key_map = {"managed": "managed_analysis", "non_managed": "non_managed_analysis"}
            scoped = analysis_data.get(key_map.get(analysis_type), {})
            if scoped and scoped.get("metadata", {}).get("total_tickets", 0) > 0:
                context_obj["scoped_incident_data"] = {
                    "metadata": scoped.get("metadata", {}),
                    "severity_analysis": scoped.get("severity_analysis", {}),
                    "automation_opportunity": scoped.get("automation_opportunity", {}),
                    "llm_context": scoped.get("llm_context", {}),
                }

        context_str = json.dumps(context_obj, indent=1, default=str)
        if len(context_str) > 80000:
            for trim_key in ["all_apps_costs_top20", "common_issue_phrases", "description_keywords", "resolution_keywords"]:
                for parent_key in ["incident_analysis", "insights_analysis_result"]:
                    if parent_key in context_obj and trim_key in context_obj[parent_key]:
                        if isinstance(context_obj[parent_key][trim_key], list):
                            context_obj[parent_key][trim_key] = context_obj[parent_key][trim_key][:5]
            context_str = json.dumps(context_obj, indent=1, default=str)

        prompt = (
            f"Analysis scope: {analysis_type.upper()}\nCurrency: {current_currency} ({currency_symbol})\n"
            f"Industry: {current_industry or 'not set'}\n\n"
            f"COMPLETE ANALYSIS CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {question}\n\n"
            "Provide a comprehensive, detailed answer."
        )
        generated = generate_with_azure(prompt, max_output_tokens=1500, temperature=0.1)
        return jsonify({"success": True, "answer": generated, "provider": "azure_openai", "deployment": AZURE_OPENAI_DEPLOYMENT, "analysis_type": analysis_type, "industry": current_industry, "currency": current_currency})
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        msg = str(e)
        if "429" in msg or "rate limit" in msg.lower():
            return jsonify({"error": "Rate limit exceeded.", "detail": msg}), 429
        return jsonify({"error": "Exception during LLM processing", "detail": msg, "traceback": tb}), 500


@app.route("/generate_executive_summary", methods=["POST"])
def generate_executive_summary():
    global latest_exec_summary
    if not global_llm_context:
        return jsonify({"error": "No analysis data available."}), 400
    try:
        req_data = request.json or {}
        user_prompt = (req_data.get("user_prompt") or "").strip()
        previous_summary = req_data.get("previous_summary")
        current_industry = analysis_data.get("industry") if analysis_data else None
        current_currency = analysis_data.get("currency", "USD") if analysis_data else "USD"
        currency_symbol = "€" if current_currency == "EUR" else "$"
        context_obj = dict(global_llm_context)
        context_obj["currency"] = current_currency
        context_obj["currency_symbol"] = currency_symbol
        context_obj["industry"] = current_industry
        context_str = json.dumps(context_obj, indent=1, default=str)
        if len(context_str) > 70000:
            for trim_key in ["common_issue_phrases", "description_keywords", "resolution_keywords"]:
                for parent_key in ["incident_analysis", "insights_analysis_result"]:
                    if parent_key in context_obj and trim_key in context_obj[parent_key]:
                        if isinstance(context_obj[parent_key][trim_key], list):
                            context_obj[parent_key][trim_key] = context_obj[parent_key][trim_key][:5]
            context_str = json.dumps(context_obj, indent=1, default=str)

        prompt = f"Currency: {current_currency} ({currency_symbol})\nIndustry: {current_industry or 'not set'}\n\nCOMPLETE ANALYSIS DATA:\n{context_str}\n\n"
        if previous_summary and user_prompt:
            prompt += f"PREVIOUS EXECUTIVE SUMMARY:\n{json.dumps(previous_summary, indent=1)}\n\nUSER REQUEST: {user_prompt}\n\nModify the executive summary. Respond with FULL updated summary in JSON."
        elif user_prompt:
            prompt += f"USER INSTRUCTIONS: {user_prompt}\n\nGenerate the Executive Summary. Respond ONLY with valid JSON."
        else:
            prompt += "Generate the Executive Summary with exactly 3 observations from DIFFERENT categories.\nRespond ONLY with valid JSON."

        generated = generate_with_azure(prompt, max_output_tokens=3500, temperature=0.25, system_prompt=SYSTEM_PROMPT_EXEC_SUMMARY)
        clean = generated.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            return jsonify({"success": True, "raw_response": generated, "parsed": False, "provider": "azure_openai"})

        latest_exec_summary = parsed
        return jsonify({"success": True, "executive_summary": parsed, "parsed": True, "provider": "azure_openai", "deployment": AZURE_OPENAI_DEPLOYMENT, "industry": current_industry, "currency": current_currency})
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        msg = str(e)
        if "429" in msg or "rate limit" in msg.lower():
            return jsonify({"error": "Rate limit exceeded.", "detail": msg}), 429
        return jsonify({"error": "Exception generating executive summary", "detail": msg}), 500


@app.route("/export_findings_csv", methods=["GET"])
def export_findings_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    current_currency = analysis_data.get("currency", "USD") if analysis_data else "USD"
    current_industry = analysis_data.get("industry") if analysis_data else None
    currency_symbol = "€" if current_currency == "EUR" else "$"

    writer.writerow(["=" * 80])
    writer.writerow(["KYNDRYL BRIDGE INTELLIGENCE - COMPLETE FINDINGS EXPORT"])
    writer.writerow(["=" * 80])
    writer.writerow([])
    writer.writerow(["Export Date", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow(["Currency", current_currency])
    writer.writerow(["Industry", (current_industry or "Not Set").replace("_", " ").title()])
    writer.writerow([])

    # Executive summary
    exec_obs = (latest_exec_summary or {}).get("observations", [])
    if exec_obs:
        writer.writerow(["EXECUTIVE SUMMARY"])
        writer.writerow(["#", "Category", "Key Metric", "Title", "Finding", "Recommendation"])
        for i, obs in enumerate(exec_obs, 1):
            writer.writerow([i, obs.get("category", ""), obs.get("metric", ""), obs.get("title", ""), obs.get("explanation", ""), obs.get("recommendation", "")])
        writer.writerow([])

    # Incident overview
    ca = analysis_data.get("complete_analysis", {}) if analysis_data else {}
    if ca and ca.get("metadata", {}).get("total_tickets", 0) > 0:
        meta = ca.get("metadata", {})
        writer.writerow(["INCIDENT ANALYSIS OVERVIEW"])
        writer.writerow(["Total Tickets", meta.get("total_tickets", 0)])
        writer.writerow(["Months of Data", meta.get("months_of_data", 0)])
        writer.writerow([])

    # VESA analytics summary
    if vesa_results:
        writer.writerow(["VESA ANALYTICS"])
        tc = vesa_results.get("topic_clusters", {})
        if tc and not tc.get("error"):
            writer.writerow(["Topic Clusters Found", tc.get("topics_found", 0)])
            for t in tc.get("topics", [])[:10]:
                writer.writerow(["", t["label"], f"{t['ticket_count']} tickets", f"{t['pct_of_total']}%", t.get("trend_direction", "")])
        st = vesa_results.get("similar_tickets", {})
        if st and not st.get("error"):
            writer.writerow(["Similar Ticket Groups", st.get("similar_groups_found", 0)])
            writer.writerow(["Problem Candidates", st.get("problem_candidates", 0)])
        writer.writerow([])

    writer.writerow(["END OF EXPORT"])
    output.seek(0)
    return Response(output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=kyndryl_bridge_findings_export.csv"})


# =========================
# Change Management Routes
# =========================

def _compute_change_metrics(df, closed_mask, failed_mask):
    """Compute change metrics for a given dataframe subset."""
    total = len(df)
    if total == 0:
        return None
    
    closed_count = int(closed_mask.sum())
    failed_count = int(failed_mask.sum())
    canceled_count = int((df["status_cd"] == "CANCELED").sum()) if "status_cd" in df.columns else 0
    cfr = round(failed_count / closed_count * 100, 2) if closed_count > 0 else 0.0

    avg_lead_time = round(float(df["lead_time_hrs"].mean()), 1) if df["lead_time_hrs"].notna().any() else 0.0
    avg_duration = round(float(df["duration_hrs"].mean()), 1) if df["duration_hrs"].notna().any() else 0.0

    # Monthly
    monthly = {}
    if "_month" in df.columns:
        monthly = df.groupby("_month").size().sort_index().to_dict()
        monthly = {k: int(v) for k, v in monthly.items()}

    # Value counts helpers
    def vc(col, n=None):
        if col not in df.columns:
            return {}
        s = df[col].value_counts()
        if n:
            s = s.head(n)
        return {str(k): int(v) for k, v in s.items()}

    def vc_closed(col, n=None):
        if col not in df.columns:
            return {}
        s = df[closed_mask][col].value_counts()
        if n:
            s = s.head(n)
        return {str(k): int(v) for k, v in s.items()}

    # Avg duration by risk
    dur_by_risk = {}
    if "risk_code_cd" in df.columns and df["duration_hrs"].notna().any():
        dur_by_risk = {
            str(k): round(float(v), 1)
            for k, v in df.groupby("risk_code_cd")["duration_hrs"].mean().items()
        }

    # Canceled by group
    canceled_group = {}
    if "status_cd" in df.columns and "assignee_group_cd" in df.columns:
        canceled_group = {
            str(k): int(v)
            for k, v in df[df["status_cd"] == "CANCELED"]["assignee_group_cd"]
            .value_counts().head(10).items()
        }

    # Failed change records
    failed_records = []
    if failed_count > 0:
        failed_df = df[failed_mask]
        for _, row in failed_df.iterrows():
            failed_records.append({
                "change_id": str(row.get("change_id", "")),
                "assignee_group_cd": str(row.get("assignee_group_cd", "")),
                "risk_code_cd": str(row.get("risk_code_cd", "")),
                "category_cd": str(row.get("category_cd", "")),
                "cmpltn_code_cd": str(row.get("cmpltn_code_cd", "")),
                "status_cd": str(row.get("status_cd", "")),
            })

    # ── High Risk & Very High Risk Changes (using risk_code_cd) ──
    high_risk_metrics = {}
    if "risk_code_cd" in df.columns:
        risk_upper = df["risk_code_cd"].fillna("").astype(str).str.upper().str.strip()
        high_risk_mask = risk_upper.isin(["HIGH", "VERY HIGH"])
        high_risk_total = int(high_risk_mask.sum())
        high_risk_closed = int((high_risk_mask & closed_mask).sum())
        high_risk_failed = int((high_risk_mask & failed_mask).sum())
        high_risk_cfr = round(high_risk_failed / high_risk_closed * 100, 2) if high_risk_closed > 0 else 0.0
        high_risk_metrics = {
            "total": high_risk_total,
            "closed": high_risk_closed,
            "failed": high_risk_failed,
            "cfr": high_risk_cfr,
            "successful": high_risk_closed - high_risk_failed,
            "pct_of_total": round(high_risk_total / total * 100, 1) if total > 0 else 0.0,
        }
        # Also break down by individual risk level
        for risk_level in ["HIGH", "VERY HIGH"]:
            rl_mask = risk_upper == risk_level
            rl_total = int(rl_mask.sum())
            rl_closed = int((rl_mask & closed_mask).sum())
            rl_failed = int((rl_mask & failed_mask).sum())
            rl_cfr = round(rl_failed / rl_closed * 100, 2) if rl_closed > 0 else 0.0
            high_risk_metrics[risk_level.lower().replace(" ", "_")] = {
                "total": rl_total,
                "closed": rl_closed,
                "failed": rl_failed,
                "cfr": rl_cfr,
            }

    # ── Emergency Changes (using change_type_cd) ──
    emergency_metrics = {}
    if "change_type_cd" in df.columns:
        change_type_upper = df["change_type_cd"].fillna("").astype(str).str.upper().str.strip()
        emergency_mask = change_type_upper.isin(["EMERGENCY", "EMERGENCY CHANGE"])
        emergency_total = int(emergency_mask.sum())
        emergency_closed = int((emergency_mask & closed_mask).sum())
        emergency_failed = int((emergency_mask & failed_mask).sum())
        emergency_cfr = round(emergency_failed / emergency_closed * 100, 2) if emergency_closed > 0 else 0.0
        emergency_metrics = {
            "total": emergency_total,
            "closed": emergency_closed,
            "failed": emergency_failed,
            "cfr": emergency_cfr,
            "successful": emergency_closed - emergency_failed,
            "pct_of_total": round(emergency_total / total * 100, 1) if total > 0 else 0.0,
        }
        # Failed emergency records for detail display
        emergency_failed_records = []
        if emergency_failed > 0:
            em_failed_df = df[emergency_mask & failed_mask]
            for _, row in em_failed_df.iterrows():
                emergency_failed_records.append({
                    "change_id": str(row.get("change_id", "")),
                    "assignee_group_cd": str(row.get("assignee_group_cd", "")),
                    "risk_code_cd": str(row.get("risk_code_cd", "")),
                    "category_cd": str(row.get("category_cd", "")),
                    "cmpltn_code_cd": str(row.get("cmpltn_code_cd", "")),
                })
        emergency_metrics["failed_records"] = emergency_failed_records

    return {
        "total": total,
        "closed_count": closed_count,
        "failed_count": failed_count,
        "canceled_count": canceled_count,
        "cfr": cfr,
        "successful_count": closed_count - failed_count,
        "avg_lead_time": avg_lead_time,
        "avg_duration": avg_duration,
        "approved_count": int((df["approval_cd"] == "Approved").sum()) if "approval_cd" in df.columns else 0,
        "rejected_count": int((df["approval_cd"] == "Rejected").sum()) if "approval_cd" in df.columns else 0,
        "status": vc("status_cd"),
        "dv_cat": vc("dv_category_l1"),
        "risk": vc("risk_code_cd"),
        "category": vc("category_cd", 15),
        "group": vc("assignee_group_cd", 15),
        "cmpltn": vc_closed("cmpltn_code_cd"),
        "approval": vc("approval_cd"),
        "monthly": monthly,
        "canceled_group": canceled_group,
        "dur_by_risk": dur_by_risk,
        "failed_records": failed_records,
        "high_risk_metrics": high_risk_metrics,
        "emergency_metrics": emergency_metrics,
    }


def _process_change_data(df):
    """Process change data and compute all metrics for overall, managed, and non-managed."""
    df = df.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip().str.lower()

    # Parse datetime columns
    for col in ["request_dttm", "scheduled_dttm", "act_st_dttm", "act_finish_dttm", "closed_dttm"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # --- Closed / Failed logic ---
    def is_closed(s):
        return str(s).lower().startswith("close")

    def is_failed_cmpltn(c):
        c = str(c).upper().strip()
        prefixes = ["BACKED OUT", "BACKOUT", "BAKOUT", "UNAUTH"]
        exact = {
            "INCOMPLETE", "OVER RAN", "OVERRAN", "ISSUES", "UNSUCCESSFUL",
            "CUSTOMER CAUSED FAILURE", "INSTALLED WITH ISSUES",
        }
        for p in prefixes:
            if c.startswith(p):
                return True
        return c in exact

    closed_mask = df["status_cd"].apply(is_closed) if "status_cd" in df.columns else pd.Series([False] * len(df))
    failed_mask = closed_mask & (df["cmpltn_code_cd"].apply(is_failed_cmpltn) if "cmpltn_code_cd" in df.columns else pd.Series([False] * len(df)))

    # Duration / lead time
    if "act_st_dttm" in df.columns and "request_dttm" in df.columns:
        df["lead_time_hrs"] = (df["act_st_dttm"] - df["request_dttm"]).dt.total_seconds() / 3600
    else:
        df["lead_time_hrs"] = float("nan")

    if "act_finish_dttm" in df.columns and "act_st_dttm" in df.columns:
        df["duration_hrs"] = (df["act_finish_dttm"] - df["act_st_dttm"]).dt.total_seconds() / 3600
    else:
        df["duration_hrs"] = float("nan")

    # Monthly period column
    if "request_dttm" in df.columns:
        df["_month"] = df["request_dttm"].dt.tz_convert(None).dt.to_period("M").astype(str)

    # Compute overall analysis
    overall_analysis = _compute_change_metrics(df, closed_mask, failed_mask)

    # Check for sso_ticket column for managed/non-managed split
    managed_analysis = None
    non_managed_analysis = None
    has_sso_split = False

    if "sso_ticket" in df.columns:
        managed_mask = df["sso_ticket"].fillna("").astype(str).str.upper() == "Y"
        non_managed_mask = df["sso_ticket"].fillna("").astype(str).str.upper() == "N"
        
        if managed_mask.any() or non_managed_mask.any():
            has_sso_split = True
            
            # Managed subset
            if managed_mask.any():
                df_managed = df[managed_mask].copy()
                closed_managed = closed_mask[managed_mask]
                failed_managed = failed_mask[managed_mask]
                managed_analysis = _compute_change_metrics(df_managed, closed_managed, failed_managed)
            
            # Non-managed subset
            if non_managed_mask.any():
                df_non_managed = df[non_managed_mask].copy()
                closed_non_managed = closed_mask[non_managed_mask]
                failed_non_managed = failed_mask[non_managed_mask]
                non_managed_analysis = _compute_change_metrics(df_non_managed, closed_non_managed, failed_non_managed)

    return {
        "overall_analysis": overall_analysis,
        "managed_analysis": managed_analysis,
        "non_managed_analysis": non_managed_analysis,
        "has_sso_split": has_sso_split,
    }


@app.route("/upload_change_data", methods=["POST"])
def upload_change_data():
    global supplementary_data
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
        return jsonify({"error": "Invalid file type. Please upload CSV, XLSX, or XLS."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"change_{filename}")
    file.save(filepath)

    try:
        df_raw = load_data(filepath)
        cols_norm = [c.replace("\ufeff", "").strip().lower() for c in df_raw.columns]
        if "change_id" not in cols_norm and "status_cd" not in cols_norm:
            return jsonify({
                "error": "Invalid change data file. Expected columns like change_id, status_cd not found.",
                "columns_found": list(df_raw.columns),
            }), 400
        total_rows = len(df_raw)
        supplementary_data["change_filepath"] = filepath
        return jsonify({
            "success": True,
            "message": f"{total_rows:,} change records ready",
            "row_count": total_rows,
        })
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": "Exception during change data processing", "detail": str(e)}), 500


@app.route("/change_analysis", methods=["POST"])
def run_change_analysis():
    global change_data
    filepath = supplementary_data.get("change_filepath")
    if not filepath:
        return jsonify({"error": "No change data file uploaded."}), 400
    try:
        df = load_data(filepath)
        result = _process_change_data(df)
        change_data = result
        return jsonify({"success": True, "result": clean_for_json(result)})
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": "Exception during change analysis", "detail": str(e), "traceback": tb}), 500


@app.route("/change_analysis_results", methods=["GET"])
def get_change_analysis():
    if not change_data:
        return jsonify({"error": "No change analysis available. Upload and analyse change data first."}), 400
    return jsonify({"success": True, "result": clean_for_json(change_data)})


# =========================
# Service Request Routes
# =========================

def _compute_sr_metrics(df):
    """Compute service request metrics for a given dataframe subset."""
    total = len(df)
    if total == 0:
        return None
    
    # PAC Keywords analysis (Recommended Actions)
    pac_keywords_counts = {}
    if "pac_keywords" in df.columns:
        pac_keywords_counts = df["pac_keywords"].value_counts().to_dict()
        pac_keywords_counts = {str(k): int(v) for k, v in pac_keywords_counts.items() if pd.notna(k)}
    
    # Category analysis (Action Categories)
    category_counts = {}
    if "category" in df.columns:
        category_counts = df["category"].value_counts().to_dict()
        category_counts = {str(k): int(v) for k, v in category_counts.items() if pd.notna(k)}
    
    # Parse pac_keywords into verb and noun components for additional insights
    verb_counts = {}
    noun_counts = {}
    if "verb" in df.columns:
        verb_counts = df["verb"].value_counts().head(20).to_dict()
        verb_counts = {str(k): int(v) for k, v in verb_counts.items() if pd.notna(k)}
    if "noun" in df.columns:
        noun_counts = df["noun"].value_counts().head(20).to_dict()
        noun_counts = {str(k): int(v) for k, v in noun_counts.items() if pd.notna(k)}
    
    # Cross-tabulation: pac_keywords by category
    pac_by_category = {}
    if "pac_keywords" in df.columns and "category" in df.columns:
        cross_tab = df.groupby(["category", "pac_keywords"]).size().reset_index(name="count")
        for cat in df["category"].dropna().unique():
            cat_data = cross_tab[cross_tab["category"] == cat].nlargest(10, "count")
            pac_by_category[str(cat)] = [
                {"action": str(row["pac_keywords"]), "count": int(row["count"])}
                for _, row in cat_data.iterrows()
            ]
    
    # Monthly trend
    monthly = {}
    if "open_dttm" in df.columns and "_month" in df.columns:
        monthly = df.groupby("_month").size().sort_index().to_dict()
        monthly = {k: int(v) for k, v in monthly.items()}
    
    # Severity distribution
    severity_counts = {}
    if "severity" in df.columns:
        severity_counts = df["severity"].value_counts().to_dict()
        severity_counts = {str(k): int(v) for k, v in severity_counts.items()}
    
    # Status distribution
    status_counts = {}
    if "status" in df.columns:
        status_counts = df["status"].value_counts().to_dict()
        status_counts = {str(k): int(v) for k, v in status_counts.items()}
    
    # Top resolver groups
    resolver_groups = {}
    if "owner_group" in df.columns:
        resolver_groups = df["owner_group"].value_counts().head(15).to_dict()
        resolver_groups = {str(k): int(v) for k, v in resolver_groups.items() if pd.notna(k)}
    
    # Resolution time stats
    resolution_stats = {}
    if "resolution_time" in df.columns:
        rt = pd.to_numeric(df["resolution_time"], errors="coerce")
        resolution_stats = {
            "mean": round(float(rt.mean()), 2) if rt.notna().any() else 0,
            "median": round(float(rt.median()), 2) if rt.notna().any() else 0,
            "p90": round(float(rt.quantile(0.9)), 2) if rt.notna().any() else 0,
        }
    
    # Automation opportunity estimation (tickets with valid pac_keywords)
    automatable = 0
    if "pac_keywords" in df.columns:
        actionable_keywords = df["pac_keywords"].dropna()
        automatable = len(actionable_keywords)
    
    automation_pct = round(automatable / total * 100, 1) if total > 0 else 0

    # Cost savings: 20 min per ticket, annualised, × rate range
    sr_cost_savings = None
    if automatable > 0:
        sr_months = 1
        if "open_dttm" in df.columns:
            sr_months = get_months_in_data(df)
        elif len(monthly) > 0:
            sr_months = max(1, len(monthly))
        sr_hours_in_period = automatable * (20 / 60)
        sr_annual_hours = round(sr_hours_in_period / sr_months * 12, 2)
        sr_cost_savings = {
            "hours_in_period": round(sr_hours_in_period, 2),
            "annual_hours_saved": sr_annual_hours,
            "cost_savings_low": round(sr_annual_hours * 8.7, 0),
            "cost_savings_high": round(sr_annual_hours * 20.7, 0),
            "rate_low": 8.7,
            "rate_high": 20.7,
            "months_of_data": sr_months,
        }

    return {
        "total": total,
        "automation_opportunity_count": automatable,
        "automation_opportunity_pct": automation_pct,
        "automation_cost_savings": sr_cost_savings,
        "pac_keywords": pac_keywords_counts,
        "categories": category_counts,
        "verbs": verb_counts,
        "nouns": noun_counts,
        "pac_by_category": pac_by_category,
        "monthly": monthly,
        "severity": severity_counts,
        "status": status_counts,
        "resolver_groups": resolver_groups,
        "resolution_stats": resolution_stats,
    }


def _process_service_request_data(df):
    """Process service request data and compute automation potential metrics for overall, managed, and non-managed."""
    df = df.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip().str.lower()
    
    # Pre-process datetime and month column
    if "open_dttm" in df.columns:
        df["open_dttm"] = pd.to_datetime(df["open_dttm"], errors="coerce", utc=True)
        df["_month"] = df["open_dttm"].dt.tz_convert(None).dt.to_period("M").astype(str)
    
    # Compute overall analysis
    overall_analysis = _compute_sr_metrics(df)
    
    # Check for sso_ticket column for managed/non-managed split
    managed_analysis = None
    non_managed_analysis = None
    has_sso_split = False
    
    if "sso_ticket" in df.columns:
        managed_mask = df["sso_ticket"].fillna("").astype(str).str.upper() == "Y"
        non_managed_mask = df["sso_ticket"].fillna("").astype(str).str.upper() == "N"
        
        if managed_mask.any() or non_managed_mask.any():
            has_sso_split = True
            
            # Managed subset
            if managed_mask.any():
                df_managed = df[managed_mask].copy()
                managed_analysis = _compute_sr_metrics(df_managed)
            
            # Non-managed subset
            if non_managed_mask.any():
                df_non_managed = df[non_managed_mask].copy()
                non_managed_analysis = _compute_sr_metrics(df_non_managed)
    
    return {
        "overall_analysis": overall_analysis,
        "managed_analysis": managed_analysis,
        "non_managed_analysis": non_managed_analysis,
        "has_sso_split": has_sso_split,
        # For backward compatibility, also include overall data at root level
        **overall_analysis,
    }


@app.route("/upload_service_requests", methods=["POST"])
def upload_service_requests():
    global supplementary_data
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
        return jsonify({"error": "Invalid file type. Please upload CSV, XLSX, or XLS."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"service_requests_{filename}")
    file.save(filepath)

    try:
        df_raw = load_data(filepath)
        cols_norm = [c.replace("\ufeff", "").strip().lower() for c in df_raw.columns]
        # Check for pac_keywords or category columns
        has_pac = any("pac_keywords" in c for c in cols_norm)
        has_cat = any("category" in c for c in cols_norm)
        if not (has_pac or has_cat):
            return jsonify({
                "error": "Invalid service request file. Expected columns like pac_keywords or category not found.",
                "columns_found": list(df_raw.columns)[:20],
            }), 400
        total_rows = len(df_raw)
        supplementary_data["service_requests_filepath"] = filepath
        return jsonify({
            "success": True,
            "message": f"{total_rows:,} service requests ready",
            "row_count": total_rows,
        })
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": "Exception during service request processing", "detail": str(e)}), 500


@app.route("/service_request_analysis", methods=["POST"])
def run_service_request_analysis():
    global service_request_data
    filepath = supplementary_data.get("service_requests_filepath")
    if not filepath:
        return jsonify({"error": "No service request file uploaded."}), 400
    try:
        df = load_data(filepath)
        result = _process_service_request_data(df)
        service_request_data = result
        return jsonify({"success": True, "result": clean_for_json(result)})
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": "Exception during service request analysis", "detail": str(e), "traceback": tb}), 500


@app.route("/service_request_results", methods=["GET"])
def get_service_request_results():
    if not service_request_data:
        return jsonify({"error": "No service request analysis available. Upload and analyse service request data first."}), 400
    return jsonify({"success": True, "result": clean_for_json(service_request_data)})


# =========================
# Performance Metrics Routes
# =========================
performance_metrics_data = None

@app.route("/upload_performance_metrics", methods=["POST"])
def upload_performance_metrics():
    """Upload performance metrics file."""
    global performance_metrics_data
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
        return jsonify({"error": "Invalid file type. Please upload CSV, XLSX, or XLS."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"performance_{filename}")
    file.save(filepath)

    try:
        df_raw = load_data(filepath)
        df_raw.columns = df_raw.columns.str.strip()
        
        # Check for required columns
        required_cols = ["Date", "TIP", "Depth%", "ILMT%", "CIT%", "TAP%", "CACF Coverage%", 
                        "Standard Services%", "Effectiveness%", "SIP%", "SIP Coverage%", 
                        "Scan Results%", "Error Category%"]
        found_cols = [c for c in required_cols if c in df_raw.columns]
        missing_cols = [c for c in required_cols if c not in df_raw.columns]
        
        if "Date" not in df_raw.columns:
            return jsonify({"error": "Required column 'Date' not found.", "columns_found": list(df_raw.columns)}), 400
        
        # Parse dates
        df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
        df_raw = df_raw.dropna(subset=["Date"])
        
        # Convert metric columns to numeric
        metric_cols = [c for c in found_cols if c != "Date"]
        for col in metric_cols:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        
        performance_metrics_data = {
            "filepath": filepath,
            "df": df_raw,
            "metrics": metric_cols,
        }
        
        date_range = f"{df_raw['Date'].min().strftime('%Y-%m-%d')} to {df_raw['Date'].max().strftime('%Y-%m-%d')}"
        
        return jsonify({
            "success": True,
            "message": f"{len(df_raw):,} records loaded ({date_range})",
            "row_count": len(df_raw),
            "date_range": date_range,
            "metrics_found": metric_cols,
            "missing_columns": missing_cols if missing_cols else None,
        })
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": "Exception during performance metrics processing", "detail": str(e)}), 500


@app.route("/performance_analysis", methods=["POST"])
def performance_analysis():
    """Analyze performance metrics - pick first of month data, compute trends."""
    global performance_metrics_data
    
    if not performance_metrics_data:
        return jsonify({"error": "No performance metrics file uploaded."}), 400
    
    df = performance_metrics_data["df"].copy()
    
    try:
        # Define metric groupings
        inventory_metrics = ["TIP", "Depth%", "ILMT%", "CIT%"]
        automation_metrics = ["TAP%", "CACF Coverage%", "Standard Services%", "Effectiveness%"]
        service_insights_metrics = ["SIP%", "SIP Coverage%", "Scan Results%", "Error Category%"]
        
        # Get available metrics
        available_inv = [m for m in inventory_metrics if m in df.columns]
        available_auto = [m for m in automation_metrics if m in df.columns]
        available_si = [m for m in service_insights_metrics if m in df.columns]
        all_metrics = available_inv + available_auto + available_si
        
        # Add month/year columns
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["YearMonth"] = df["Date"].dt.to_period("M")
        
        # Function to pick the best day for each month
        def get_first_of_month_data(group):
            """Pick first of month, or earliest day with all values present."""
            group = group.sort_values("Day")
            
            # Try day 1 first
            day1 = group[group["Day"] == 1]
            if len(day1) > 0:
                row = day1.iloc[0]
                # Check if all metrics have values
                if not row[all_metrics].isna().any():
                    return row
            
            # Otherwise find earliest day with all values present
            for _, row in group.iterrows():
                if not row[all_metrics].isna().any():
                    return row
            
            # Fallback to first available row
            return group.iloc[0]
        
        # Get monthly data points
        monthly_data = []
        for ym in df["YearMonth"].unique():
            month_df = df[df["YearMonth"] == ym]
            if len(month_df) > 0:
                best_row = get_first_of_month_data(month_df)
                monthly_data.append({
                    "period": str(ym),
                    "date": best_row["Date"].strftime("%Y-%m-%d"),
                    **{m: float(best_row[m]) if pd.notna(best_row[m]) else None for m in all_metrics}
                })
        
        # Sort by period
        monthly_data = sorted(monthly_data, key=lambda x: x["period"])
        
        # Get latest values
        latest = monthly_data[-1] if monthly_data else {}
        
        # Build section data
        def build_section(metrics, section_name):
            section = {
                "name": section_name,
                "latest": {},
                "trends": {},
            }
            for m in metrics:
                if m in latest:
                    section["latest"][m] = latest.get(m)
                    section["trends"][m] = [{"period": d["period"], "value": d.get(m)} for d in monthly_data]
            return section
        
        inventory_section = build_section(available_inv, "Inventory")
        automation_section = build_section(available_auto, "Automation")
        service_section = build_section(available_si, "Service Insights")
        
        # Calculate summary - first half vs second half averages
        mid_point = len(monthly_data) // 2
        first_half = monthly_data[:mid_point] if mid_point > 0 else monthly_data
        second_half = monthly_data[mid_point:] if mid_point > 0 else monthly_data
        
        def calc_avg(data_list, metric):
            vals = [d.get(metric) for d in data_list if d.get(metric) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None
        
        def get_verdict(first_avg, second_avg):
            if first_avg is None or second_avg is None:
                return "unknown"
            diff = second_avg - first_avg
            if abs(diff) < 1:
                return "stable"
            return "rising" if diff > 0 else "falling"
        
        summary = {
            "periods_analyzed": len(monthly_data),
            "first_half_period": f"{monthly_data[0]['period']} to {first_half[-1]['period']}" if first_half else None,
            "second_half_period": f"{second_half[0]['period']} to {monthly_data[-1]['period']}" if second_half else None,
            "metrics": {},
            "section_verdicts": {},
            "overall_verdict": "stable",
        }
        
        rising_count = 0
        falling_count = 0
        
        for section_name, metrics in [("Inventory", available_inv), ("Automation", available_auto), ("Service Insights", available_si)]:
            section_rising = 0
            section_falling = 0
            for m in metrics:
                first_avg = calc_avg(first_half, m)
                second_avg = calc_avg(second_half, m)
                verdict = get_verdict(first_avg, second_avg)
                summary["metrics"][m] = {
                    "first_half_avg": first_avg,
                    "second_half_avg": second_avg,
                    "verdict": verdict,
                }
                if verdict == "rising":
                    rising_count += 1
                    section_rising += 1
                elif verdict == "falling":
                    falling_count += 1
                    section_falling += 1
            
            # Section verdict
            if section_rising > section_falling:
                summary["section_verdicts"][section_name] = "rising"
            elif section_falling > section_rising:
                summary["section_verdicts"][section_name] = "falling"
            else:
                summary["section_verdicts"][section_name] = "stable"
        
        # Overall verdict
        if rising_count > falling_count + 2:
            summary["overall_verdict"] = "rising"
        elif falling_count > rising_count + 2:
            summary["overall_verdict"] = "falling"
        else:
            summary["overall_verdict"] = "stable"
        
        result = {
            "monthly_data": monthly_data,
            "inventory": inventory_section,
            "automation": automation_section,
            "service_insights": service_section,
            "summary": summary,
            "latest_period": latest.get("period"),
            "latest_date": latest.get("date"),
        }
        
        return jsonify({"success": True, "result": clean_for_json(result)})
        
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": "Exception during performance analysis", "detail": str(e)}), 500


def _process_inventory_for_eol_eos(inv_df):
    """Process inventory dataframe to extract EOL/EOS/Extended Support/EOES data.
    
    Supports two data formats:
    1. Date-based: eol_date and eos_date columns with dates (count if past today)
    2. Value-based: eol and eos columns where value equals "eol" or "eos"
    3. EOES status: eoes_status column with values EOES<1Y, EOES>1Y (Extended Support), EOES (past)
    
    Uses osnamei column for OS family detection (WINDOWS, LINUX, etc.)
    Uses osversion for specific version info in examples.
    
    Categories:
    - EOL: End of Life
    - EOS: End of Service  
    - Extended Support: Approaching extended support end (EOES<1Y or EOES>1Y)
    - EOES: Past extended support (status = EOES)
    
    Returns:
    - summary: total_servers, eol_count, eos_count, extended_support_count, eoes_count
    - os_breakdown: per-OS family data
    - broader_categories: Windows, Linux, Cisco, Others aggregates
    - detailed_os_data: list of OS entries sorted by total for graphs
    """
    import pandas as pd
    
    df = inv_df.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip().str.lower()
    today = pd.Timestamp.now(tz=None).normalize()
    
    total_servers = len(df)
    
    # Detect which format we have
    has_eol_date = "eol_date" in df.columns
    has_eos_date = "eos_date" in df.columns
    has_eol_value = "eol" in df.columns
    has_eos_value = "eos" in df.columns
    has_eoes_status = "eoes_status" in df.columns
    has_eoes_date = "end_of_extended_support_date" in df.columns
    
    eol_count = 0
    eos_count = 0
    extended_support_count = 0
    eoes_count = 0
    eol_mask = pd.Series([False] * len(df), index=df.index)
    eos_mask = pd.Series([False] * len(df), index=df.index)
    extended_support_mask = pd.Series([False] * len(df), index=df.index)
    eoes_mask = pd.Series([False] * len(df), index=df.index)
    
    # Try value-based columns first (eol column where value == "eol")
    if has_eol_value:
        eol_mask = df["eol"].fillna("").astype(str).str.strip().str.lower() == "eol"
        eol_count = int(eol_mask.sum())
    elif has_eol_date:
        # Parse date and count if past today
        parsed = pd.to_datetime(df["eol_date"], errors="coerce", utc=True)
        df["eol_date_parsed"] = parsed.dt.tz_convert(None) if parsed.dt.tz is not None else parsed
        eol_mask = df["eol_date_parsed"].notna() & (df["eol_date_parsed"] < today)
        eol_count = int(eol_mask.sum())
    
    if has_eos_value:
        eos_mask = df["eos"].fillna("").astype(str).str.strip().str.lower() == "eos"
        eos_count = int(eos_mask.sum())
    elif has_eos_date:
        # Parse date and count if past today
        parsed = pd.to_datetime(df["eos_date"], errors="coerce", utc=True)
        df["eos_date_parsed"] = parsed.dt.tz_convert(None) if parsed.dt.tz is not None else parsed
        eos_mask = df["eos_date_parsed"].notna() & (df["eos_date_parsed"] < today)
        eos_count = int(eos_mask.sum())
    
    # Extended Support and EOES detection
    # IMPORTANT: Only devices that have already reached EOS can be considered for extended support
    if has_eoes_status:
        eoes_status_upper = df["eoes_status"].fillna("").astype(str).str.upper()
        # Extended Support = approaching extended support end (EOES<1Y + EOES>1Y) AND already past EOS
        extended_support_mask = eoes_status_upper.isin(["EOES<1Y", "EOES>1Y"]) & eos_mask
        extended_support_count = int(extended_support_mask.sum())
        # EOES = PAST extended support (status = "EOES") AND already past EOS
        eoes_mask_status = eoes_status_upper == "EOES"
        eoes_mask = eoes_mask_status & eos_mask
        eoes_count = int(eoes_mask.sum())
    elif has_eoes_date:
        parsed_eoes = pd.to_datetime(df["end_of_extended_support_date"], errors="coerce", utc=True)
        df["eoes_date_parsed"] = parsed_eoes.dt.tz_convert(None) if hasattr(parsed_eoes.dt, 'tz_convert') else parsed_eoes
        one_year_from_now = today + pd.DateOffset(years=1)
        # Extended Support = approaching (within 1 year) AND already past EOS
        extended_support_mask = df["eoes_date_parsed"].notna() & (df["eoes_date_parsed"] < one_year_from_now) & (df["eoes_date_parsed"] >= today) & eos_mask
        extended_support_count = int(extended_support_mask.sum())
        # EOES = past extended support AND already past EOS
        eoes_mask = df["eoes_date_parsed"].notna() & (df["eoes_date_parsed"] < today) & eos_mask
        eoes_count = int(eoes_mask.sum())
    
    # OS breakdown - prefer osnamei for family detection (it has clean values like "WINDOWS", "LINUX")
    # Must prioritize osnamei over osname since osname has verbose values
    os_col = None
    # First pass: look specifically for osnamei
    for col in df.columns:
        if col.lower() == "osnamei":
            os_col = col
            break
    # Second pass: fallback to other options if osnamei not found
    if os_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ["osname", "os_name", "os", "operating_system"]:
                os_col = col
                break
    
    # Look for version column
    version_col = None
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["osversion", "os_version", "version"]:
            version_col = col
            break
    
    # Also get the full osname column for better examples
    osname_col = None
    for col in df.columns:
        if col.lower() == "osname":
            osname_col = col
            break
    
    os_breakdown = {}
    broader_categories = {
        "Windows": {"total": 0, "eol": 0, "eos": 0, "extended_support": 0, "eoes": 0},
        "Linux": {"total": 0, "eol": 0, "eos": 0, "extended_support": 0, "eoes": 0},
        "Cisco": {"total": 0, "eol": 0, "eos": 0, "extended_support": 0, "eoes": 0},
        "Others": {"total": 0, "eol": 0, "eos": 0, "extended_support": 0, "eoes": 0},
    }
    detailed_os_data = []
    
    if os_col:
        for os_name, group in df.groupby(os_col):
            if pd.isna(os_name) or str(os_name).strip() == "":
                continue
            
            os_name_str = str(os_name)
            os_lower = os_name_str.lower()
            group_indices = group.index
            os_eol = int(eol_mask.loc[group_indices].sum())
            os_eos = int(eos_mask.loc[group_indices].sum())
            os_extended_support = int(extended_support_mask.loc[group_indices].sum())
            os_eoes = int(eoes_mask.loc[group_indices].sum())
            group_count = len(group)
            
            # Get most common version for this OS family
            os_version_str = ""
            if version_col and version_col in group.columns:
                versions = group[version_col].dropna().astype(str)
                if len(versions) > 0:
                    os_version_str = versions.value_counts().index[0] if len(versions.value_counts()) > 0 else ""
            
            # Get most common full osname for better display
            os_full_name = ""
            if osname_col and osname_col in group.columns:
                full_names = group[osname_col].dropna().astype(str)
                if len(full_names) > 0:
                    os_full_name = full_names.value_counts().index[0] if len(full_names.value_counts()) > 0 else ""
            
            os_breakdown[os_name_str] = {
                "count": group_count,
                "eol_count": os_eol,
                "eos_count": os_eos,
                "extended_support_count": os_extended_support,
                "eoes_count": os_eoes,
                "osversion": os_version_str,
                "osname_full": os_full_name,
            }
            
            # Add to detailed list for graphs
            detailed_os_data.append({
                "os_name": os_name_str,
                "os_full_name": os_full_name or os_name_str,
                "total": group_count,
                "eol": os_eol,
                "eos": os_eos,
                "extended_support": os_extended_support,
                "eoes": os_eoes,
            })
            
            # Categorize into broader categories
            # Use the full osname for better matching when available
            match_str = (os_full_name or os_name_str).lower()
            
            # Windows detection
            if "windows" in match_str or "win server" in match_str:
                broader_categories["Windows"]["total"] += group_count
                broader_categories["Windows"]["eol"] += os_eol
                broader_categories["Windows"]["eos"] += os_eos
                broader_categories["Windows"]["extended_support"] += os_extended_support
                broader_categories["Windows"]["eoes"] += os_eoes
            # Cisco detection - must check BEFORE generic "ios" to avoid false matches
            elif ("cisco" in match_str or 
                  match_str.startswith("ios") or  # IOS at start usually means Cisco IOS
                  "nx-os" in match_str or 
                  "nxos" in match_str or
                  "aci" in match_str and "cisco" in os_name_str.lower()):
                broader_categories["Cisco"]["total"] += group_count
                broader_categories["Cisco"]["eol"] += os_eol
                broader_categories["Cisco"]["eos"] += os_eos
                broader_categories["Cisco"]["extended_support"] += os_extended_support
                broader_categories["Cisco"]["eoes"] += os_eoes
            # Linux detection - expanded to catch more variants
            elif ("linux" in match_str or 
                  "red hat" in match_str or 
                  "redhat" in match_str or
                  "rhel" in match_str or 
                  "centos" in match_str or 
                  "ubuntu" in match_str or
                  "debian" in match_str or
                  "suse" in match_str or
                  "oracle linux" in match_str or
                  "fedora" in match_str or
                  "amazon linux" in match_str or
                  "alpine" in match_str):
                broader_categories["Linux"]["total"] += group_count
                broader_categories["Linux"]["eol"] += os_eol
                broader_categories["Linux"]["eos"] += os_eos
                broader_categories["Linux"]["extended_support"] += os_extended_support
                broader_categories["Linux"]["eoes"] += os_eoes
            else:
                broader_categories["Others"]["total"] += group_count
                broader_categories["Others"]["eol"] += os_eol
                broader_categories["Others"]["eos"] += os_eos
                broader_categories["Others"]["extended_support"] += os_extended_support
                broader_categories["Others"]["eoes"] += os_eoes
    
    # Sort detailed OS data by total count for better graphs
    detailed_os_data.sort(key=lambda x: x["total"], reverse=True)
    
    # Remove zero entries from broader categories
    broader_categories = {k: v for k, v in broader_categories.items() if v["total"] > 0}
    
    # ── Extended Support breakdown by FULL osname (not family grouping) ──
    # This gives per-OS-name granularity (e.g., "Microsoft Windows Server 2019 Standard")
    extended_support_by_osname = []
    full_os_col = osname_col or os_col  # prefer osname for full verbose names
    if full_os_col and extended_support_count > 0:
        ext_supp_df = df[extended_support_mask]
        if full_os_col in ext_supp_df.columns:
            os_counts = ext_supp_df[full_os_col].fillna("(unknown)").astype(str).value_counts()
            for os_name_val, cnt in os_counts.items():
                if cnt > 0:
                    # Get most common osversion for this OS name
                    osversion_str = ""
                    if version_col and version_col in ext_supp_df.columns:
                        os_subset = ext_supp_df[ext_supp_df[full_os_col].fillna("(unknown)").astype(str) == str(os_name_val)]
                        versions = os_subset[version_col].dropna().astype(str)
                        versions = versions[versions.str.strip() != ""]
                        if len(versions) > 0:
                            osversion_str = versions.value_counts().index[0]
                    extended_support_by_osname.append({
                        "os_full_name": str(os_name_val),
                        "count": int(cnt),
                        "osversion": osversion_str,
                    })
    
    return {
        "summary": {
            "total_servers": total_servers,
            "eol_count": eol_count,
            "eos_count": eos_count,
            "extended_support_count": extended_support_count,
            "eoes_count": eoes_count,
            "eol_percentage": round(eol_count / total_servers * 100, 1) if total_servers > 0 else 0,
            "eos_percentage": round(eos_count / total_servers * 100, 1) if total_servers > 0 else 0,
            "extended_support_percentage": round(extended_support_count / total_servers * 100, 1) if total_servers > 0 else 0,
            "eoes_percentage": round(eoes_count / total_servers * 100, 1) if total_servers > 0 else 0,
        },
        "os_breakdown": os_breakdown,
        "broader_categories": broader_categories,
        "detailed_os_data": detailed_os_data[:15],  # Top 15 for readability in graphs
        "extended_support_by_osname": extended_support_by_osname,
    }
# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION ENGINE — Backend (Elasticsearch) + Frontend (Cookie APIs)
# ═══════════════════════════════════════════════════════════════════════════════

extraction_jobs = {}  # job_id -> {status, log, file_path, ...}

ELASTIC_REGIONS = {
    "es.was1": {"url": "https://dp-elastic-ag1-prod-int.kyndryl.net/elasticsearch/", "auth": ("task_load_aiops", "fPEsyhU9BHzqXP5X7tcJQ")},
    "es.ams1": {"url": "https://dp-elastic-eu1-prod-int.kyndryl.net:443/elasticsearch/", "auth": ("task_load_aiops", "-oed_u3rr4AnKe_Wao9Hmi")},
    "es.tok1": {"url": "https://dp-elastic-jp1-prod-int.kyndryl.net:443/elasticsearch/", "auth": ("task_analytics_aiops", "drd8kpc.efx0GFZ4pfm")},
    "es.che1": {"url": "https://dp-elastic-in1-prod-int.kyndryl.net:443/elasticsearch/", "auth": ("task_load_aiops", "35rsGMu7UZjX4rF9xD")},
    "es.syd1": {"url": "https://dp-elastic-ap1-prod-int.kyndryl.net/elasticsearch/", "auth": ("task_load_aiops", "-qMuevNFCyA72Fwg.D")},
}
DEFAULT_ELASTIC = ELASTIC_REGIONS["es.was1"]

SEARCH_BATCH = 10000
SCROLL_DUR = "5m"
REQ_TIMEOUT = (10000, 10000)
NO_KW_INDICES = {"delivery.inventory.common-data"}

INC_COLUMNS = [
    "category", "closed_dttm", "resolution", "incident_code_id", "priority_df",
    "open_dttm", "autogen", "hostname", "resolution_code", "sso_ticket",
    "mttr_excl_hold", "business_application", "suggested_automata", "closure_code",
    "description", "reassignments", "assignment_grp_parent", "ownergroup", "label",
    "abstract", "abstract_desc", "dv_severity", "ostype",
]

DATASET_DEFS = {
    "business_app_mapping":      {"label": "Business Application Mapping", "source": "frontend"},
    "incidents":                 {"label": "Incidents",                    "source": "backend", "index": "cdi_incident_tickets_processed"},
    "service_requests":          {"label": "Service Requests",             "source": "backend", "index": "cdi_service_requests_processed"},
    "change_requests":           {"label": "Change Requests",              "source": "backend", "index": "cdi_changerequests_processed"},
    "config_processed_enriched": {"label": "Inventory / Config (Enriched)","source": "backend", "index": "cdi_configurations_processed", "extra": "delivery.inventory.common-data"},
    "actionable_insights":       {"label": "Actionable Insights V1 Merged", "source": "frontend"},
}

BROWSER_UA = "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Mobile Safari/537.36"
BIZ_API = "/api/aiops-selfservice/v1/download/business-applications-mapping"
AI_ACTIONABLE = "/api/aiops/v1/actionableInsights/actionableData"
AI_TOP = "/api/aiops/v1/actionableInsights/getTopCategories"
AI_REC = "/api/aiops/v1/actionableInsights/getRecommendedActionsObservations"

def _flatten(d, parent="", sep="."):
    items = []
    for k, v in d.items():
        nk = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            items.extend(_flatten(v, nk, sep).items())
        elif isinstance(v, list):
            items.append((nk, json.dumps(v, ensure_ascii=False)))
        else:
            items.append((nk, v))
    return dict(items)

def _html_strip(h):
    if not h: return ""
    class S(HTMLParser):
        def __init__(self): super().__init__(); self.p=[]
        def handle_data(self,d): self.p.append(d)
    s=S(); s.feed(h); return re.sub(r'\s+',' ',' '.join(s.p)).strip()

def _safe_name(n):
    n=(n or "account").strip(); return re.sub(r'[\\/*?:"<>|]','',n) or "account"

def _get_elastic(dp):
    for k,c in ELASTIC_REGIONS.items():
        if dp.startswith(k): return c
    return DEFAULT_ELASTIC

def _elastic_session():
    s = http_requests.Session()
    r = Retry(total=3, connect=3, read=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
    a = HTTPAdapter(max_retries=r, pool_connections=20, pool_maxsize=20)
    s.mount("http://",a); s.mount("https://",a)
    s.headers.update({"Content-Type":"application/json","Accept-Encoding":"gzip, deflate"})
    return s

def _download_index(tenant_id, index, dp, log_fn):
    ec = _get_elastic(dp); url=ec["url"].rstrip("/"); auth=ec["auth"]
    field = "tenant_id.keyword" if index not in NO_KW_INDICES else "tenant_id"
    q = {"query":{"bool":{"must":[{"term":{field:tenant_id}}]}},"size":SEARCH_BATCH,"sort":["_doc"]}
    if index == "cdi_incident_tickets_processed": q["_source"] = INC_COLUMNS
    all_docs=[]; scroll_id=None; sess=_elastic_session()
    try:
        log_fn(f"Downloading {index} for {tenant_id}…")
        r = sess.post(f"{url}/{index}/_search?scroll={SCROLL_DUR}", auth=auth, json=q, timeout=REQ_TIMEOUT, verify=False)
        if r.status_code != 200: log_fn(f"Error: {r.status_code}"); return []
        d = r.json(); scroll_id = d.get("_scroll_id"); hits = d.get("hits",{}).get("hits",[])
        all_docs.extend(hits); log_fn(f"  Batch: {len(hits)}")
        while scroll_id and hits:
            sr = sess.post(f"{url}/_search/scroll", auth=auth, json={"scroll":SCROLL_DUR,"scroll_id":scroll_id}, timeout=REQ_TIMEOUT, verify=False)
            if sr.status_code != 200: break
            sd = sr.json(); scroll_id = sd.get("_scroll_id"); hits = sd.get("hits",{}).get("hits",[])
            all_docs.extend(hits); log_fn(f"  Batch: {len(hits)} | Total: {len(all_docs)}")
        log_fn(f"  Total: {len(all_docs)}")
        return [_flatten(doc.get("_source",{})) for doc in all_docs]
    except Exception as e: log_fn(f"  Failed: {e}"); return []
    finally:
        if scroll_id:
            try: sess.delete(f"{url}/_search/scroll", auth=auth, json={"scroll_id":[scroll_id]}, timeout=REQ_TIMEOUT, verify=False)
            except: pass
        sess.close()

def _enrich_eoes(config_data, inv_data, log_fn):
    cdf = pd.DataFrame(config_data) if config_data else pd.DataFrame()
    idf = pd.DataFrame(inv_data) if inv_data else pd.DataFrame()
    if cdf.empty or idf.empty: return cdf
    HC = ("hostname","host_name","host","name","device_name","ci_name")
    chc = next((c for c in cdf.columns if c.lower() in HC), None)
    ihc = next((c for c in idf.columns if c.lower() in HC), None)
    if not chc or not ihc: return cdf
    edc = next((c for c in idf.columns if "end_of_extended_support" in c.lower()), None)
    if not edc: return cdf
    lu = idf[[ihc,edc]].dropna(subset=[ihc]).copy()
    lu[ihc] = lu[ihc].astype(str).str.strip().str.lower()
    lu = lu.drop_duplicates(subset=[ihc]).set_index(ihc)[edc]
    edf = cdf.copy()
    hk = edf[chc].astype(str).str.strip().str.lower()
    edf["end_of_extended_support_date"] = hk.map(lu)
    log_fn(f"EoES: matched {edf['end_of_extended_support_date'].notna().sum()}/{len(edf)}")
    today = pd.Timestamp.now().normalize()
    def cls(v):
        if pd.isna(v) or str(v).strip() in ("","nan","None","NaT"): return ""
        try:
            dt = pd.to_datetime(v, errors="raise")
            if hasattr(dt,"tzinfo") and dt.tzinfo: dt = dt.tz_localize(None)
            if dt <= today: return "EOES"
            elif (dt-today).days <= 365: return "EOES<1Y"
            else: return "EOES>1Y"
        except: return ""
    edf["eoes_status"] = edf["end_of_extended_support_date"].apply(cls)
    return edf

def _post_json(sess, url, headers, body, skip_404=False):
    r = sess.post(url, headers=headers, json=body, timeout=6000)
    if r.status_code == 404 and skip_404: return None
    if r.status_code >= 400: raise http_requests.HTTPError(f"{r.status_code} {r.reason}")
    return r.json()

def _build_fe_cfg(acct, cookie):
    ou = acct["ou_id"]
    meta = {"dx_tenant":ou,"bacId":acct["tenant_id"],"contractId":acct.get("contract_ids",[]),
            "tenantId":acct["tenant_id"],"aiopsDataSegment":acct["dataplane"],
            "bestPracticeDataSegment":acct["dataplane"],"sslDataSegment":acct["dataplane"],
            "customerId":acct.get("customer_id",""),"customerName":acct.get("customer_name",acct.get("name","")),
            "gsmaCode":acct.get("gsma_code",[])}
    return {"base_url":f"https://{ou}.account.delivery.kyndryl.net","x_tenant":ou,"cookie":cookie,"ouMetaData":meta}

def _requires_frontend_access(dataset_ids):
    for ds_id in dataset_ids or []:
        dd = DATASET_DEFS.get(ds_id) or {}
        if dd.get("source") == "frontend":
            return True
    return False

AI_TICKET_DET = "/api/aiops/v1/actionableInsights/getTicketDetails"

_BP_SUFFIX_TOKENS = {
    "microsoft","os","windows","linux","aix","solaris","unix","hpux","vmware",
    "cisco","juniper","network","storage","esx","rhel","centos","ubuntu",
    "debian","suse","oracle","ibm","hpe","dell","emc",
}
_PLAT_MAP = {"WIN":"Windows","LNX":"Linux","AIX":"AIX","SOL":"Solaris","HPX":"HP-UX","ESX":"VMware ESX"}

def _parse_hostname(record):
    hn = (record.get("hostname") or "").strip()
    if not hn:
        raw = (record.get("device_name") or "").strip()
        if raw:
            parts = raw.split("_"); keep = []
            for part in parts:
                if part.lower() in _BP_SUFFIX_TOKENS: break
                keep.append(part)
            hn = "_".join(keep) if keep else raw
    return hn

def _parse_platform(record):
    plat = (record.get("platform") or record.get("device_machine_type") or "").strip()
    return _PLAT_MAP.get(plat.upper(), plat) if plat else ""

def _fetch_insights_rich(sess, base, headers, cfg, insight_type, log_fn):
    """Fetch all insights with full detail: hostnames, platforms, ticket_rows."""
    ou = cfg["ouMetaData"]
    body = {"ouMetaData":{k:ou[k] for k in ["bacId","contractId","tenantId","aiopsDataSegment","customerName","customerId","gsmaCode","dx_tenant"]},"type":insight_type,"requestedInsightId":None}
    label = "actionableInsights" if insight_type == "actionable" else "growthInsights"
    log_fn(f"Calling actionableData ({insight_type})…")
    resp = _post_json(sess, base+AI_ACTIONABLE, headers, body)
    msg = resp.get("MESSAGE", resp); lst = msg.get("actionableInsights", [])
    if not lst: log_fn(f"No {label} returned."); return []
    sd = msg.get("snapshotDate") or lst[0].get("snapshotDate")
    mo = msg.get("maxMonth") or lst[0].get("maxMonth")
    if not sd or not mo: raise RuntimeError("No snapshotDate/month")
    id2meta = {int(i["insightId"]):{"title":i.get("insightTitle",""),"category":i.get("insightCategory","")} for i in lst if "insightId" in i}
    ids = sorted(id2meta.keys()); log_fn(f"Found {len(ids)} {label}.")
    results = []
    for idx, iid in enumerate(ids, 1):
        db = {"ouMetaData":cfg["ouMetaData"],"snapshotDate":sd,"insightId":iid,"month":mo,"scopeFilter":{},"type":insight_type}
        top = _post_json(sess, base+AI_TOP, headers, db, skip_404=True)
        if top is None: time.sleep(0.05); continue
        tb = {**db,"insightCategory":id2meta[iid]["category"],"source":[],"pageSize":10000,"from":0,"sort":{"field":"open_dttm","direction":"desc"},"searchQuery":"","ticketNum":""}
        tresp = _post_json(sess, base+AI_TICKET_DET, headers, tb, skip_404=True)
        ticket_rows, real_hostnames, platform_counts = [], [], {}
        if tresp and isinstance(tresp.get("data"), list):
            ticket_rows = tresp["data"]; seen = set()
            for d in ticket_rows:
                hn = _parse_hostname(d); plat = _parse_platform(d)
                if hn and hn not in seen: seen.add(hn); real_hostnames.append(hn)
                if plat: platform_counts[plat] = platform_counts.get(plat, 0) + 1
            real_hostnames.sort()
        if not real_hostnames:
            real_hostnames = sorted({d.get("metric_category") for d in top.get("data",[]) if d.get("metric_category")})
        rec = _post_json(sess, base+AI_REC, headers, db, skip_404=True)
        actions, obs, recom = [], "", ""
        if rec and isinstance(rec.get("data"), list):
            for d in rec["data"]:
                a = d.get("action") or d.get("issue_name") or ""
                if a: actions.append(a)
                if not obs: obs = _html_strip(d.get("observations",""))
                if not recom: recom = _html_strip(d.get("process_instructions",""))
        if not actions: actions = [""]
        results.append({"insight_id":iid,"insight_title":id2meta[iid]["title"],"actions":actions,
                        "observation":obs,"recommendation":recom,
                        "hostname_cell":", ".join(real_hostnames),"total_hosts":len(real_hostnames),
                        "platform_counts":platform_counts,
                        "platform_summary":", ".join(f"{p}: {c}" for p,c in sorted(platform_counts.items())) if platform_counts else "",
                        "ticket_rows":ticket_rows})
        log_fn(f"  [{idx}/{len(ids)}] id={iid}: {len(real_hostnames)} hosts, {len(actions)} actions")
        time.sleep(0.05)
    return results

def _make_sheet_name(iid, title, suffix="", used=None, prefix=""):
    for ch in "\\/*?:[]": title = title.replace(ch, "")
    pre = f"{prefix}-" if prefix else ""; tag = f"{pre}{iid} - "; suf = f" - {suffix}" if suffix else ""
    avail = 31 - len(tag) - len(suf); short = title[:max(avail, 4)].strip(); name = f"{tag}{short}{suf}"
    if used is not None:
        base, n = name, 1
        while name in used: n += 1; name = f"{base[:28]}-{n}"
        used.add(name)
    return name

def _build_summary_df(insight, all_platforms):
    rows = []
    for act in insight["actions"]:
        row = {"insight_id":insight["insight_id"],"insight_title":insight["insight_title"],
               "hostname":insight["hostname_cell"],"total_hosts":insight["total_hosts"],
               "platform_summary":insight["platform_summary"]}
        for plat in all_platforms: row[plat] = insight["platform_counts"].get(plat, 0)
        row["action"] = act; row["observation"] = insight["observation"]; row["recommendation"] = insight["recommendation"]
        rows.append(row)
    return pd.DataFrame(rows)

def _build_details_df(insight):
    if not insight["ticket_rows"]: return pd.DataFrame([{"note":"No ticket detail data."}])
    df = pd.DataFrame(insight["ticket_rows"])
    prio = ["hostname","device_name","platform","device_machine_type","device_type","device_vendor",
            "device_model","device_firmware","check_name","check_result","check_severity","check_category",
            "cert_cn","cert_expiration_date","pub_key_size","pub_key_alg","TLS_used","net_port","scan_date"]
    ordered = [c for c in prio if c in df.columns] + [c for c in df.columns if c not in prio]
    return df[ordered]

_XML_BAD = re.compile(r"[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD]")

def _clean_excel(df):
    """Make DataFrame safe for openpyxl: stringify dicts/lists, remove XML-illegal chars."""
    if df is None or df.empty:
        return df
    c = df.copy()
    def _cl(x):
        if pd.isna(x): return None
        if isinstance(x, (dict, list)): x = json.dumps(x, ensure_ascii=False)
        x = str(x)
        x = _XML_BAD.sub(" ", x)
        x = x.encode("utf-8", "ignore").decode("utf-8", "ignore")
        return x
    for col in c.columns:
        c[col] = c[col].map(_cl)
    return c

# --- Extraction job runners ---

def _run_biz_app(acct, cookie, jid):
    def log(m): extraction_jobs[jid]["log"] += m+"\n"
    try:
        cfg = _build_fe_cfg(acct, cookie)
        h = {"Accept":"application/json","Content-Type":"application/json","Origin":"https://www.kyndryl.com","Referer":"https://www.kyndryl.com/","x-tenant":cfg["x_tenant"],"Cookie":cookie,"User-Agent":"Mozilla/5.0"}
        log("Calling BAM API…")
        r = http_requests.post(cfg["base_url"]+BIZ_API, headers=h, json={"ouMetaData":cfg["ouMetaData"]}, verify=True, timeout=6000)
        r.raise_for_status(); data = r.json()
        if not isinstance(data,list): raise ValueError("Expected list")
        df = pd.DataFrame(data)
        tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".xlsx",prefix="bam_"); tmp.close()
        df.to_excel(tmp.name, index=False)
        log(f"✅ Done. {len(df)} rows"); extraction_jobs[jid].update(file_path=tmp.name, download_name=f"{_safe_name(acct.get('customer_name',''))}_bam.xlsx", status="done", row_count=len(df))
    except Exception as e:
        extraction_jobs[jid]["log"] += f"\n❌ {e}\n"; extraction_jobs[jid]["status"] = "error"

def _run_backend(acct, cookie, jid, index, extra=None):
    def log(m): extraction_jobs[jid]["log"] += m+"\n"
    try:
        tid = acct["tenant_id"]; dp = acct["dataplane"]
        data = _download_index(tid, index, dp, log)
        df = pd.DataFrame(data); log(f"Rows: {len(df)}")
        if extra and index == "cdi_configurations_processed":
            log("Fetching inventory for EoES…")
            inv = _download_index(tid, extra, dp, log)
            if inv: df = _enrich_eoes(data, inv, log)
        tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".xlsx",prefix=f"{index}_"); tmp.close()
        with pd.ExcelWriter(tmp.name, engine="openpyxl") as w: _clean_excel(df).to_excel(w, index=False, sheet_name="data")
        log(f"✅ Done. {len(df)} rows"); extraction_jobs[jid].update(file_path=tmp.name, download_name=f"{_safe_name(acct.get('customer_name',''))}_{index}.xlsx", status="done", row_count=len(df))
    except Exception as e:
        extraction_jobs[jid]["log"] += f"\n❌ {e}\n"; extraction_jobs[jid]["status"] = "error"

def _run_insights(acct, cookie, jid):
    """Produce A-/G- multi-sheet workbook that analyse_insights_file can parse."""
    def log(m): extraction_jobs[jid]["log"] += m+"\n"
    try:
        cfg = _build_fe_cfg(acct, cookie)
        base = cfg["base_url"].rstrip("/")
        h = {"accept":"application/json, text/plain, */*","content-type":"application/json",
             "user-agent":BROWSER_UA,"origin":"https://www.kyndryl.com","referer":"https://www.kyndryl.com/",
             "x-tenant":cfg["x_tenant"],"cookie":cookie}
        s = http_requests.Session()
        log("── Fetching Actionable + ThreatCon Insights ──")
        act = _fetch_insights_rich(s, base, h, cfg, "actionable", log)
        log("── Fetching Growth Insights ──")
        grw = _fetch_insights_rich(s, base, h, cfg, "growthInsights", log)
        if not act and not grw: raise RuntimeError("No insights returned.")
        all_ins = act + grw
        all_plats = sorted({p for ins in all_ins for p in ins["platform_counts"]})
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx", prefix="insights_combined_"); tmp.close()
        used = set()
        with pd.ExcelWriter(tmp.name, engine="openpyxl") as writer:
            for ins in act:
                iid, title = ins["insight_id"], ins["insight_title"]
                sn = _make_sheet_name(iid, title, suffix="", used=used, prefix="A")
                dn = _make_sheet_name(iid, title, suffix="Det", used=used, prefix="A")
                _build_summary_df(ins, all_plats).to_excel(writer, index=False, sheet_name=sn)
                _build_details_df(ins).to_excel(writer, index=False, sheet_name=dn)
                log(f"  [A] {sn}")
            for ins in grw:
                iid, title = ins["insight_id"], ins["insight_title"]
                sn = _make_sheet_name(iid, title, suffix="", used=used, prefix="G")
                dn = _make_sheet_name(iid, title, suffix="Det", used=used, prefix="G")
                _build_summary_df(ins, all_plats).to_excel(writer, index=False, sheet_name=sn)
                _build_details_df(ins).to_excel(writer, index=False, sheet_name=dn)
                log(f"  [G] {sn}")
        name = _safe_name(acct.get("customer_name", acct.get("name", "")))
        log(f"✅ Done. {len(act)} actionable + {len(grw)} growth → {len(used)} sheets.")
        extraction_jobs[jid].update(file_path=tmp.name, download_name=f"{name}_insights_combined.xlsx", status="done", row_count=len(act)+len(grw))
    except Exception as e:
        extraction_jobs[jid]["log"] += f"\n❌ {e}\n"; extraction_jobs[jid]["status"] = "error"

# --- Extraction API routes ---

@app.route("/api/accounts")
def api_accounts():
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "ou_list_stripped.json")
    if not os.path.exists(p): return jsonify({"error":"ou_list_stripped.json not found"}), 404
    with open(p, encoding="utf-8") as f: return jsonify(json.load(f))

@app.route("/api/validate", methods=["POST"])
def api_validate():
    data = request.get_json(force=True)
    cookie = (data.get("cookie") or "").strip(); acct = data.get("account")
    selected_datasets = data.get("datasets") or []
    if not acct:
        return jsonify({"error":"Account required"}), 400
    if not _requires_frontend_access(selected_datasets):
        return jsonify({
            "valid": True,
            "message": "Local extraction ready",
            "datasets": [{"id": k, "label": v["label"]} for k, v in DATASET_DEFS.items()],
            "mode": "backend",
        })
    if not cookie:
        return jsonify({"error":"Cookie required for frontend datasets"}), 400
    cfg = _build_fe_cfg(acct, cookie)
    base = cfg["base_url"].rstrip("/")
    h = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": "https://www.kyndryl.com",
        "Referer": "https://www.kyndryl.com/",
        "x-tenant": cfg["x_tenant"],
        "Cookie": cookie,
        "User-Agent": "Mozilla/5.0",
    }
    try:
        # Validate against the same frontend BAM API used by the working extraction flow.
        r = http_requests.post(
            base + BIZ_API,
            headers=h,
            json={"ouMetaData": cfg["ouMetaData"]},
            verify=True,
            timeout=45,
            allow_redirects=False,
        )

        ctype = (r.headers.get("content-type") or "").lower()
        if r.status_code in (301, 302, 303, 307, 308):
            loc = r.headers.get("location", "")
            return jsonify({"valid": False, "message": f"Authentication redirect detected. {loc or 'Please refresh the cookie from the working session.'}"})
        if r.status_code in (401, 403):
            return jsonify({"valid": False, "message": "Authentication failed — cookie may be expired or missing required tokens."})
        if r.status_code >= 500:
            return jsonify({"valid": False, "message": f"Server error ({r.status_code})"})
        if "text/html" in ctype:
            preview = re.sub(r"\s+", " ", r.text[:180]).strip()
            return jsonify({"valid": False, "message": f"Received HTML instead of API JSON. Cookie/session is being redirected. {preview}"})

        payload = r.json()
        if not isinstance(payload, list):
            return jsonify({"valid": False, "message": "Unexpected response shape from validation API."})

        return jsonify({
            "valid": True,
            "message": "Access validated",
            "datasets": [{"id": k, "label": v["label"]} for k, v in DATASET_DEFS.items()],
        })
    except ValueError as e:
        return jsonify({"valid": False, "message": f"Validation response was not JSON: {e}"})
    except http_requests.exceptions.ConnectionError:
        return jsonify({"valid":False,"message":"Cannot connect — VPN may be required."})
    except http_requests.exceptions.Timeout:
        return jsonify({"valid":False,"message":"Connection timed out."})
    except Exception as e:
        return jsonify({"valid":False,"message":str(e)})

@app.route("/api/extract", methods=["POST"])
def api_extract():
    data = request.get_json(force=True)
    cookie = (data.get("cookie") or "").strip(); acct = data.get("account"); dsets = data.get("datasets",[])
    if not acct or not dsets:
        return jsonify({"error":"Missing fields"}), 400
    if _requires_frontend_access(dsets) and not cookie:
        return jsonify({"error":"Cookie required for selected frontend datasets"}), 400
    job_ids = {}
    for ds_id in dsets:
        dd = DATASET_DEFS.get(ds_id); 
        if not dd: continue
        jid = str(uuid.uuid4())
        extraction_jobs[jid] = {"status":"running","log":"","file_path":None,"download_name":None,"dataset_id":ds_id,"label":dd["label"],"row_count":0}
        if dd["source"] == "frontend" and ds_id == "business_app_mapping":
            t = threading.Thread(target=_run_biz_app, args=(acct,cookie,jid), daemon=True)
        elif dd["source"] == "frontend" and ds_id == "actionable_insights":
            t = threading.Thread(target=_run_insights, args=(acct,cookie,jid), daemon=True)
        elif dd["source"] == "backend":
            t = threading.Thread(target=_run_backend, args=(acct,cookie,jid,dd["index"],dd.get("extra")), daemon=True)
        else: extraction_jobs[jid]["status"] = "error"; continue
        t.start(); job_ids[ds_id] = jid
    return jsonify({"jobs":job_ids})

@app.route("/api/status/<job_id>")
def api_ext_status(job_id):
    j = extraction_jobs.get(job_id)
    if not j: return jsonify({"error":"Not found"}), 404
    return jsonify({"status":j["status"],"log":j["log"],"row_count":j.get("row_count",0),"label":j.get("label","")})

@app.route("/api/download/<job_id>")
def api_ext_download(job_id):
    j = extraction_jobs.get(job_id)
    if not j or not j.get("file_path"): return jsonify({"error":"Not ready"}), 404
    return send_file(j["file_path"], as_attachment=True, download_name=j.get("download_name") or os.path.basename(j["file_path"]))


if __name__ == "__main__":
    provider = "Azure OpenAI" if AZURE_CONFIGURED else "NOT CONFIGURED"
    print("Starting Kyndryl Bridge Intelligence — Unified Platform")
    print(f"LLM Provider: {provider}")
    print(f"Deployment: {AZURE_OPENAI_DEPLOYMENT} | API Version: {AZURE_OPENAI_API_VERSION}")
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
