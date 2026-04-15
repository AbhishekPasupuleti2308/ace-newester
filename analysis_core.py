"""
analysis_core.py — Core incident analysis functions (hostnames, severity, automation, etc.)
Extracted from original app.py with no logic changes.
"""
import pandas as pd
from collections import Counter
from preprocessing import add_month_col, get_months_in_data, calc_projected_yearly_hours
from config import INDUSTRY_DOWNTIME_COSTS, NOISE_COST_PER_HOUR, convert_currency


# =========================
# Common Issues Analysis
# =========================
def analyze_common_issues(df, top_n=20):
    result = {}
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did", "will", "would", "could",
                  "should", "may", "might", "shall", "can", "need", "dare", "ought",
                  "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
                  "as", "into", "through", "during", "before", "after", "above", "below",
                  "between", "out", "off", "over", "under", "again", "further", "then",
                  "once", "here", "there", "when", "where", "why", "how", "all", "both",
                  "each", "few", "more", "most", "other", "some", "such", "no", "nor",
                  "not", "only", "own", "same", "so", "than", "too", "very", "just",
                  "don", "now", "and", "but", "or", "if", "it", "its", "this", "that",
                  "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
                  "him", "his", "she", "her", "they", "them", "their", "what", "which",
                  "who", "whom", "-", "--", ":", ".", ",", ";", "!", "?", ""}

    if "description" in df.columns:
        desc_series = df["description"].dropna().astype(str)
        if len(desc_series) > 0:
            desc_truncated = desc_series.str[:120].str.strip().str.lower()
            desc_counts = desc_truncated.value_counts().head(top_n)
            result["top_descriptions"] = [
                {"description": str(d), "count": int(c), "pct": round(c / len(df) * 100, 2)}
                for d, c in desc_counts.items()
            ]
            all_words = " ".join(desc_series.str.lower().values).split()
            meaningful_words = [w for w in all_words if len(w) > 2 and w not in stop_words]
            word_freq = Counter(meaningful_words).most_common(30)
            result["description_keywords"] = [{"word": w, "count": c} for w, c in word_freq]
            bigrams = []
            words_list = [w for w in all_words if w not in stop_words and len(w) > 2]
            for i in range(len(words_list) - 1):
                bigrams.append(f"{words_list[i]} {words_list[i+1]}")
            bigram_freq = Counter(bigrams).most_common(20)
            result["common_issue_phrases"] = [{"phrase": p, "count": c} for p, c in bigram_freq]

    if "resolution" in df.columns:
        res_series = df["resolution"].dropna().astype(str)
        if len(res_series) > 0:
            res_truncated = res_series.str[:120].str.strip().str.lower()
            res_counts = res_truncated.value_counts().head(top_n)
            result["top_resolutions"] = [
                {"resolution": str(r), "count": int(c), "pct": round(c / len(df) * 100, 2)}
                for r, c in res_counts.items()
            ]
            all_res_words = " ".join(res_series.str.lower().values).split()
            meaningful_res = [w for w in all_res_words if len(w) > 2 and w not in stop_words]
            res_word_freq = Counter(meaningful_res).most_common(30)
            result["resolution_keywords"] = [{"word": w, "count": c} for w, c in res_word_freq]

    if "category" in df.columns:
        cat_counts = df["category"].dropna().value_counts().head(15)
        result["top_categories"] = [
            {"category": str(c), "count": int(v), "pct": round(v / len(df) * 100, 2)}
            for c, v in cat_counts.items()
        ]

    if "description" in df.columns and "priority_df" in df.columns:
        for sev in [1, 2]:
            sev_df = df[df["priority_df"] == sev]
            if "description" in sev_df.columns and len(sev_df) > 0:
                sev_desc = sev_df["description"].dropna().astype(str).str[:120].str.strip().str.lower()
                sev_desc_counts = sev_desc.value_counts().head(10)
                result[f"sev{sev}_top_descriptions"] = [
                    {"description": str(d), "count": int(c)}
                    for d, c in sev_desc_counts.items()
                ]

    if "business_application" in df.columns and "description" in df.columns:
        top_apps = df["business_application"].value_counts().head(5).index.tolist()
        app_issues = {}
        for app_name in top_apps:
            if pd.isna(app_name):
                continue
            app_df = df[df["business_application"] == app_name]
            app_desc = app_df["description"].dropna().astype(str).str[:120].str.strip().str.lower()
            app_desc_counts = app_desc.value_counts().head(5)
            app_issues[str(app_name)] = [
                {"description": str(d), "count": int(c)}
                for d, c in app_desc_counts.items()
            ]
        result["issues_by_top_app"] = app_issues

    return result


# =========================
# Trends Analysis
# =========================
def analyze_trends(df):
    result = {}

    if "reassignments" in df.columns:
        reass = df["reassignments"].dropna()
        if len(reass) > 0:
            result["reassignment_stats"] = {
                "mean": round(float(reass.mean()), 2),
                "median": round(float(reass.median()), 2),
                "max": int(reass.max()),
                "zero_count": int((reass == 0).sum()),
                "zero_pct": round(float((reass == 0).sum()) / len(df) * 100, 1),
            }
            labels = ['0', '1', '2', '3', '4-5', '6-10', '11-50', '50+']
            buckets = pd.cut(reass, bins=[-.5, 0.5, 1.5, 2.5, 3.5, 5.5, 10.5, 50.5, float('inf')], labels=labels)
            dist = buckets.value_counts().reindex(labels, fill_value=0)
            result["reassignment_distribution"] = [
                {"bucket": str(b), "count": int(c)} for b, c in dist.items()
            ]
            if "priority_df" in df.columns:
                sev_reass = df.dropna(subset=["reassignments", "priority_df"]).groupby("priority_df")["reassignments"].agg(["mean", "count"]).round(2)
                result["reassignment_by_severity"] = [
                    {"severity": int(k), "avg_reassignments": round(float(v["mean"]), 2), "count": int(v["count"])}
                    for k, v in sev_reass.iterrows() if not pd.isna(k)
                ]

    if "assignment_grp_parent" in df.columns:
        agp = df["assignment_grp_parent"].dropna().astype(str)
        if len(agp) > 0:
            agp_counts = agp.value_counts().head(20)
            result["assignment_grp_parent_top"] = [
                {"group": str(g), "count": int(c), "pct": round(c / len(df) * 100, 1)}
                for g, c in agp_counts.items()
            ]
            if "priority_df" in df.columns:
                agp_sev = df.dropna(subset=["assignment_grp_parent", "priority_df"]).groupby("assignment_grp_parent")["priority_df"].apply(
                    lambda x: x.value_counts().to_dict()
                )
                agp_with_sev = []
                _si = lambda v: int(v) if pd.notna(v) else 0
                for grp in agp_counts.index[:15]:
                    sev_dist = agp_sev.get(grp, {})
                    agp_with_sev.append({
                        "group": str(grp),
                        "total": int(agp_counts[grp]),
                        "sev1": _si(sev_dist.get(1, 0)),
                        "sev2": _si(sev_dist.get(2, 0)),
                        "sev3": _si(sev_dist.get(3, 0)),
                        "sev4": _si(sev_dist.get(4, 0)),
                    })
                result["assignment_grp_parent_by_severity"] = agp_with_sev
            if "mttr_excl_hold" in df.columns:
                agp_mttr = df.dropna(subset=["assignment_grp_parent", "mttr_excl_hold"]).groupby("assignment_grp_parent")["mttr_excl_hold"].agg(["mean", "count"])
                agp_mttr = agp_mttr[agp_mttr["count"] >= 3].sort_values("mean", ascending=False).head(15)
                result["assignment_grp_parent_by_mttr"] = [
                    {"group": str(g), "avg_mttr": round(float(v["mean"]), 2), "count": int(v["count"])}
                    for g, v in agp_mttr.iterrows()
                ]

    if "ownergroup" in df.columns:
        og = df["ownergroup"].dropna().astype(str)
        if len(og) > 0:
            og_counts = og.value_counts().head(20)
            result["ownergroup_top"] = [
                {"group": str(g), "count": int(c), "pct": round(c / len(df) * 100, 1)}
                for g, c in og_counts.items()
            ]
            if "priority_df" in df.columns:
                og_sev = df.dropna(subset=["ownergroup", "priority_df"]).groupby("ownergroup")["priority_df"].apply(
                    lambda x: x.value_counts().to_dict()
                )
                og_with_sev = []
                _si = lambda v: int(v) if pd.notna(v) else 0
                for grp in og_counts.index[:15]:
                    sev_dist = og_sev.get(grp, {})
                    og_with_sev.append({
                        "group": str(grp),
                        "total": int(og_counts[grp]),
                        "sev1": _si(sev_dist.get(1, 0)),
                        "sev2": _si(sev_dist.get(2, 0)),
                        "sev3": _si(sev_dist.get(3, 0)),
                        "sev4": _si(sev_dist.get(4, 0)),
                    })
                result["ownergroup_by_severity"] = og_with_sev
            if "mttr_excl_hold" in df.columns:
                og_mttr = df.dropna(subset=["ownergroup", "mttr_excl_hold"]).groupby("ownergroup")["mttr_excl_hold"].agg(["mean", "count"])
                og_mttr = og_mttr[og_mttr["count"] >= 3].sort_values("mean", ascending=False).head(15)
                result["ownergroup_by_mttr"] = [
                    {"group": str(g), "avg_mttr": round(float(v["mean"]), 2), "count": int(v["count"])}
                    for g, v in og_mttr.iterrows()
                ]

    if "open_dttm" in df.columns and "priority_df" in df.columns and pd.api.types.is_datetime64_any_dtype(df["open_dttm"]):
        df_v = df.dropna(subset=["open_dttm", "priority_df"]).copy()
        if len(df_v) > 0:
            df_v = add_month_col(df_v)
            sev_monthly = df_v.groupby(["month", "priority_df"]).size().unstack(fill_value=0)
            trend = []
            for month in sev_monthly.index:
                row = {"month": str(month)}
                for sev in sev_monthly.columns:
                    if not pd.isna(sev):
                        row[f"sev{int(sev)}"] = int(sev_monthly.loc[month, sev])
                trend.append(row)
            result["severity_monthly_trend"] = trend

    if "open_dttm" in df.columns and pd.api.types.is_datetime64_any_dtype(df["open_dttm"]):
        df_open = df.dropna(subset=["open_dttm"]).copy()
        if len(df_open) > 0:
            hourly = df_open["open_dttm"].dt.hour.value_counts().sort_index()
            result["open_hourly"] = {str(k): int(v) for k, v in hourly.items()}

    if "closed_dttm" in df.columns and pd.api.types.is_datetime64_any_dtype(df["closed_dttm"]):
        df_close = df.dropna(subset=["closed_dttm"]).copy()
        if len(df_close) > 0:
            hourly_c = df_close["closed_dttm"].dt.hour.value_counts().sort_index()
            result["close_hourly"] = {str(k): int(v) for k, v in hourly_c.items()}

    return result


# =========================
# Ticket Timing
# =========================
def analyze_ticket_timing(df):
    err = {"error": True, "message": "", "details": {}}
    if "open_dttm" not in df.columns:
        err["message"] = "open_dttm column not present"
        return err
    if not pd.api.types.is_datetime64_any_dtype(df["open_dttm"]):
        err["message"] = "open_dttm is not datetime"
        return err

    df_valid = df.dropna(subset=["open_dttm"]).copy()
    if df_valid.empty:
        err["message"] = "No parseable open_dttm values"
        return err

    df_valid["hour"] = df_valid["open_dttm"].dt.hour
    df_valid["day_of_week"] = df_valid["open_dttm"].dt.dayofweek
    df_valid["day_name"] = df_valid["open_dttm"].dt.day_name()

    heatmap = df_valid.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
    heatmap = heatmap.reindex(index=range(7), columns=range(24), fill_value=0)

    hourly = df_valid["hour"].value_counts().sort_index().to_dict()
    daily = df_valid["day_name"].value_counts().to_dict()

    peak_hour = max(hourly, key=hourly.get) if hourly else None
    peak_day = max(daily, key=daily.get) if daily else None

    return {
        "error": False,
        "heatmap_data": heatmap.values.tolist(),
        "hours": list(range(24)),
        "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "hourly_distribution": {str(k): int(v) for k, v in hourly.items()},
        "daily_distribution": {str(k): int(v) for k, v in daily.items()},
        "parsed_count": int(len(df_valid)),
        "peak_hour": int(peak_hour) if peak_hour is not None else None,
        "peak_day": str(peak_day) if peak_day is not None else None,
    }


# =========================
# Hostname Analysis
# =========================
def analyze_hostnames(df):
    if "hostname" not in df.columns:
        return {}
    hostname_counts = df["hostname"].value_counts().head(20)

    top_mttr_hostnames = {}
    if "mttr_excl_hold" in df.columns:
        mttr_by_host = (
            df.dropna(subset=["hostname", "mttr_excl_hold"])
            .groupby("hostname")["mttr_excl_hold"]
            .agg(["mean", "count"])
        )
        mttr_by_host = mttr_by_host[mttr_by_host["count"] >= 3].sort_values("mean", ascending=False).head(10)
        top_mttr_hostnames = {
            str(k): {"avg_mttr": round(float(v["mean"]), 2), "ticket_count": int(v["count"])}
            for k, v in mttr_by_host.iterrows()
        }

    return {
        "top_hostnames": {str(k): int(v) for k, v in hostname_counts.to_dict().items()},
        "total_unique_hostnames": int(df["hostname"].nunique()),
        "top_mttr_hostnames": top_mttr_hostnames,
        "chart_data": {
            "labels": hostname_counts.index.tolist(),
            "values": hostname_counts.values.tolist()
        },
    }


# =========================
# Suggested Automata
# =========================
def analyze_suggested_automata(df):
    if "suggested_automata" not in df.columns:
        return {}
    counts = df["suggested_automata"].value_counts()
    total = len(df)
    pct = (counts / total * 100).round(2) if total > 0 else counts * 0
    return {
        "counts": counts.to_dict(),
        "percentages": pct.to_dict(),
        "chart_data": {
            "labels": counts.index.tolist(),
            "values": counts.values.tolist(),
            "percentages": pct.values.tolist()
        },
    }


# =========================
# Severity Analysis
# =========================
def analyze_severity(df):
    if "priority_df" not in df.columns:
        return {}
    severity_counts = df["priority_df"].value_counts().sort_index()
    high_priority = df[df["priority_df"].isin([1, 2])]
    high_priority_count = int(len(high_priority))
    high_priority_percentage = round(high_priority_count / len(df) * 100, 2) if len(df) > 0 else 0.0

    mttr_by_severity = {}
    if "mttr_excl_hold" in df.columns:
        mttr_by_severity = (
            df.groupby("priority_df")["mttr_excl_hold"]
            .agg([("mean", "mean"), ("median", "median"), ("min", "min"), ("max", "max"), ("count", "count")])
            .round(2)
            .to_dict("index")
        )

    sev_counts = {}
    for k, v in severity_counts.to_dict().items():
        if pd.isna(k):
            continue
        try:
            sev_counts[int(k)] = int(v)
        except Exception:
            sev_counts[str(k)] = int(v)

    labels, values = [], []
    for s in severity_counts.index.tolist():
        if pd.isna(s):
            continue
        try:
            labels.append(f"Severity {int(s)}")
        except Exception:
            labels.append(f"Severity {s}")
        values.append(int(severity_counts.loc[s]))

    return {
        "severity_counts": sev_counts,
        "high_priority_count": high_priority_count,
        "high_priority_percentage": float(high_priority_percentage),
        "mttr_by_severity": mttr_by_severity,
        "chart_data": {"labels": labels, "values": values},
    }


# =========================
# Automation Opportunity
# =========================

# Label to Suggested Automata mapping
# Used for tickets where suggested_automata is "unknown" or "other requests"
LABEL_TO_AUTOMATA_MAPPING = {
    # Access & Identity
    "access points": "access issues",
    "access request others": "access issues",
    "account disabled / suspended": "accounts locked and unlocked",
    "account expired / reset": "accounts locked and unlocked",
    "account extension": "access issues",
    "account locked / unlock": "accounts locked and unlocked",
    "account termination/activation": "access issues",
    "folder access": "access issues",
    "server access": "access issues",
    "password issue / reset": "password reset",
    
    # Infrastructure & Monitoring
    "alerts others": "check-errpt-alert",
    "av related": "hardware issues",
    "backup fail/miss": "missed-and-failed-backup",
    "domain controller": "server-unavailable",
    "entry in log": "sysinfo-and-errlog-check",
    "os problem": "service-restart",
    "process alerts": "sysinfo-and-errlog-check",
    "reboot": "service-restart",
    "server hang/node down": "server-unavailable",
    "service stopped": "service-restart",
    
    # Hardware
    "battery issue": "mobile device issues",
    "firewall issue": "hardware issues",
    "vmware issue": "hardware issues",
    "phone issue": "mobile device issues",
    "printer issue": "printer related issues",
    
    # Applications & Database
    "browser issue": "application issues",
    "dl related": "application issues",
    "excel issue": "application issues",
    "sql connectivity issue": "database-handler",
    "sql job failure": "batch-job-schedule-failures",
    
    # Network
    "dns issue": "network connectivity issues",
    "network issue": "network connectivity issues",
    "rdp issue": "network connectivity issues",
    "switch port / reset": "network connectivity issues",
    "vpn issue": "network connectivity issues",
    "wifi issue": "network connectivity issues",
    
    # Storage & Capacity
    "disk space": "disk-handler",
    "drive access": "drive related issues",
    "high cpu utilization": "cpu issues",
    "high page file usage": "high-memory-swap-space-handler",
    "memory issue": "high-memory-swap-space-handler",
    
    # Email & Mailbox
    "email / outlook issues": "mailbox related issues",
    
    # Other - these map to order requests (excluded category)
    "new joiner/leaver": "order requests",
    "sw /hw installation": "order requests",
}

# Automata types to exclude from automation opportunity count
# ONLY these three are excluded
EXCLUDED_AUTOMATA = {"other-handler", "unknown", "other requests"}

# Broader category mapping for suggested_automata
AUTOMATA_TO_CATEGORY = {
    # IAM (Identity & Access Management)
    "access issues": "IAM",
    "accounts locked and unlocked": "IAM",
    "acccounts locked and unlocked": "IAM",  # Handle typo variant
    "password reset": "IAM",
    
    # Infrastructure
    "check-errpt-alert": "Infra",
    "missed-and-failed-backup": "Infra",
    "application issues": "Infra",
    "database-handler": "Infra",
    "network-handler": "Infra",
    "server-unavailable": "Infra",
    "drive related issues": "Infra",
    "sysinfo-and-errlog-check": "Infra",
    "cpu-high-handler": "Infra",
    "high-memory-swap-space-handler": "Infra",
    "network connectivity issues": "Infra",
    "service-restart": "Infra",
    "batch-job-schedule-failures": "Infra",
    "application-handler": "Infra",
    "itm-agent-offline": "Infra",
    "cpu issues": "Infra",
    "mf_application": "Infra",
    "mf_batch-job-schedule-failures": "Infra",
    "mf_database": "Infra",
    "mf_missed-and-failed-backup": "Infra",
    
    # Hardware
    "hardware issues": "Hardware",
    "mobile device issues": "Hardware",
    "printer related issues": "Hardware",
    
    # Capacity
    "disk-handler": "Capacity",
    
    # Mailbox
    "mailbox related issues": "Mailbox",
    
    # Others
    "other-handler": "Others",
    "other requests": "Others",
    "order requests": "Others",
    "unknown": "Others",
}


def analyze_automation_opportunity(df):
    """
    Automation Opportunity Calculation Logic:
    
    1. Exclude ONLY tickets with closure_code:
       - "Remediation with Corrective Closure"
       - "Remediation with Validation Closure"
    
    2. From remaining tickets, pick ONLY those where suggested_automata is NOT blank/null
    
    3. For tickets with suggested_automata = "unknown" or "other requests":
       - Look at the "label" field and map to suggested_automata using LABEL_TO_AUTOMATA_MAPPING
    
    4. Exclude tickets whose final automata is "other-handler", "unknown", or "other requests"
    
    5. All remaining tickets = automation opportunity
    
    6. Group by broader categories (IAM, Infra, Hardware, Capacity, Mailbox, Others)
    """
    result = {
        "total_tickets_analyzed": int(len(df)),
        "automation_opportunity_count": 0,
        "automation_opportunity_percentage": 0.0,
        "opportunities_by_automata": {},
        "opportunities_by_category": {},
        "potential_time_savings_hours": 0.0,
        "analysis_method": "suggested_automata_with_label_mapping",
        "exclusions": {},
        "label_mappings_applied": 0,
    }
    
    if len(df) == 0:
        return result
    
    working_df = df.copy()
    
    # Step 1: Exclude ONLY tickets with these specific closure codes
    excluded_closure_codes = [
        "remediation with corrective closure",
        "remediation with validation closure",
    ]
    
    closure_excluded_count = 0
    if "closure_code" in working_df.columns:
        closure_series = working_df["closure_code"].fillna("").astype(str).str.strip().str.lower()
        exclusion_mask = closure_series.isin(excluded_closure_codes)
        closure_excluded_count = int(exclusion_mask.sum())
        working_df = working_df[~exclusion_mask].copy()
    
    result["exclusions"]["closure_code_excluded"] = closure_excluded_count
    result["tickets_after_closure_exclusion"] = int(len(working_df))
    
    # Step 1b: Exclude MAINFRAME tickets (ostype column)
    mainframe_excluded_count = 0
    if "ostype" in working_df.columns:
        ostype_series = working_df["ostype"].fillna("").astype(str).str.strip().str.upper()
        mainframe_mask = ostype_series == "MAINFRAME"
        mainframe_excluded_count = int(mainframe_mask.sum())
        working_df = working_df[~mainframe_mask].copy()
    
    result["exclusions"]["mainframe_excluded"] = mainframe_excluded_count
    result["tickets_after_closure_exclusion"] = int(len(working_df))
    
    if len(working_df) == 0:
        result["analysis_note"] = "All tickets were excluded due to closure codes or mainframe ostype."
        return result
    
    # Check for required columns
    if "suggested_automata" not in working_df.columns:
        result["analysis_note"] = "No 'suggested_automata' column found in dataset."
        result["analysis_method"] = "suggested_automata_column_missing"
        return result
    
    # Step 2: Pick ONLY tickets where suggested_automata is NOT blank/null
    # Normalize suggested_automata
    working_df["_suggested_automata_normalized"] = (
        working_df["suggested_automata"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    
    # Blank values that mean "no suggested automata"
    blank_values = {"", "nan", "none", "null", "na", "n/a"}
    
    # Filter to ONLY tickets with non-blank suggested_automata
    has_automata_mask = ~working_df["_suggested_automata_normalized"].isin(blank_values)
    tickets_with_automata = working_df[has_automata_mask].copy()
    tickets_without_automata_count = int((~has_automata_mask).sum())
    
    result["exclusions"]["no_suggested_automata"] = tickets_without_automata_count
    result["tickets_with_suggested_automata"] = int(len(tickets_with_automata))
    
    if len(tickets_with_automata) == 0:
        result["analysis_note"] = "No tickets have a valid suggested_automata value."
        return result
    
    # Initialize label column normalization if label exists
    if "label" in tickets_with_automata.columns:
        tickets_with_automata["_label_normalized"] = (
            tickets_with_automata["label"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        tickets_with_automata["_label_normalized"] = ""
    
    # Step 3: For tickets with "unknown" or "other requests", apply label mapping
    needs_label_mapping = {"unknown", "other requests"}
    
    label_mappings_applied = 0
    label_mapping_details = {}
    
    def determine_final_automata(row):
        nonlocal label_mappings_applied
        current_automata = row["_suggested_automata_normalized"]
        
        # Only apply label mapping if automata is "unknown" or "other requests"
        if current_automata in needs_label_mapping:
            label_norm = row["_label_normalized"]
            if label_norm in LABEL_TO_AUTOMATA_MAPPING:
                mapped_automata = LABEL_TO_AUTOMATA_MAPPING[label_norm]
                # Track the mapping
                if label_norm not in label_mapping_details:
                    label_mapping_details[label_norm] = {"mapped_to": mapped_automata, "count": 0}
                label_mapping_details[label_norm]["count"] += 1
                label_mappings_applied += 1
                return mapped_automata
        
        # Return original automata (including unknown/other requests if no label mapping found)
        return current_automata
    
    # Apply the mapping
    tickets_with_automata["_final_automata"] = tickets_with_automata.apply(determine_final_automata, axis=1)
    
    result["label_mappings_applied"] = label_mappings_applied
    result["label_mapping_details"] = label_mapping_details
    
    # Step 4: Exclude tickets with automata in EXCLUDED_AUTOMATA (after mapping)
    excluded_automata_normalized = {ea.lower() for ea in EXCLUDED_AUTOMATA}
    excluded_automata_mask = tickets_with_automata["_final_automata"].isin(excluded_automata_normalized)
    excluded_automata_count = int(excluded_automata_mask.sum())
    
    # Track which automata types were excluded
    excluded_automata_breakdown = (
        tickets_with_automata[excluded_automata_mask]["_final_automata"]
        .value_counts()
        .to_dict()
    )
    
    result["exclusions"]["excluded_automata_types"] = excluded_automata_breakdown
    result["exclusions"]["total_excluded_automata"] = excluded_automata_count
    
    # Step 5: Final automation opportunity = tickets NOT excluded by automata type
    automatable_tickets = tickets_with_automata[~excluded_automata_mask].copy()
    
    opp_count = int(len(automatable_tickets))
    # Percentage is based on tickets AFTER closure exclusion (not just those with automata)
    total_after_closure = result["tickets_after_closure_exclusion"]
    opp_pct = round(opp_count / total_after_closure * 100, 2) if total_after_closure > 0 else 0.0
    
    # Also calculate percentage based on tickets with suggested_automata
    tickets_with_automata_count = result["tickets_with_suggested_automata"]
    opp_pct_of_automata = round(opp_count / tickets_with_automata_count * 100, 2) if tickets_with_automata_count > 0 else 0.0
    
    # Calculate opportunities by automata type
    opportunities_by_automata = {}
    automata_counts = {}
    if len(automatable_tickets) > 0:
        automata_counts = automatable_tickets["_final_automata"].value_counts().to_dict()
        opportunities_by_automata = {str(k): int(v) for k, v in automata_counts.items()}
    
    result["opportunities_by_automata"] = opportunities_by_automata
    
    # Build automata-to-category lookup for frontend display
    automata_category_map = {}
    for automata_name in opportunities_by_automata.keys():
        automata_category_map[automata_name] = AUTOMATA_TO_CATEGORY.get(automata_name, "Others")
    result["automata_category_map"] = automata_category_map
    
    # Step 6: Group by broader categories
    opportunities_by_category = {"IAM": 0, "Infra": 0, "Hardware": 0, "Capacity": 0, "Mailbox": 0, "Others": 0}
    
    if len(automatable_tickets) > 0:
        for automata, count in automata_counts.items():
            category = AUTOMATA_TO_CATEGORY.get(automata, "Others")
            opportunities_by_category[category] += count
    
    # Remove zero categories
    result["opportunities_by_category"] = {k: v for k, v in opportunities_by_category.items() if v > 0}
    
    # Calculate top categories with percentage based on AUTOMATABLE tickets (not total)
    # Sort categories by count and calculate percentage
    sorted_categories = sorted(
        [(k, v) for k, v in opportunities_by_category.items() if v > 0],
        key=lambda x: x[1],
        reverse=True
    )
    
    top_categories_list = []
    for cat_name, cat_count in sorted_categories[:5]:  # Top 5 categories
        cat_pct = round(cat_count / opp_count * 100, 1) if opp_count > 0 else 0.0
        top_categories_list.append({
            "category": cat_name,
            "count": cat_count,
            "percentage_of_automatable": cat_pct
        })
    
    result["top_categories_list"] = top_categories_list
    
    # Calculate total for top categories (for PPT display)
    top_cats_total = sum(c["count"] for c in top_categories_list)
    top_cats_pct = round(top_cats_total / opp_count * 100, 1) if opp_count > 0 else 0.0
    result["top_categories_total"] = top_cats_total
    result["top_categories_percentage"] = top_cats_pct  # This is % of automatable, not total
    result["top_categories_names"] = ", ".join(c["category"] for c in top_categories_list[:3])
    
    # Calculate potential time savings
    potential_time_savings = 0.0
    if "mttr_excl_hold" in automatable_tickets.columns and len(automatable_tickets) > 0:
        potential_time_savings = float(automatable_tickets["mttr_excl_hold"].sum())
    
    result["automation_opportunity_count"] = opp_count
    result["automation_opportunity_percentage"] = float(opp_pct)
    result["automation_opportunity_pct_of_automata_tickets"] = float(opp_pct_of_automata)
    result["potential_time_savings_hours"] = round(potential_time_savings, 2)
    
    # Track currently automated tickets (those with remediation closure codes)
    result["currently_automated_count"] = closure_excluded_count
    result["currently_automated_percentage"] = round(closure_excluded_count / result["total_tickets_analyzed"] * 100, 1) if result["total_tickets_analyzed"] > 0 else 0.0
    
    # Additional breakdown for reporting
    if "priority_df" in automatable_tickets.columns and len(automatable_tickets) > 0:
        sev_breakdown = automatable_tickets["priority_df"].value_counts().to_dict()
        result["automation_by_severity"] = {
            f"Severity {int(k)}" if not pd.isna(k) else "Unknown": int(v)
            for k, v in sev_breakdown.items()
        }
    
    # Closure code distribution of automatable tickets (for reference)
    if "closure_code" in automatable_tickets.columns and len(automatable_tickets) > 0:
        result["automatable_closure_codes"] = (
            automatable_tickets["closure_code"]
            .fillna("(blank)")
            .value_counts()
            .head(10)
            .to_dict()
        )
    
    result["analysis_note"] = (
        f"Analysis started with {total_after_closure:,} tickets (after excluding {closure_excluded_count:,} with automated closure codes). "
        f"Of these, {tickets_with_automata_count:,} have a suggested_automata value. "
        f"{label_mappings_applied:,} tickets with 'unknown'/'other requests' were remapped via label field. "
        f"{excluded_automata_count:,} tickets excluded due to 'other-handler'/'unknown'/'other requests' automata. "
        f"Final automation opportunity: {opp_count:,} tickets ({opp_pct}% of all, {opp_pct_of_automata}% of tickets with automata)."
    )
    
    return result


# =========================
# Resolution Metrics
# =========================
def analyze_resolution_metrics(df):
    results = {}
    if "resolution_code" in df.columns:
        results["resolution_code_distribution"] = df["resolution_code"].value_counts().head(10).to_dict()
    if "category" in df.columns:
        results["category_distribution"] = df["category"].value_counts().head(10).to_dict()
    if "mttr_excl_hold" in df.columns:
        results["mttr_statistics"] = {
            "mean": float(df["mttr_excl_hold"].mean()),
            "median": float(df["mttr_excl_hold"].median()),
            "min": float(df["mttr_excl_hold"].min()),
            "max": float(df["mttr_excl_hold"].max()),
            "std": float(df["mttr_excl_hold"].std()),
        }
    if "open_dttm" in df.columns and pd.api.types.is_datetime64_any_dtype(df["open_dttm"]):
        df_copy = add_month_col(df.copy())
        monthly_counts = df_copy["month"].value_counts().sort_index()
        results["monthly_ticket_trend"] = {
            "months": [str(m) for m in monthly_counts.index],
            "counts": monthly_counts.values.tolist(),
        }
    return results


# =========================
# Top Applications
# =========================
def analyze_top_applications(df, top_n=3, industry=None, currency="USD"):
    if "business_application" not in df.columns:
        return {}

    if industry and industry in INDUSTRY_DOWNTIME_COSTS:
        downtime_cost_rate = convert_currency(INDUSTRY_DOWNTIME_COSTS[industry], currency)
    else:
        downtime_cost_rate = None

    app_counts = df["business_application"].value_counts().head(top_n)
    results = {}

    for app_name in app_counts.index:
        app_df = df[df["business_application"] == app_name]
        downtime_df = app_df[app_df["priority_df"].isin([1, 2])] if "priority_df" in app_df.columns else app_df.iloc[0:0]
        noise_df = app_df[app_df["priority_df"].isin([3, 4])] if "priority_df" in app_df.columns else app_df.iloc[0:0]
        projected_yearly_downtime = calc_projected_yearly_hours(downtime_df)
        projected_yearly_noise = calc_projected_yearly_hours(noise_df)
        avg_monthly_downtime = round(projected_yearly_downtime / 12, 2)
        avg_monthly_noise = round(projected_yearly_noise / 12, 2)
        projected_yearly_downtime_cost = round(projected_yearly_downtime * downtime_cost_rate, 0) if downtime_cost_rate is not None else None
        noise_cost_rate = convert_currency(NOISE_COST_PER_HOUR, currency)
        projected_yearly_noise_cost = round(projected_yearly_noise * noise_cost_rate, 0)

        sev_breakdown = {}
        if "priority_df" in app_df.columns:
            for k, v in app_df["priority_df"].value_counts().to_dict().items():
                if pd.isna(k):
                    continue
                try:
                    sev_breakdown[int(k)] = int(v)
                except Exception:
                    sev_breakdown[str(k)] = int(v)

        results[str(app_name)] = {
            "total_tickets": int(app_counts[app_name]),
            "severity_breakdown": sev_breakdown,
            "downtime": {
                "ticket_count": int(len(downtime_df)),
                "avg_monthly_hours": avg_monthly_downtime,
                "projected_yearly_hours": projected_yearly_downtime,
                "projected_yearly_cost": projected_yearly_downtime_cost,
            },
            "noise": {
                "ticket_count": int(len(noise_df)),
                "avg_monthly_hours": avg_monthly_noise,
                "projected_yearly_hours": projected_yearly_noise,
                "projected_yearly_cost": projected_yearly_noise_cost,
            },
        }

    return results


# =========================
# All Applications Costs
# =========================
def analyze_all_applications_costs(df, industry=None, currency="USD"):
    if "business_application" not in df.columns:
        return {}

    if industry and industry in INDUSTRY_DOWNTIME_COSTS:
        downtime_cost_rate = convert_currency(INDUSTRY_DOWNTIME_COSTS[industry], currency)
    else:
        downtime_cost_rate = None

    all_apps = df["business_application"].unique()
    results = {}

    for app_name in all_apps:
        if pd.isna(app_name):
            continue
        app_df = df[df["business_application"] == app_name]
        downtime_df = app_df[app_df["priority_df"].isin([1, 2])] if "priority_df" in app_df.columns else app_df.iloc[0:0]
        noise_df = app_df[app_df["priority_df"].isin([3, 4])] if "priority_df" in app_df.columns else app_df.iloc[0:0]

        projected_yearly_downtime = calc_projected_yearly_hours(downtime_df)
        projected_yearly_noise = calc_projected_yearly_hours(noise_df)

        projected_yearly_downtime_cost = round(projected_yearly_downtime * downtime_cost_rate, 0) if downtime_cost_rate is not None else None
        noise_cost_rate = convert_currency(NOISE_COST_PER_HOUR, currency)
        projected_yearly_noise_cost = round(projected_yearly_noise * noise_cost_rate, 0)

        results[str(app_name)] = {
            "application": str(app_name),
            "total_tickets": int(len(app_df)),
            "downtime": {
                "ticket_count": int(len(downtime_df)),
                "projected_yearly_hours": projected_yearly_downtime,
                "projected_yearly_cost": projected_yearly_downtime_cost,
            },
            "noise": {
                "ticket_count": int(len(noise_df)),
                "projected_yearly_hours": projected_yearly_noise,
                "projected_yearly_cost": projected_yearly_noise_cost,
            }
        }

    return dict(sorted(
        results.items(),
        key=lambda x: (x[1]["downtime"]["projected_yearly_hours"] + x[1]["noise"]["projected_yearly_hours"]),
        reverse=True
    ))


# =========================
# Key Highlights (Incident)
# =========================
def compute_key_highlights(analysis_result, df):
    """
    Scan all incident analysis outputs and surface only genuinely actionable
    findings.  Each highlight is a dict:
        icon         – string key for SVG icon lookup on frontend
        title        – short headline
        metric       – bold number / percentage
        reason       – one-liner explaining *why* this was picked
        detail_lines – list of strings for expanded detail view
        navigate_to  – inner tab name to jump to for full data
        category     – tag for grouping (AUTOMATION, COST, MTTR, VOLUME, ROUTING, TREND, TIMING)
        severity     – "critical" | "warning" | "info"
    """
    highlights = []
    total_tickets = analysis_result.get("metadata", {}).get("total_tickets", 0)
    months = analysis_result.get("metadata", {}).get("months_of_data", 1) or 1
    if total_tickets == 0:
        return highlights

    # ── 1. AUTOMATION OPPORTUNITY ──────────────────────────────────────
    ao = analysis_result.get("automation_opportunity", {})
    opp_pct = ao.get("automation_opportunity_percentage", 0)
    opp_count = ao.get("automation_opportunity_count", 0)
    if opp_pct >= 20 and opp_count >= 10:
        savings = ao.get("automation_cost_savings") or {}
        hrs = savings.get("annual_hours_saved", 0)
        low = savings.get("cost_savings_low", 0)
        high = savings.get("cost_savings_high", 0)
        detail = f"{opp_count:,} tickets can be automated"
        if hrs > 0:
            detail += f" · ~{hrs:,.0f} hrs/yr saved"
        if low > 0 and high > 0:
            detail += f" · ${low:,.0f}–${high:,.0f}/yr"
        sev = "critical" if opp_pct >= 40 else "warning"

        detail_lines = [
            f"Total tickets analysed (after exclusions): {ao.get('tickets_after_closure_exclusion', '—'):,}",
            f"Tickets with suggested automata: {ao.get('tickets_with_suggested_automata', '—'):,}",
            f"Label-based remappings applied: {ao.get('label_mappings_applied', 0):,}",
            f"Currently auto-remediated: {ao.get('currently_automated_count', 0):,} ({ao.get('currently_automated_percentage', 0):.1f}%)",
        ]
        excl = ao.get("exclusions", {})
        if excl.get("mainframe_excluded", 0) > 0:
            detail_lines.insert(1, f"Mainframe tickets excluded: {excl['mainframe_excluded']:,}")
        # Top categories
        top_cats = ao.get("top_categories_list", [])
        if top_cats:
            detail_lines.append("Top automatable categories:")
            for cat in top_cats[:5]:
                detail_lines.append(f"  {cat['category']}: {cat['count']:,} tickets ({cat['percentage_of_automatable']:.1f}% of automatable)")
        if hrs > 0:
            detail_lines.append(f"Projected annual hours saved: {hrs:,.0f} hrs")
        if low > 0 and high > 0:
            detail_lines.append(f"Estimated annual cost savings: ${low:,.0f} – ${high:,.0f}")

        highlights.append({
            "icon": "automation", "title": "Automation Opportunity",
            "metric": f"{opp_pct:.1f}%",
            "reason": detail,
            "detail_lines": detail_lines,
            "navigate_to": "automation",
            "category": "AUTOMATION", "severity": sev,
        })

    # ── 2. HIGH-SEVERITY CONCENTRATION ─────────────────────────────────
    sa = analysis_result.get("severity_analysis", {})
    hp_pct = sa.get("high_priority_percentage", 0)
    hp_count = sa.get("high_priority_count", 0)
    if hp_pct >= 15 and hp_count >= 5:
        sev_counts = sa.get("severity_counts", {})
        mttr_by_sev = sa.get("mttr_by_severity", {})
        detail_lines = [
            f"Sev 1: {sev_counts.get(1, sev_counts.get('1', 0)):,} tickets",
            f"Sev 2: {sev_counts.get(2, sev_counts.get('2', 0)):,} tickets",
        ]
        for s_key in ["1", "2", 1, 2]:
            sev_mttr = mttr_by_sev.get(s_key) or mttr_by_sev.get(str(s_key))
            if sev_mttr and isinstance(sev_mttr, dict):
                detail_lines.append(f"Sev {s_key} avg MTTR: {sev_mttr.get('mean', 0):.1f} hrs (median {sev_mttr.get('median', 0):.1f} hrs, {int(sev_mttr.get('count', 0)):,} tickets)")
        detail_lines.append(f"High-severity tickets represent {hp_pct:.1f}% of total volume — these drive the bulk of downtime cost")
        highlights.append({
            "icon": "severity", "title": "High-Severity Concentration",
            "metric": f"{hp_pct:.1f}%",
            "reason": f"{hp_count:,} Sev 1-2 tickets — significant share of total volume driving downtime",
            "detail_lines": detail_lines,
            "navigate_to": "severity",
            "category": "VOLUME", "severity": "critical" if hp_pct >= 30 else "warning",
        })

    # ── 3. TOP HOSTNAME BY TICKETS (volume-dominant) ───────────────────
    ha = analysis_result.get("hostname_analysis", {})
    top_hosts = ha.get("top_hostnames", {})
    if top_hosts:
        sorted_hosts = sorted(top_hosts.items(), key=lambda x: x[1], reverse=True)
        top_host, top_count = sorted_hosts[0]
        top_host_pct = round(top_count / total_tickets * 100, 1)
        if top_host_pct >= 5 and top_count >= 20:
            detail_lines = [f"#{i+1}  {h}: {c:,} tickets ({c/total_tickets*100:.1f}%)" for i, (h, c) in enumerate(sorted_hosts[:5])]
            detail_lines.append(f"Top 5 hosts account for {sum(c for _, c in sorted_hosts[:5]):,} tickets ({sum(c for _, c in sorted_hosts[:5])/total_tickets*100:.1f}% of total)")
            highlights.append({
                "icon": "server", "title": "Dominant Host by Volume",
                "metric": f"{top_count:,} tickets ({top_host_pct}%)",
                "reason": f"\"{top_host}\" alone generates {top_host_pct}% of all incidents — investigate root cause",
                "detail_lines": detail_lines,
                "navigate_to": "hostnames",
                "category": "VOLUME", "severity": "warning" if top_host_pct >= 10 else "info",
            })

    # ── 4. MTTR OUTLIER HOSTS (high MTTR with enough tickets) ──────────
    ctx = analysis_result.get("llm_context", {})
    overall_mttr = (ctx.get("overall_mttr_stats") or {}).get("mean", 0)
    top_mttr_hosts = ctx.get("top_hostnames_by_highest_avg_mttr", [])
    if overall_mttr > 0 and top_mttr_hosts:
        flagged = [h for h in top_mttr_hosts
                    if h.get("avg_mttr_hours", 0) >= overall_mttr * 3
                    and h.get("ticket_count", 0) >= 5]
        if flagged:
            worst = flagged[0]
            ratio = round(worst["avg_mttr_hours"] / overall_mttr, 1)
            detail_lines = [
                f"Overall average MTTR: {overall_mttr:.1f} hrs",
                f"Flagged hosts (≥3× avg MTTR AND ≥5 tickets):",
            ]
            for h in flagged[:5]:
                r = round(h["avg_mttr_hours"] / overall_mttr, 1)
                detail_lines.append(f"  {h['hostname']}: {h['avg_mttr_hours']:.1f} hrs avg ({h['ticket_count']} tickets, {r}× slower)")
            detail_lines.append("Only hosts with 5+ tickets are flagged — low-volume outliers are excluded")
            highlights.append({
                "icon": "clock", "title": "MTTR Outlier Host",
                "metric": f"{worst['avg_mttr_hours']:.1f} hrs avg",
                "reason": f"\"{worst['hostname']}\" ({worst['ticket_count']} tickets) resolves {ratio}× slower than average ({overall_mttr:.1f} hrs)",
                "detail_lines": detail_lines,
                "navigate_to": "hostnames",
                "category": "MTTR", "severity": "warning",
            })

    # ── 5. MTTR OUTLIER APPS ───────────────────────────────────────────
    top_mttr_apps = ctx.get("applications_highest_avg_mttr", [])
    if overall_mttr > 0 and top_mttr_apps:
        flagged_apps = [a for a in top_mttr_apps
                        if a.get("avg_mttr_hours", 0) >= overall_mttr * 3
                        and a.get("ticket_count", 0) >= 5]
        if flagged_apps:
            worst_app = flagged_apps[0]
            ratio = round(worst_app["avg_mttr_hours"] / overall_mttr, 1)
            detail_lines = [
                f"Overall average MTTR: {overall_mttr:.1f} hrs",
                f"Flagged applications (≥3× avg MTTR AND ≥5 tickets):",
            ]
            for a in flagged_apps[:5]:
                r = round(a["avg_mttr_hours"] / overall_mttr, 1)
                detail_lines.append(f"  {a['application']}: {a['avg_mttr_hours']:.1f} hrs avg ({a['ticket_count']} tickets, {r}× slower)")
            highlights.append({
                "icon": "app", "title": "Slow-Resolving Application",
                "metric": f"{worst_app['avg_mttr_hours']:.1f} hrs avg",
                "reason": f"\"{worst_app['application']}\" ({worst_app['ticket_count']} tickets) takes {ratio}× longer than average to resolve",
                "detail_lines": detail_lines,
                "navigate_to": "applications",
                "category": "MTTR", "severity": "warning",
            })

    # ── 6. REASSIGNMENT HOTSPOT ────────────────────────────────────────
    trends = analysis_result.get("trends", {})
    reass_stats = trends.get("reassignment_stats", {})
    overall_reass_mean = reass_stats.get("mean", 0)
    reass_by_sev = trends.get("reassignment_by_severity", [])
    if reass_by_sev and overall_reass_mean > 0:
        for entry in reass_by_sev:
            if entry.get("severity") in [1, 2] and entry.get("avg_reassignments", 0) >= 3 and entry.get("count", 0) >= 5:
                detail_lines = [
                    f"Overall avg reassignments: {overall_reass_mean:.2f}",
                    f"Reassignment breakdown by severity:",
                ]
                for e in reass_by_sev:
                    if e.get("count", 0) > 0:
                        detail_lines.append(f"  Sev {e['severity']}: {e['avg_reassignments']:.2f} avg ({e['count']:,} tickets)")
                zero_pct = reass_stats.get("zero_pct", 0)
                detail_lines.append(f"Tickets with zero reassignments: {zero_pct:.1f}%")
                detail_lines.append("High reassignment count on critical tickets indicates routing confusion or unclear ownership")
                highlights.append({
                    "icon": "shuffle", "title": "High-Severity Ticket Bouncing",
                    "metric": f"{entry['avg_reassignments']:.1f} avg reassignments",
                    "reason": f"Sev {entry['severity']} tickets ({entry['count']:,}) are reassigned {entry['avg_reassignments']:.1f}× on avg — slows resolution of critical incidents",
                    "detail_lines": detail_lines,
                    "navigate_to": "trends",
                    "category": "ROUTING", "severity": "warning",
                })
                break

    # ── 7. ASSIGNMENT GROUP WITH DISPROPORTIONATE VOLUME ───────────────
    agp_top = trends.get("assignment_grp_parent_top", [])
    if agp_top and total_tickets > 0:
        top_grp = agp_top[0]
        grp_pct = top_grp.get("pct", 0)
        if grp_pct >= 25 and top_grp.get("count", 0) >= 20:
            detail_lines = ["Top assignment groups by volume:"]
            for g in agp_top[:7]:
                detail_lines.append(f"  {g['group']}: {g['count']:,} tickets ({g['pct']:.1f}%)")
            highlights.append({
                "icon": "users", "title": "Dominant Assignment Group",
                "metric": f"{grp_pct:.0f}% of all tickets",
                "reason": f"\"{top_grp['group']}\" handles {top_grp['count']:,} tickets — potential bottleneck or scope imbalance",
                "detail_lines": detail_lines,
                "navigate_to": "trends",
                "category": "ROUTING", "severity": "info",
            })

    # ── 8. ASSIGNMENT GROUP WITH SLOW MTTR ─────────────────────────────
    agp_mttr = trends.get("assignment_grp_parent_by_mttr", [])
    if agp_mttr and overall_mttr > 0:
        flagged_grps = [g for g in agp_mttr
                        if g.get("avg_mttr", 0) >= overall_mttr * 2.5
                        and g.get("count", 0) >= 10]
        if flagged_grps:
            worst_g = flagged_grps[0]
            ratio = round(worst_g["avg_mttr"] / overall_mttr, 1)
            detail_lines = [
                f"Overall avg MTTR: {overall_mttr:.1f} hrs",
                "Flagged groups (≥2.5× avg MTTR AND ≥10 tickets):",
            ]
            for g in flagged_grps[:5]:
                r = round(g["avg_mttr"] / overall_mttr, 1)
                detail_lines.append(f"  {g['group']}: {g['avg_mttr']:.1f} hrs avg ({g['count']} tickets, {r}× slower)")
            highlights.append({
                "icon": "snail", "title": "Slow Assignment Group",
                "metric": f"{worst_g['avg_mttr']:.1f} hrs avg MTTR",
                "reason": f"\"{worst_g['group']}\" ({worst_g['count']} tickets) resolves {ratio}× slower than overall average",
                "detail_lines": detail_lines,
                "navigate_to": "trends",
                "category": "MTTR", "severity": "warning",
            })

    # ── 9. MONTHLY TREND — SPIKE OR CLIMB ─────────────────────────────
    monthly_trend = ctx.get("monthly_ticket_trend", [])
    if len(monthly_trend) >= 3:
        counts = [m["tickets"] for m in monthly_trend]
        avg_count = sum(counts) / len(counts)
        last_3_avg = sum(counts[-3:]) / 3
        if avg_count > 0 and last_3_avg > avg_count * 1.3 and last_3_avg >= 20:
            pct_rise = round((last_3_avg - avg_count) / avg_count * 100, 0)
            detail_lines = ["Monthly ticket volumes:"]
            for m in monthly_trend:
                marker = " <<<" if m["tickets"] == max(counts) else ""
                detail_lines.append(f"  {m['month']}: {m['tickets']:,}{marker}")
            detail_lines.append(f"Overall monthly avg: {avg_count:.0f}")
            detail_lines.append(f"Last 3 months avg: {last_3_avg:.0f} (+{pct_rise:.0f}% above overall)")
            highlights.append({
                "icon": "trending_up", "title": "Rising Incident Trend",
                "metric": f"+{pct_rise:.0f}% vs average",
                "reason": f"Last 3 months avg {last_3_avg:.0f} tickets/mo vs overall {avg_count:.0f} — investigate whether a new issue pattern is emerging",
                "detail_lines": detail_lines,
                "navigate_to": "charts",
                "category": "TREND", "severity": "warning",
            })
        if avg_count > 10:
            max_count = max(counts)
            max_idx = counts.index(max_count)
            if max_count >= avg_count * 2:
                spike_month = monthly_trend[max_idx]["month"]
                detail_lines = ["Monthly ticket volumes:"]
                for m in monthly_trend:
                    marker = " <<< SPIKE" if m["tickets"] == max_count else ""
                    detail_lines.append(f"  {m['month']}: {m['tickets']:,}{marker}")
                detail_lines.append(f"Overall monthly avg: {avg_count:.0f}")
                detail_lines.append(f"Spike month volume: {max_count:,} ({round(max_count/avg_count, 1)}× average)")
                highlights.append({
                    "icon": "zap", "title": "Ticket Volume Spike",
                    "metric": f"{max_count:,} tickets in {spike_month}",
                    "reason": f"That month had {round(max_count/avg_count, 1)}× the average volume — check for outage or change-related surge",
                    "detail_lines": detail_lines,
                    "navigate_to": "charts",
                    "category": "TREND", "severity": "info",
                })

    # ── 10. PEAK TIMING CONCENTRATION ──────────────────────────────────
    timing = analysis_result.get("timing_analysis", {})
    if not timing.get("error", True):
        peak_hour = timing.get("peak_hour")
        peak_day = timing.get("peak_day")
        hourly = timing.get("hourly_distribution", {})
        daily = timing.get("daily_distribution", {})
        if peak_hour is not None and hourly:
            peak_val = hourly.get(str(peak_hour), 0)
            total_parsed = timing.get("parsed_count", 1)
            peak_hour_pct = round(peak_val / total_parsed * 100, 1) if total_parsed > 0 else 0
            if peak_hour_pct >= 10:
                detail_lines = [
                    f"Peak hour: {peak_hour}:00 — {peak_val:,} tickets ({peak_hour_pct}%)",
                    f"Peak day: {peak_day or '—'}",
                    "Hourly distribution (top 5):",
                ]
                sorted_hours = sorted(hourly.items(), key=lambda x: x[1], reverse=True)[:5]
                for hr, cnt in sorted_hours:
                    detail_lines.append(f"  {hr}:00 — {cnt:,} tickets")
                if daily:
                    detail_lines.append("Daily distribution:")
                    sorted_days = sorted(daily.items(), key=lambda x: x[1], reverse=True)
                    for day, cnt in sorted_days:
                        detail_lines.append(f"  {day}: {cnt:,}")
                highlights.append({
                    "icon": "clock_peak", "title": "Peak Hour Concentration",
                    "metric": f"{peak_hour_pct}% at {peak_hour}:00",
                    "reason": f"{peak_val:,} tickets open at hour {peak_hour}:00 ({peak_day or ''} is busiest day) — consider staffing alignment",
                    "detail_lines": detail_lines,
                    "navigate_to": "charts",
                    "category": "TIMING", "severity": "info",
                })

    # ── 11. TOP APPLICATION DOWNTIME COST ──────────────────────────────
    top_apps = analysis_result.get("top_applications", {})
    if top_apps:
        for app_name, app_data in list(top_apps.items())[:1]:
            dt_cost = (app_data.get("downtime") or {}).get("projected_yearly_cost")
            dt_hrs = (app_data.get("downtime") or {}).get("projected_yearly_hours", 0)
            noise_cost = (app_data.get("noise") or {}).get("projected_yearly_cost", 0)
            sev_bk = app_data.get("severity_breakdown", {})
            if dt_cost is not None and dt_cost > 0 and dt_hrs > 0:
                currency_code = analysis_result.get("metadata", {}).get("currency", "USD")
                sym = "\u20ac" if currency_code == "EUR" else "$"
                detail_lines = [
                    f"Application: {app_name}",
                    f"Total tickets: {app_data.get('total_tickets', 0):,}",
                    f"Severity breakdown: {', '.join(f'Sev {k}: {v}' for k, v in sorted(sev_bk.items()))}",
                    f"Projected yearly downtime: {dt_hrs:.1f} hrs",
                    f"Projected yearly downtime cost: {sym}{dt_cost:,.0f}",
                ]
                if noise_cost:
                    detail_lines.append(f"Projected yearly noise cost: {sym}{noise_cost:,.0f}")
                highlights.append({
                    "icon": "dollar", "title": "Highest Downtime Cost Application",
                    "metric": f"{sym}{dt_cost:,.0f}/yr",
                    "reason": f"\"{app_name}\" projects {dt_hrs:.0f} downtime hrs/yr — top candidate for reliability investment",
                    "detail_lines": detail_lines,
                    "navigate_to": "applications",
                    "category": "COST", "severity": "critical" if dt_cost >= 500000 else "warning",
                })

    # ── 12. NOISE COST ─────────────────────────────────────────────────
    all_app_costs = analysis_result.get("all_applications_costs", {})
    total_noise_cost = sum(
        (v.get("noise") or {}).get("projected_yearly_cost", 0) or 0
        for v in all_app_costs.values()
    )
    if total_noise_cost >= 10000:
        currency_code = analysis_result.get("metadata", {}).get("currency", "USD")
        sym = "\u20ac" if currency_code == "EUR" else "$"
        # Top noise contributors
        noise_ranked = sorted(
            [(k, (v.get("noise") or {}).get("projected_yearly_cost", 0) or 0) for k, v in all_app_costs.items()],
            key=lambda x: x[1], reverse=True
        )
        detail_lines = [f"Total projected noise cost (Sev 3-4): {sym}{total_noise_cost:,.0f}/yr", "Top contributors:"]
        for app_name, ncost in noise_ranked[:5]:
            if ncost > 0:
                detail_lines.append(f"  {app_name}: {sym}{ncost:,.0f}/yr")
        detail_lines.append("Noise = Sev 3 & 4 tickets at $144/hr fixed rate")
        highlights.append({
            "icon": "megaphone", "title": "Projected Noise Cost (Sev 3-4)",
            "metric": f"{sym}{total_noise_cost:,.0f}/yr",
            "reason": "Aggregate cost of low-severity tickets across all applications — automation and self-service can reduce this",
            "detail_lines": detail_lines,
            "navigate_to": "applications",
            "category": "COST", "severity": "warning" if total_noise_cost >= 50000 else "info",
        })

    # ── 13. COMMON RECURRING ISSUE ─────────────────────────────────────
    ci = analysis_result.get("common_issues", {})
    top_descs = ci.get("top_descriptions", [])
    if top_descs and total_tickets > 0:
        top = top_descs[0]
        desc_pct = top.get("pct", 0)
        desc_count = top.get("count", 0)
        if desc_pct >= 3 and desc_count >= 15:
            desc_text = top.get("description", "")[:80]
            detail_lines = ["Top recurring descriptions:"]
            for d in top_descs[:7]:
                detail_lines.append(f"  \"{d['description'][:70]}\" — {d['count']:,} tickets ({d['pct']:.1f}%)")
            phrases = ci.get("common_issue_phrases", [])
            if phrases:
                detail_lines.append("Common keyword phrases:")
                for p in phrases[:5]:
                    detail_lines.append(f"  \"{p['phrase']}\" — {p['count']:,} occurrences")
            highlights.append({
                "icon": "repeat", "title": "Most Recurring Issue",
                "metric": f"{desc_count:,} tickets ({desc_pct:.1f}%)",
                "reason": f"\"{desc_text}\" — most common description; candidate for runbook or auto-remediation",
                "detail_lines": detail_lines,
                "navigate_to": "trends",
                "category": "VOLUME", "severity": "info",
            })

    # ── Sort: critical first, then warning, then info ──────────────────
    sev_order = {"critical": 0, "warning": 1, "info": 2}
    highlights.sort(key=lambda h: sev_order.get(h.get("severity", "info"), 9))

    return highlights


def perform_analysis_on_dataframe(df, analysis_type="complete", industry=None, currency="USD"):
    months_of_data = get_months_in_data(df)
    automation_opp = analyze_automation_opportunity(df)

    # ── Automation cost savings: 20 min per ticket, annualised, × rate range ──
    opp_count = automation_opp.get("automation_opportunity_count", 0)
    if opp_count > 0 and months_of_data > 0:
        hours_in_period = opp_count * (20 / 60)  # 20 min per ticket
        annual_hours = round(hours_in_period / months_of_data * 12, 2)
        cost_low = round(annual_hours * 8.7, 0)
        cost_high = round(annual_hours * 20.7, 0)
        automation_opp["automation_cost_savings"] = {
            "hours_in_period": round(hours_in_period, 2),
            "annual_hours_saved": annual_hours,
            "cost_savings_low": cost_low,
            "cost_savings_high": cost_high,
            "rate_low": 8.7,
            "rate_high": 20.7,
            "months_of_data": months_of_data,
        }
    else:
        automation_opp["automation_cost_savings"] = None

    hostname_analysis = analyze_hostnames(df)
    automata_analysis = analyze_suggested_automata(df)
    severity_analysis = analyze_severity(df)
    top_applications = analyze_top_applications(df, industry=industry, currency=currency)
    all_applications_costs = analyze_all_applications_costs(df, industry=industry, currency=currency)
    timing_analysis = analyze_ticket_timing(df)
    resolution_metrics = analyze_resolution_metrics(df)
    common_issues = analyze_common_issues(df)
    trends = analyze_trends(df)
    llm_context = build_llm_context(df)

    # Build result dict first, then compute highlights from it
    result = {
        "metadata": {
            "total_tickets": int(len(df)),
            "analysis_type": analysis_type,
            "industry": industry,
            "currency": currency,
            "date_range": {
                "start": df["open_dttm"].min().isoformat() if "open_dttm" in df.columns and not df["open_dttm"].isna().all() else None,
                "end": df["open_dttm"].max().isoformat() if "open_dttm" in df.columns and not df["open_dttm"].isna().all() else None,
            },
            "months_of_data": months_of_data,
        },
        "hostname_analysis": hostname_analysis,
        "automata_analysis": automata_analysis,
        "severity_analysis": severity_analysis,
        "top_applications": top_applications,
        "all_applications_costs": all_applications_costs,
        "timing_analysis": timing_analysis,
        "automation_opportunity": automation_opp,
        "resolution_metrics": resolution_metrics,
        "common_issues": common_issues,
        "trends": trends,
        "llm_context": llm_context,
    }

    # Compute key highlights
    result["key_highlights"] = compute_key_highlights(result, df)

    return result


# =========================
# Managed / Non-Managed Split
# =========================
def split_managed_nonmanaged(df):
    if "sso_ticket" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    sso = df["sso_ticket"].fillna("").astype(str).str.upper()
    return df[sso == "Y"].copy(), df[sso == "N"].copy()


def build_comparison_data(managed_df, non_managed_df):
    def _stats(df, label):
        total = len(df)
        unique_hosts = int(df["hostname"].nunique()) if "hostname" in df.columns else 0
        unique_apps = int(df["business_application"].nunique()) if "business_application" in df.columns else 0
        avg_mttr = round(float(df["mttr_excl_hold"].mean()), 2) if "mttr_excl_hold" in df.columns and len(df) > 0 else 0.0
        sev12 = int(df[df["priority_df"].isin([1, 2])].shape[0]) if "priority_df" in df.columns else 0
        sev34 = int(df[df["priority_df"].isin([3, 4])].shape[0]) if "priority_df" in df.columns else 0
        months = get_months_in_data(df)
        incidents_per_month = round(total / months, 2) if months > 0 else 0.0
        incidents_per_device = round(total / unique_hosts, 2) if unique_hosts > 0 else 0.0
        mttr_by_sev = {}
        if "priority_df" in df.columns and "mttr_excl_hold" in df.columns:
            grp = df.dropna(subset=["mttr_excl_hold"]).groupby("priority_df")["mttr_excl_hold"].mean().round(2)
            mttr_by_sev = {str(int(k)) if not pd.isna(k) else "?": round(float(v), 2) for k, v in grp.items()}
        monthly_trend = []
        if "open_dttm" in df.columns and len(df) > 0:
            df_copy = add_month_col(df.dropna(subset=["open_dttm"]).copy())
            monthly = df_copy["month"].value_counts().sort_index()
            monthly_trend = [{"month": str(m), "tickets": int(c)} for m, c in monthly.items()]
        return {
            "label": label, "total_tickets": total, "unique_hostnames": unique_hosts,
            "unique_applications": unique_apps, "avg_mttr": avg_mttr,
            "sev1_2_tickets": sev12, "sev3_4_tickets": sev34,
            "incidents_per_month": incidents_per_month, "incidents_per_device": incidents_per_device,
            "mttr_by_severity": mttr_by_sev, "months_of_data": months, "monthly_trend": monthly_trend,
        }

    managed_stats = _stats(managed_df, "Managed") if len(managed_df) > 0 else None
    non_managed_stats = _stats(non_managed_df, "Non-Managed") if len(non_managed_df) > 0 else None
    return {"managed": managed_stats, "non_managed": non_managed_stats}


# =========================
# LLM Context Builder
# =========================
def build_llm_context(df):
    ctx = {}

    if "hostname" in df.columns:
        vc = df["hostname"].fillna("(blank)").value_counts().head(20)
        ctx["top_hostnames_by_ticket_count"] = [
            {"rank": i+1, "hostname": str(k), "tickets": int(v)}
            for i, (k, v) in enumerate(vc.items())
        ]
        ctx["tickets_by_hostname_top20"] = ctx["top_hostnames_by_ticket_count"]
        ctx["total_unique_hostnames"] = int(df["hostname"].nunique())

    if "hostname" in df.columns and "priority_df" in df.columns:
        high_sev_df = df[df["priority_df"].isin([1, 2])].copy()
        hs_vc = high_sev_df["hostname"].fillna("(blank)").value_counts().head(20)
        ctx["top_hostnames_by_high_severity_tickets"] = [
            {"rank": i+1, "hostname": str(k), "high_sev_tickets": int(v)}
            for i, (k, v) in enumerate(hs_vc.items())
        ]

    if "business_application" in df.columns:
        vc = df["business_application"].fillna("(blank)").value_counts().head(20)
        ctx["tickets_by_application_top20"] = [{"application": str(k), "tickets": int(v)} for k, v in vc.items()]
        ctx["total_unique_applications"] = int(df["business_application"].nunique())

    if "hostname" in df.columns and "mttr_excl_hold" in df.columns:
        grp = (
            df.dropna(subset=["hostname", "mttr_excl_hold"])
            .groupby("hostname")["mttr_excl_hold"]
            .agg(["mean", "count"])
        )
        grp = grp[grp["count"] >= 3].sort_values("mean", ascending=False).head(10)
        ctx["top_hostnames_by_highest_avg_mttr"] = [
            {"rank": i+1, "hostname": str(k), "avg_mttr_hours": round(float(v["mean"]), 2), "ticket_count": int(v["count"])}
            for i, (k, v) in enumerate(grp.iterrows())
        ]
        ctx["hostnames_highest_avg_mttr"] = ctx["top_hostnames_by_highest_avg_mttr"]

    if "business_application" in df.columns and "mttr_excl_hold" in df.columns:
        grp = (
            df.dropna(subset=["business_application", "mttr_excl_hold"])
            .groupby("business_application")["mttr_excl_hold"]
            .agg(["mean", "count"])
        )
        grp = grp[grp["count"] >= 3].sort_values("mean", ascending=False).head(10)
        ctx["applications_highest_avg_mttr"] = [
            {"application": str(k), "avg_mttr_hours": round(float(v["mean"]), 2), "ticket_count": int(v["count"])}
            for k, v in grp.iterrows()
        ]

    if "open_dttm" in df.columns and pd.api.types.is_datetime64_any_dtype(df["open_dttm"]):
        df_v = df.dropna(subset=["open_dttm"]).copy()
        if len(df_v) > 0:
            df_v["hour"] = df_v["open_dttm"].dt.hour
            df_v["day_name"] = df_v["open_dttm"].dt.day_name()
            hourly = df_v["hour"].value_counts().sort_index()
            daily = df_v["day_name"].value_counts()
            ctx["tickets_by_hour_of_day"] = {str(k): int(v) for k, v in hourly.items()}
            ctx["tickets_by_day_of_week"] = {str(k): int(v) for k, v in daily.items()}
            ctx["peak_hour"] = int(hourly.idxmax())
            ctx["peak_day_of_week"] = str(daily.idxmax())

    if "open_dttm" in df.columns and pd.api.types.is_datetime64_any_dtype(df["open_dttm"]):
        df_copy = add_month_col(df.dropna(subset=["open_dttm"]).copy())
        monthly = df_copy["month"].value_counts().sort_index()
        ctx["monthly_ticket_trend"] = [{"month": str(m), "tickets": int(c)} for m, c in monthly.items()]
        if len(monthly) > 0:
            ctx["busiest_month"] = str(monthly.idxmax())
            ctx["quietest_month"] = str(monthly.idxmin())

    if "priority_df" in df.columns:
        sev = df["priority_df"].value_counts().sort_index()
        ctx["tickets_by_severity"] = {str(int(k)) if not pd.isna(k) else "unknown": int(v) for k, v in sev.items()}

    if "mttr_excl_hold" in df.columns:
        valid = df["mttr_excl_hold"].dropna()
        if len(valid) > 0:
            ctx["overall_mttr_stats"] = {
                "mean": round(float(valid.mean()), 2),
                "median": round(float(valid.median()), 2),
                "min": round(float(valid.min()), 2),
                "max": round(float(valid.max()), 2),
            }

    if "hostname" in df.columns:
        unique_hosts = df["hostname"].nunique()
        if unique_hosts > 0:
            ctx["avg_incidents_per_device"] = round(len(df) / unique_hosts, 2)

    return ctx
