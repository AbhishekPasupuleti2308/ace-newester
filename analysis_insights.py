"""
analysis_insights.py — Dedicated insight analysis engine.

Reads the new combined-insights Excel (summary sheet + detail sheet per insight),
categorises into Actionable / Growth / ThreatCon, and produces rich per-insight
analytics (counts, splits, chart data, device tables) for the frontend.
"""

import pandas as pd
import numpy as np
import re
import math
import traceback


# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────

def _safe(val):
    """Return None for NaN/Inf, else the value."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def _vc(series, n=10):
    """Value counts → list of {label, count}."""
    vc = series.fillna("(blank)").astype(str).value_counts().head(n)
    return [{"label": str(k), "count": int(v)} for k, v in vc.items()]


def _clean_hostname(h):
    """Normalise hostname strings."""
    if pd.isna(h):
        return None
    s = str(h).strip().lower()
    # Some detail sheets append _os_type to the device name
    for suffix in ["_microsoft_os_windows", "_linux", "_aix", "_vmware"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    # Remove FQDN suffixes
    if "." in s:
        s = s.split(".")[0]
    return s if s else None


def _categorise_insight(prefix, num):
    if prefix == "A" and num < 1000:
        return "actionable"
    if prefix == "A" and num >= 3000:
        return "threatcon"
    if prefix == "G":
        return "growth"
    return "other"


# ───────────────────────────────────────────────────────────────
# Broader category grouping
# ───────────────────────────────────────────────────────────────

# Explicit mapping: insight_num → broader_category
_EXPLICIT_CATEGORY_MAP = {
    # Cybersecurity and Resiliency
    43:   "Cybersecurity and Resiliency",   # SSL Certificates with weak Public Key Length or using weak protocols
    50:   "Cybersecurity and Resiliency",   # Windows servers allow weak encryption
    55:   "Cybersecurity and Resiliency",   # Windows servers requiring disable of SMBv1 protocol
    64:   "Cybersecurity and Resiliency",   # Device with most expiring SSL certificates
    221:  "Cybersecurity and Resiliency",   # Cisco devices missing SAT UDC data
    3038: "Cybersecurity and Resiliency",   # ThreatCon - F5 Nation State Breach
    3039: "Cybersecurity and Resiliency",   # ThreatCon - Cisco ASA FTD IOS ZeroDay
    3074: "Cybersecurity and Resiliency",   # ThreatCon - Fortinet
    3075: "Cybersecurity and Resiliency",   # ThreatCon - MS Patches - Bravo
    3078: "Cybersecurity and Resiliency",   # ThreatCon - MS Patch Tuesday
    1105: "Cybersecurity and Resiliency",   # Long Expired Business Application SSL Certificates
    1111: "Cybersecurity and Resiliency",   # Service Requests with most security related issues
    1057: "Cybersecurity and Resiliency",   # Servers without patching
    1093: "Cybersecurity and Resiliency",   # Servers hosting critical apps without latest patch
}

# Keyword-based fallback mapping (applied to title)
_KEYWORD_CATEGORY_MAP = [
    (["ssl", "certificate", "cert", "tls", "encryption", "weak key", "smb", "smbv1",
      "threat", "threatcon", "cve", "vulnerab", "patch", "security",
      "backup", "critical server", "backup scope",
      "health check", "log retention", "non-compliance",
      "weak protocol", "weak public key"],                                  "Cybersecurity and Resiliency"),
    (["monitor", "offline", "not available", "observ", "apm", "syslog",
      "zabbix", "telemetry", "without monitoring", "no evidence of apm",
      "infrastructure monitoring"],                                          "Observability"),
    (["incident", "ticket", "frequent", "most incident", "most auto-resolved",
      "disk related", "process down", "service down"],                       "Incident Management"),
    (["automat", "playbook", "automata", "auto-resolved", "resolving incident"],  "Automation"),
    (["cpu", "memory", "swap", "performance", "high cpu", "resource optim",
      "iops", "utilization", "month-end"],                                   "Performance & Capacity"),
    (["disk", "storage", "filesystem", "capacity"],                          "Storage"),
    (["eol", "eos", "end of life", "end of service", "lifecycle", "legacy",
      "bios", "uefi"],                                                       "Lifecycle & Compliance"),
    (["change", "change group"],                                             "Change Management"),
    (["network", "network device", "endpoint", "connection lost", "brocade",
      "cisco"],                                                              "Network"),
    (["application", "business application", "database", "ms sql", "index",
      "app"],                                                                "Application Health"),
    (["cloud", "cloud ready", "migration"],                                  "Cloud & Modernisation"),
    (["service request", "sr "],                                             "Service Requests"),
    (["non-managed", "non managed", "kyndryl non"],                          "Non-Managed Ops"),
    (["aiops", "readiness", "best practice", "sat",
      "commit applied", "roundrobin"],                                       "Configuration & Compliance"),
    (["energy", "emission", "green"],                                        "Sustainability"),
    (["catalogue", "catalog", "missing"],                                    "Asset Management"),
]


def _assign_broader_category(insight_id_num, title):
    """Assign a broader category to an insight based on explicit map then keyword matching."""
    # Check explicit map first
    if insight_id_num in _EXPLICIT_CATEGORY_MAP:
        return _EXPLICIT_CATEGORY_MAP[insight_id_num]

    # Keyword fallback
    title_lower = (title or "").lower()
    for keywords, category in _KEYWORD_CATEGORY_MAP:
        if any(kw in title_lower for kw in keywords):
            return category

    return "Other"


# ───────────────────────────────────────────────────────────────
# Detail analysers — one per column pattern family
# ───────────────────────────────────────────────────────────────

def _analyse_incident_detail(df):
    """For insight details that are incident/ticket rows."""
    result = {"type": "incidents", "total_rows": len(df)}
    cols = set(df.columns)

    if "hostname" in cols:
        result["by_hostname"] = _vc(df["hostname"], 20)
        result["unique_hosts"] = int(df["hostname"].nunique())

    if "severity" in cols or "dv_severity" in cols:
        sev_col = "severity" if "severity" in cols else "dv_severity"
        result["by_severity"] = _vc(df[sev_col])

    if "category" in cols:
        result["by_category"] = _vc(df["category"], 15)

    if "business_application" in cols:
        result["by_application"] = _vc(df["business_application"], 15)

    if "suggested_automata" in cols:
        result["by_automata"] = _vc(df["suggested_automata"], 15)

    if "component" in cols:
        result["by_component"] = _vc(df["component"], 15)

    if "status" in cols:
        result["by_status"] = _vc(df["status"])

    if "sso_ticket" in cols:
        result["by_sso"] = _vc(df["sso_ticket"])

    if "ostype" in cols:
        result["by_os"] = _vc(df["ostype"])

    if "server_function" in cols:
        result["by_server_function"] = _vc(df["server_function"], 10)

    if "open_dttm" in cols:
        try:
            dt = pd.to_datetime(df["open_dttm"], errors="coerce", utc=True)
            valid = dt.dropna()
            if len(valid):
                monthly = valid.dt.to_period("M").value_counts().sort_index()
                result["monthly_trend"] = [
                    {"month": str(m), "count": int(c)} for m, c in monthly.items()
                ]
        except Exception:
            pass

    return result


def _analyse_health_check_detail(df):
    """For SAT / health-check detail rows (check_name, check_result, device_name …)."""
    result = {"type": "health_checks", "total_rows": len(df)}
    cols = set(df.columns)

    host_col = "device_name" if "device_name" in cols else ("hostname" if "hostname" in cols else None)
    if host_col:
        cleaned = df[host_col].apply(_clean_hostname).dropna()
        result["by_hostname"] = _vc(cleaned, 20)
        result["unique_hosts"] = int(cleaned.nunique())

    if "check_name" in cols:
        result["by_check_name"] = _vc(df["check_name"])

    if "check_result" in cols:
        result["by_check_result"] = _vc(df["check_result"])

    if "check_severity" in cols:
        result["by_severity"] = _vc(df["check_severity"])

    if "check_category" in cols:
        result["by_category"] = _vc(df["check_category"])

    if "device_machine_type" in cols:
        result["by_machine_type"] = _vc(df["device_machine_type"])

    if "device_vendor" in cols:
        result["by_vendor"] = _vc(df["device_vendor"])

    if "device_model" in cols:
        result["by_model"] = _vc(df["device_model"], 15)

    if "device_firmware" in cols:
        result["by_firmware"] = _vc(df["device_firmware"], 10)

    if "device_domain" in cols:
        result["by_domain"] = _vc(df["device_domain"])

    if "check_id" in cols:
        result["by_check_id"] = _vc(df["check_id"])

    return result


def _analyse_certificate_detail(df):
    """For SSL certificate detail rows."""
    result = {"type": "certificates", "total_rows": len(df)}
    cols = set(df.columns)

    if "hostname" in cols:
        result["by_hostname"] = _vc(df["hostname"], 20)
        result["unique_hosts"] = int(df["hostname"].nunique())

    if "platform" in cols:
        result["by_platform"] = _vc(df["platform"])

    if "pub_key_size" in cols:
        result["by_key_size"] = _vc(df["pub_key_size"].astype(str))

    if "pub_key_alg" in cols:
        result["by_algorithm"] = _vc(df["pub_key_alg"])

    if "TLS_used" in cols:
        result["by_tls_version"] = _vc(df["TLS_used"])

    if "cert_expiration_date" in cols:
        try:
            dt = pd.to_datetime(df["cert_expiration_date"], errors="coerce", utc=True)
            now = pd.Timestamp.utcnow()
            expired = int((dt < now).sum())
            expiring_90d = int(((dt >= now) & (dt < now + pd.Timedelta(days=90))).sum())
            result["expired_count"] = expired
            result["expiring_90d"] = expiring_90d
        except Exception:
            pass

    if "application_running" in cols:
        result["by_application"] = _vc(df["application_running"])

    if "cert_issuer" in cols:
        result["by_issuer"] = _vc(df["cert_issuer"])

    if "age_range" in cols:
        result["by_age_range"] = _vc(df["age_range"])

    if "business_application" in cols:
        result["by_business_app"] = _vc(df["business_application"])

    return result


def _analyse_vulnerability_detail(df):
    """For ThreatCon / CVE vulnerability detail rows."""
    result = {"type": "vulnerabilities", "total_rows": len(df)}
    cols = set(df.columns)

    host_col = "host_name" if "host_name" in cols else ("hostname" if "hostname" in cols else None)
    if host_col:
        cleaned = df[host_col].apply(_clean_hostname).dropna()
        result["by_hostname"] = _vc(cleaned, 20)
        result["unique_hosts"] = int(cleaned.nunique())

    if "operating_system" in cols:
        result["by_os"] = _vc(df["operating_system"])

    if "vendor" in cols:
        result["by_vendor"] = _vc(df["vendor"])

    if "base_score_severity" in cols:
        result["by_severity"] = _vc(df["base_score_severity"])

    if "base_score" in cols:
        try:
            scores = pd.to_numeric(df["base_score"], errors="coerce").dropna()
            if len(scores):
                result["avg_base_score"] = round(float(scores.mean()), 2)
                result["max_base_score"] = round(float(scores.max()), 2)
        except Exception:
            pass

    if "cve_id" in cols:
        result["by_cve"] = _vc(df["cve_id"])

    if "vuln_status" in cols:
        result["by_vuln_status"] = _vc(df["vuln_status"])

    if "lifecycle_status" in cols:
        result["by_lifecycle"] = _vc(df["lifecycle_status"])

    if "remediation_status" in cols:
        result["by_remediation"] = _vc(df["remediation_status"])

    if "device_type" in cols:
        result["by_device_type"] = _vc(df["device_type"])

    # CVE age analysis: how long since CVE was published
    cve_date_col = None
    for c in ["cve_published_date", "published_date", "cve_date", "publish_date",
              "first_found_date", "first_found", "disclosure_date"]:
        if c in cols:
            cve_date_col = c
            break
    if cve_date_col:
        try:
            today = pd.Timestamp.now(tz=None).normalize()
            dt = pd.to_datetime(df[cve_date_col], errors="coerce")
            if hasattr(dt.dt, 'tz_convert') and dt.dt.tz is not None:
                dt = dt.dt.tz_convert(None)
            valid = dt.dropna()
            if len(valid) > 0:
                age_days = (today - valid).dt.days
                result["cve_age_stats"] = {
                    "oldest_days": int(age_days.max()),
                    "newest_days": int(age_days.min()),
                    "avg_days": int(age_days.mean()),
                    "oldest_date": str(valid.min().date()),
                    "newest_date": str(valid.max().date()),
                }
                # Buckets: <30d, 30-90d, 90-365d, 1-3yr, 3yr+
                buckets = {"<30 days": 0, "30-90 days": 0, "90-365 days": 0, "1-3 years": 0, "3+ years": 0}
                for d in age_days:
                    if d < 30: buckets["<30 days"] += 1
                    elif d < 90: buckets["30-90 days"] += 1
                    elif d < 365: buckets["90-365 days"] += 1
                    elif d < 1095: buckets["1-3 years"] += 1
                    else: buckets["3+ years"] += 1
                result["cve_age_buckets"] = [{"label": k, "count": v} for k, v in buckets.items() if v > 0]
        except Exception:
            pass

    return result


def _analyse_change_detail(df):
    """For change management detail rows."""
    result = {"type": "changes", "total_rows": len(df)}
    cols = set(df.columns)

    if "change_type_cd" in cols:
        result["by_change_type"] = _vc(df["change_type_cd"])
    if "category_cd" in cols:
        result["by_category"] = _vc(df["category_cd"])
    if "cmpltn_code_cd" in cols:
        result["by_completion"] = _vc(df["cmpltn_code_cd"])
    if "risk_cd" in cols:
        result["by_risk"] = _vc(df["risk_cd"])
    if "assignee_group_cd" in cols:
        result["by_group"] = _vc(df["assignee_group_cd"])
    return result


def _analyse_endpoint_detail(df):
    """For endpoint / CACF detail rows."""
    result = {"type": "endpoints", "total_rows": len(df)}
    cols = set(df.columns)

    if "hostname" in cols:
        result["by_hostname"] = _vc(df["hostname"], 20)
        result["unique_hosts"] = int(df["hostname"].nunique())
    if "job_status" in cols:
        result["by_status"] = _vc(df["job_status"])
    if "job_type" in cols:
        result["by_type"] = _vc(df["job_type"])
    if "rc_group" in cols:
        result["by_rc_group"] = _vc(df["rc_group"])
    return result


def _analyse_cloud_ready_detail(df):
    """For cloud-ready server detail rows."""
    result = {"type": "cloud_ready", "total_rows": len(df)}
    cols = set(df.columns)

    if "hostname" in cols:
        result["by_hostname"] = _vc(df["hostname"], 20)
        result["unique_hosts"] = int(df["hostname"].nunique())
    if "osname" in cols:
        result["by_os"] = _vc(df["osname"])
    if "eol" in cols:
        result["by_eol"] = _vc(df["eol"])
    if "eos" in cols:
        result["by_eos"] = _vc(df["eos"])
    if "machinetype" in cols:
        result["by_machine_type"] = _vc(df["machinetype"])
    if "image_usage" in cols:
        result["by_usage"] = _vc(df["image_usage"])
    if "virtualflag" in cols:
        result["by_virtual"] = _vc(df["virtualflag"])
    if "ci_lifecyclestate" in cols:
        result["by_lifecycle"] = _vc(df["ci_lifecyclestate"])
    return result


def _analyse_resource_detail(df):
    """For resource optimisation / KPI detail rows."""
    result = {"type": "resource_metrics", "total_rows": len(df)}
    cols = set(df.columns)

    if "hostname" in cols:
        result["by_hostname"] = _vc(df["hostname"], 20)
        result["unique_hosts"] = int(df["hostname"].nunique())
    if "kpi_name" in cols:
        result["by_kpi"] = _vc(df["kpi_name"])
    if "ostype" in cols:
        result["by_os"] = _vc(df["ostype"])
    if "server_type" in cols:
        result["by_server_type"] = _vc(df["server_type"])
    if "virtualflag" in cols:
        result["by_virtual"] = _vc(df["virtualflag"])

    # Numeric KPI summaries
    if "kpi_value" in cols or "kpi_value_num" in cols:
        val_col = "kpi_value_num" if "kpi_value_num" in cols else "kpi_value"
        try:
            nums = pd.to_numeric(df[val_col], errors="coerce").dropna()
            if len(nums):
                result["kpi_stats"] = {
                    "mean": round(float(nums.mean()), 2),
                    "median": round(float(nums.median()), 2),
                    "min": round(float(nums.min()), 2),
                    "max": round(float(nums.max()), 2),
                }
        except Exception:
            pass

    if "disk_id" in cols:
        result["by_disk"] = _vc(df["disk_id"], 15)

    return result


def _analyse_patching_detail(df):
    """For patching / patch compliance detail rows."""
    result = {"type": "patching", "total_rows": len(df)}
    cols = set(df.columns)

    host_col = None
    for c in ["hostname", "host_name", "device_name"]:
        if c in cols:
            host_col = c
            break
    if host_col:
        result["by_hostname"] = _vc(df[host_col], 20)
        result["unique_hosts"] = int(df[host_col].nunique())

    for c in ["operating_system_name", "operating_system", "osname"]:
        if c in cols:
            result["by_os"] = _vc(df[c])
            break

    if "life_cycle_status" in cols:
        result["by_lifecycle"] = _vc(df["life_cycle_status"])
    if "category" in cols:
        result["by_category"] = _vc(df["category"])
    if "manufacturer" in cols:
        result["by_manufacturer"] = _vc(df["manufacturer"])
    if "item_class" in cols:
        result["by_class"] = _vc(df["item_class"])
    if "subcategory" in cols:
        result["by_subcategory"] = _vc(df["subcategory"])

    return result


def _analyse_generic_detail(df):
    """Fallback analyser — grab counts for any string column."""
    result = {"type": "generic", "total_rows": len(df)}

    host_col = None
    for c in ["hostname", "host_name", "device_name"]:
        if c in df.columns:
            host_col = c
            break
    if host_col:
        result["by_hostname"] = _vc(df[host_col], 20)
        result["unique_hosts"] = int(df[host_col].nunique())

    # Auto-detect up to 6 categorical columns
    cat_cols_done = 0
    for col in df.columns:
        if col == host_col or cat_cols_done >= 6:
            break
        if df[col].dtype == "object" and df[col].nunique() < 50:
            key = f"by_{col}"
            result[key] = _vc(df[col], 10)
            cat_cols_done += 1

    return result


def _pick_analyser(df):
    """Route a detail DataFrame to the right analyser."""
    cols = set(df.columns)

    # Note-only (no real data)
    if cols == {"note"} or len(df) <= 1 and "note" in cols:
        note_val = str(df.iloc[0, 0]) if len(df) else ""
        return {"type": "no_data", "total_rows": 0, "note": note_val}

    # Incident / ticket data
    if "incident_code_id" in cols or ("abstract" in cols and "severity" in cols):
        return _analyse_incident_detail(df)

    # Health check / SAT data
    if "check_name" in cols or "check_result" in cols:
        return _analyse_health_check_detail(df)

    # Certificate data
    if "cert_cn" in cols or "cert_expiration_date" in cols:
        return _analyse_certificate_detail(df)

    # Vulnerability / ThreatCon data
    if "cve_id" in cols or "base_score_severity" in cols:
        return _analyse_vulnerability_detail(df)

    # Change data
    if "change_id" in cols or "change_type_cd" in cols:
        return _analyse_change_detail(df)

    # Endpoint CACF
    if "job_status" in cols and "job_type" in cols:
        return _analyse_endpoint_detail(df)

    # Cloud ready
    if "eol" in cols and "eos" in cols and "machinetype" in cols:
        return _analyse_cloud_ready_detail(df)

    # Resource / KPI
    if "kpi_name" in cols or "kpi_value" in cols or "kpi_value_num" in cols:
        return _analyse_resource_detail(df)

    # Patching / catalogue
    if "life_cycle_status" in cols or "item_class" in cols:
        return _analyse_patching_detail(df)

    # Fallback
    return _analyse_generic_detail(df)


# ───────────────────────────────────────────────────────────────
# Build aggregated device stats (numbers, not raw table)
# ───────────────────────────────────────────────────────────────

def _build_device_stats(df):
    """Return aggregated device stats: by_os, by_platform, by_vendor, etc. — numbers only, no raw rows."""
    if df is None or len(df) == 0:
        return None

    cols = set(df.columns)
    stats = {"total_devices": len(df)}

    # Find the hostname column
    host_col = None
    for c in ["hostname", "host_name", "device_name"]:
        if c in cols:
            host_col = c
            break
    if host_col:
        cleaned = df[host_col].apply(_clean_hostname).dropna()
        stats["unique_hosts"] = int(cleaned.nunique())

    # OS breakdown
    for c in ["platform", "operating_system", "osname", "ostype",
              "operating_system_name", "operating_system_type"]:
        if c in cols:
            stats["by_os"] = _vc(df[c], 15)
            break

    # Machine type
    if "device_machine_type" in cols:
        stats["by_machine_type"] = _vc(df["device_machine_type"])
    elif "machinetype" in cols:
        stats["by_machine_type"] = _vc(df["machinetype"])

    # Vendor
    for c in ["device_vendor", "vendor", "manufacturer"]:
        if c in cols:
            stats["by_vendor"] = _vc(df[c], 10)
            break

    # Model
    for c in ["device_model", "model"]:
        if c in cols:
            stats["by_model"] = _vc(df[c], 10)
            break

    # Firmware
    if "device_firmware" in cols:
        stats["by_firmware"] = _vc(df["device_firmware"], 10)

    # Domain
    if "device_domain" in cols:
        stats["by_domain"] = _vc(df["device_domain"], 10)
    elif "domain" in cols:
        stats["by_domain"] = _vc(df["domain"], 10)

    # Check results / severity (for health check detail)
    if "check_result" in cols:
        stats["by_check_result"] = _vc(df["check_result"])
    if "check_name" in cols:
        stats["by_check_name"] = _vc(df["check_name"], 10)
    if "check_severity" in cols:
        stats["by_check_severity"] = _vc(df["check_severity"])

    # Certificate-specific
    if "pub_key_size" in cols:
        stats["by_key_size"] = _vc(df["pub_key_size"].astype(str))
    if "pub_key_alg" in cols:
        stats["by_algorithm"] = _vc(df["pub_key_alg"])
    if "TLS_used" in cols:
        stats["by_tls_version"] = _vc(df["TLS_used"])
    if "cert_cn" in cols:
        stats["unique_certs"] = int(df["cert_cn"].nunique())

    # Vulnerability-specific
    if "base_score_severity" in cols:
        stats["by_severity"] = _vc(df["base_score_severity"])
    if "cve_id" in cols:
        stats["unique_cves"] = int(df["cve_id"].nunique())
    if "vuln_status" in cols:
        stats["by_vuln_status"] = _vc(df["vuln_status"])

    # Lifecycle / status
    if "life_cycle_status" in cols:
        stats["by_lifecycle"] = _vc(df["life_cycle_status"])
    if "ci_lifecyclestate" in cols:
        stats["by_lifecycle"] = _vc(df["ci_lifecyclestate"])

    # Category
    if "category" in cols:
        stats["by_category"] = _vc(df["category"], 10)
    if "check_category" in cols:
        stats["by_check_category"] = _vc(df["check_category"], 10)

    # Severity (incidents)
    if "severity" in cols:
        stats["by_severity"] = _vc(df["severity"])
    elif "dv_severity" in cols:
        stats["by_severity"] = _vc(df["dv_severity"])

    # Status
    if "status" in cols:
        stats["by_status"] = _vc(df["status"])
    if "job_status" in cols:
        stats["by_status"] = _vc(df["job_status"])

    # KPI
    if "kpi_name" in cols:
        stats["by_kpi"] = _vc(df["kpi_name"])

    # Virtual flag
    if "virtualflag" in cols:
        stats["by_virtual"] = _vc(df["virtualflag"])

    # Server type
    if "server_type" in cols:
        stats["by_server_type"] = _vc(df["server_type"])

    # Automata
    if "suggested_automata" in cols:
        stats["by_automata"] = _vc(df["suggested_automata"], 15)

    # Business application
    if "business_application" in cols:
        stats["by_application"] = _vc(df["business_application"], 15)

    # SSO ticket
    if "sso_ticket" in cols:
        stats["by_sso"] = _vc(df["sso_ticket"])

    return stats


# ───────────────────────────────────────────────────────────────
# Build device table for frontend (truncated to keep payloads sane)
# ───────────────────────────────────────────────────────────────

_DEVICE_TABLE_MAX_ROWS = 200
_DEVICE_TABLE_MAX_COLS = 12

# Columns we prefer to show (in order) when building a device table
_PREFERRED_COLS = [
    "hostname", "host_name", "device_name",
    "operating_system", "osname", "ostype", "platform",
    "severity", "dv_severity", "base_score_severity", "check_severity",
    "base_score",
    "status", "check_result", "vuln_status", "lifecycle_status", "life_cycle_status",
    "business_application",
    "category", "check_name", "cve_id",
    "vendor", "device_vendor", "manufacturer",
    "device_model", "model",
    "cert_cn", "cert_expiration_date",
    "TLS_used", "pub_key_size",
    "kpi_name", "kpi_value", "kpi_value_num",
    "check_id", "check_category",
    "description",
]


def _build_device_table(df):
    """Return a lightweight {columns, rows} object for the frontend table."""
    if df is None or len(df) == 0:
        return None

    # Pick the best columns to show
    available = [c for c in _PREFERRED_COLS if c in df.columns]
    # Add any remaining columns up to the limit
    for c in df.columns:
        if c not in available and len(available) < _DEVICE_TABLE_MAX_COLS:
            # Skip very long text or internal columns
            if c in ("record_category", "metric_category", "gsma_code", "contract_id",
                      "customer_id", "customer_name", "country_code", "note"):
                continue
            available.append(c)
        if len(available) >= _DEVICE_TABLE_MAX_COLS:
            break

    if not available:
        return None

    subset = df[available].head(_DEVICE_TABLE_MAX_ROWS).copy()

    # Clean column names for display
    display_cols = []
    for c in available:
        pretty = c.replace("_", " ").title()
        display_cols.append({"key": c, "label": pretty})

    rows = []
    for _, row in subset.iterrows():
        r = {}
        for c in available:
            val = row[c]
            if pd.isna(val):
                r[c] = None
            elif isinstance(val, (np.integer, np.floating)):
                r[c] = float(val) if isinstance(val, np.floating) else int(val)
            else:
                r[c] = str(val)[:200]  # Truncate long strings
        rows.append(r)

    return {
        "columns": display_cols,
        "rows": rows,
        "total_rows": len(df),
        "showing": len(rows),
    }


# ───────────────────────────────────────────────────────────────
# Parse recommendation text into sections
# ───────────────────────────────────────────────────────────────

def _parse_recommendation(text):
    """Split a big recommendation blob into Situation / Problem / Recommendation / Data Provided / Actions."""
    if not text or not isinstance(text, str):
        return {}

    sections = {}
    # Common headings found in these recommendation blobs
    headings = [
        "Situation", "Problem", "Recommendation", "Data provided",
        "Recommended actions", "Recommended Actions", "Reference",
        "Action 1", "Action 2", "Action 3",
    ]

    current_key = "overview"
    current_lines = []

    for line in text.split("\n"):
        stripped = line.strip()
        matched = False
        for h in headings:
            if stripped.lower().startswith(h.lower()):
                # Save previous section
                if current_lines:
                    sections[current_key] = "\n".join(current_lines).strip()
                current_key = h.lower().replace(" ", "_")
                remainder = stripped[len(h):].strip().lstrip("-:").strip()
                current_lines = [remainder] if remainder else []
                matched = True
                break
        if not matched:
            current_lines.append(line)

    if current_lines:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


# ───────────────────────────────────────────────────────────────
# Main public function
# ───────────────────────────────────────────────────────────────

def analyse_insights_file(filepath):
    """
    Read the combined-insights Excel and return a structured dict with
    three top-level keys: actionable, growth, threatcon.

    Each contains a list of insight objects with:
      - id, title, category, total_hosts, hostnames
      - action, observation, recommendation (text)
      - rec_sections (parsed recommendation)
      - platform_summary (dict of platform → count)
      - detail_analysis (rich analytics from the detail sheet)
      - device_table (columns + rows for the frontend grid)
    """

    xls = pd.ExcelFile(filepath)
    sheets = xls.sheet_names

    result = {
        "actionable": [],
        "growth": [],
        "threatcon": [],
        "summary": {},
    }

    i = 0
    while i < len(sheets):
        summary_sheet = sheets[i]
        m = re.match(r"([AG])-(\d+)", summary_sheet)
        if not m:
            i += 1
            continue

        prefix = m.group(1)
        num = int(m.group(2))
        cat = _categorise_insight(prefix, num)

        # Find the matching detail sheet (next sheet ending with "- Det")
        detail_sheet = None
        if i + 1 < len(sheets) and "- Det" in sheets[i + 1]:
            detail_sheet = sheets[i + 1]
            i += 2
        else:
            i += 1

        try:
            sdf = pd.read_excel(xls, summary_sheet)
            ddf = pd.read_excel(xls, detail_sheet) if detail_sheet else None
        except Exception as e:
            print(f"[analysis_insights] Error reading {summary_sheet}: {e}")
            continue

        # ── Extract summary fields ──
        row = sdf.iloc[0] if len(sdf) else pd.Series()

        insight_id = f"{prefix}-{num}"
        title = str(row.get("insight_title", summary_sheet)).strip()
        total_hosts = int(row.get("total_hosts", 0)) if not pd.isna(row.get("total_hosts", None)) else 0
        action = str(row.get("action", "")) if not pd.isna(row.get("action", None)) else ""
        observation = str(row.get("observation", "")) if not pd.isna(row.get("observation", None)) else ""
        recommendation = str(row.get("recommendation", "")) if not pd.isna(row.get("recommendation", None)) else ""

        # Hostnames list
        hostnames_raw = str(row.get("hostname", "")) if not pd.isna(row.get("hostname", None)) else ""
        hostnames = [h.strip() for h in hostnames_raw.split(",") if h.strip()] if hostnames_raw else []

        # Platform counts from summary columns
        platform_summary = {}
        platform_cols = ["Windows", "WINDOWS", "Linux", "0000", "EMTY", "INTEL", "VMCL", "VMVC"]
        for pc in platform_cols:
            if pc in sdf.columns:
                val = row.get(pc, 0)
                if not pd.isna(val) and int(val) > 0:
                    platform_summary[pc] = int(val)
        # Also check platform_summary text field
        ps_text = str(row.get("platform_summary", "")) if not pd.isna(row.get("platform_summary", None)) else ""

        # Parse recommendation into sections
        rec_sections = _parse_recommendation(recommendation)

        # ── Analyse detail sheet ──
        detail_analysis = None
        device_stats = None
        if ddf is not None and len(ddf) > 0:
            try:
                detail_analysis = _pick_analyser(ddf)
                device_stats = _build_device_stats(ddf)
            except Exception as e:
                detail_analysis = {"type": "error", "message": str(e)}
                traceback.print_exc()

        # Assign broader category
        broader_category = _assign_broader_category(num, title)

        insight_obj = {
            "id": insight_id,
            "num": num,
            "prefix": prefix,
            "category": cat,
            "broader_category": broader_category,
            "title": title,
            "total_hosts": total_hosts,
            "hostnames": hostnames[:50],  # Cap for payload size
            "hostname_count": len(hostnames),
            "action": action,
            "observation": observation[:3000],  # Truncate very long text
            "recommendation": recommendation[:5000],
            "rec_sections": rec_sections,
            "platform_summary": platform_summary,
            "platform_summary_text": ps_text,
            "detail_analysis": detail_analysis,
            "device_stats": device_stats,
        }

        if cat in result:
            result[cat].append(insight_obj)

    # Sort each category by ID
    for cat in ("actionable", "growth", "threatcon"):
        result[cat].sort(key=lambda x: x["num"])

    result["summary"] = {
        "actionable_count": len(result["actionable"]),
        "growth_count": len(result["growth"]),
        "threatcon_count": len(result["threatcon"]),
        "total_insights": len(result["actionable"]) + len(result["growth"]) + len(result["threatcon"]),
    }

    # Build broader category grouping
    all_insights = result["actionable"] + result["growth"] + result["threatcon"]
    category_groups = {}
    for ins in all_insights:
        bc = ins.get("broader_category", "Other")
        if bc not in category_groups:
            category_groups[bc] = {
                "category": bc,
                "insights": [],
                "total_hosts": 0,
                "actionable_count": 0,
                "growth_count": 0,
                "threatcon_count": 0,
            }
        category_groups[bc]["insights"].append(ins["id"])
        category_groups[bc]["total_hosts"] += ins.get("total_hosts", 0)
        if ins["category"] == "actionable":
            category_groups[bc]["actionable_count"] += 1
        elif ins["category"] == "growth":
            category_groups[bc]["growth_count"] += 1
        elif ins["category"] == "threatcon":
            category_groups[bc]["threatcon_count"] += 1

    # Sort categories by total insight count descending
    sorted_categories = sorted(
        category_groups.values(),
        key=lambda x: len(x["insights"]),
        reverse=True,
    )
    result["broader_categories"] = sorted_categories

    return result
