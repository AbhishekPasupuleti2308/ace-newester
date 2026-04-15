"""
analysis_vesa.py — VESA-style analytics: Semantic Topic Clustering, Similar Ticket Detection,
MTTR Prediction. These are new capabilities inspired by Kyndryl VESA services.
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
import hashlib
from preprocessing import add_month_col


# ═══════════════════════════════════════════════════════════
# 1. SEMANTIC TOPIC CLUSTERING & EMERGING TREND ANALYSIS
# ═══════════════════════════════════════════════════════════

# Stop words for text processing
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "don", "now", "and",
    "but", "or", "if", "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "what", "which", "who", "whom",
    "up", "down", "about", "get", "got", "set", "also", "any", "per",
    "new", "one", "two", "use", "see", "due", "via", "etc", "using",
    "used", "been", "null", "none", "nan", "n/a", "na",
})

# Domain-specific stop words common in IT tickets
_IT_STOP_WORDS = frozenset({
    "please", "ticket", "incident", "resolved", "closed", "opened",
    "updated", "assigned", "alert", "event", "monitor", "system",
    "server", "service", "application", "error", "issue", "problem",
    "request", "change", "noted", "check", "checking", "checked",
})


def _tokenize(text):
    """Clean tokenize: lowercase, remove punctuation, filter stops."""
    text = re.sub(r'[^a-zA-Z0-9\s\-\_\.]', ' ', str(text).lower())
    tokens = text.split()
    return [t for t in tokens if len(t) > 2 and t not in _STOP_WORDS and t not in _IT_STOP_WORDS]


def _extract_bigrams(tokens):
    """Extract meaningful bigrams from tokens."""
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]


def _assign_topic_label(keywords):
    """Assign a human-readable topic label based on dominant keywords."""
    kw_set = set(keywords)
    patterns = [
        ({"backup", "failed", "failure", "job", "restore", "commvault", "veeam"}, "Backup & Restore Failures"),
        ({"disk", "space", "storage", "capacity", "filesystem", "full"}, "Disk Space / Storage Issues"),
        ({"cpu", "memory", "utilization", "high", "threshold", "performance"}, "Resource Utilization Alerts"),
        ({"network", "connectivity", "connection", "timeout", "latency", "dns", "dhcp"}, "Network Connectivity Issues"),
        ({"certificate", "ssl", "cert", "expired", "expiring", "tls"}, "Certificate Expiration"),
        ({"password", "reset", "account", "locked", "access", "login"}, "Access & Authentication Issues"),
        ({"reboot", "restart", "hung", "unresponsive", "down", "crashed"}, "Service/Server Restarts"),
        ({"patch", "update", "upgrade", "vulnerability", "security"}, "Patching & Security Updates"),
        ({"database", "sql", "oracle", "db2", "query", "deadlock"}, "Database Issues"),
        ({"batch", "job", "scheduled", "cron", "automation", "task"}, "Batch Job Failures"),
        ({"hardware", "dimm", "power", "fan", "motherboard", "raid"}, "Hardware Failures"),
        ({"application", "app", "deploy", "release", "code", "exception"}, "Application Errors"),
        ({"monitoring", "alert", "threshold", "nagios", "zabbix", "prometheus"}, "Monitoring Alerts"),
        ({"change", "maintenance", "window", "outage", "planned"}, "Change-Related Issues"),
    ]
    for pattern_words, label in patterns:
        if len(kw_set & pattern_words) >= 2:
            return label
    if keywords:
        return f"Topic: {' / '.join(keywords[:3]).title()}"
    return "Uncategorized"


def analyze_topic_clusters(df, n_clusters=12, min_cluster_size=5):
    """
    Perform keyword-based topic clustering on incident descriptions.
    Uses TF-IDF-like weighting and co-occurrence to group tickets into topics.
    Returns topic clusters with their trends over time.
    """
    if "description" not in df.columns:
        return {"error": "No description column available"}

    desc_df = df.dropna(subset=["description"]).copy()
    if len(desc_df) < 20:
        return {"error": "Not enough tickets with descriptions for clustering", "count": len(desc_df)}

    # Tokenize all descriptions
    desc_df["_tokens"] = desc_df["description"].apply(_tokenize)
    desc_df["_bigrams"] = desc_df["_tokens"].apply(_extract_bigrams)

    # Build document frequency for IDF-like weighting
    n_docs = len(desc_df)
    doc_freq = Counter()
    for tokens in desc_df["_tokens"]:
        doc_freq.update(set(tokens))

    # Filter to meaningful terms (appear in >0.5% but <30% of docs)
    min_df = max(3, int(n_docs * 0.005))
    max_df = int(n_docs * 0.30)
    vocab = {term for term, freq in doc_freq.items() if min_df <= freq <= max_df}

    if len(vocab) < 10:
        # Relax constraints
        min_df = 2
        max_df = int(n_docs * 0.50)
        vocab = {term for term, freq in doc_freq.items() if min_df <= freq <= max_df}

    # Build bigram frequencies
    bigram_freq = Counter()
    for bigrams in desc_df["_bigrams"]:
        bigram_freq.update([b for b in bigrams if all(w in vocab for w in b.split())])

    # Create topic seeds from top bigrams and frequent terms
    top_bigrams = bigram_freq.most_common(n_clusters * 3)
    top_terms = [(term, doc_freq[term]) for term in vocab]
    top_terms.sort(key=lambda x: x[1], reverse=True)

    # Build clusters using keyword co-occurrence
    clusters = {}
    assigned = set()

    # Phase 1: Seed from top bigrams
    for bigram, count in top_bigrams:
        if len(clusters) >= n_clusters:
            break
        words = bigram.split()
        # Check these words aren't already claimed
        if any(w in assigned for w in words):
            continue
        # Find related terms (co-occur with these words)
        related = Counter()
        for idx, row in desc_df.iterrows():
            tokens = set(row["_tokens"]) & vocab
            if any(w in tokens for w in words):
                related.update(tokens - set(words))
        top_related = [w for w, _ in related.most_common(5) if w not in assigned]
        cluster_kw = words + top_related[:3]
        label = _assign_topic_label(cluster_kw)
        cid = f"c{len(clusters)}"
        clusters[cid] = {
            "keywords": cluster_kw,
            "label": label,
            "seed_bigram": bigram,
            "indices": [],
        }
        assigned.update(words)

    # Phase 2: Assign tickets to clusters
    for idx, row in desc_df.iterrows():
        tokens = set(row["_tokens"])
        best_cluster = None
        best_score = 0
        for cid, cluster in clusters.items():
            score = len(tokens & set(cluster["keywords"]))
            if score > best_score:
                best_score = score
                best_cluster = cid
        if best_cluster and best_score >= 2:
            clusters[best_cluster]["indices"].append(idx)

    # Filter out tiny clusters
    clusters = {cid: c for cid, c in clusters.items() if len(c["indices"]) >= min_cluster_size}

    if not clusters:
        return {"error": "Could not form meaningful topic clusters from the data"}

    # Build result with trend data
    has_time = "open_dttm" in df.columns and pd.api.types.is_datetime64_any_dtype(df["open_dttm"])
    topic_results = []

    for cid, cluster in sorted(clusters.items(), key=lambda x: len(x[1]["indices"]), reverse=True):
        cluster_df = desc_df.loc[cluster["indices"]]
        topic_info = {
            "topic_id": cid,
            "label": cluster["label"],
            "keywords": cluster["keywords"][:6],
            "ticket_count": len(cluster["indices"]),
            "pct_of_total": round(len(cluster["indices"]) / len(desc_df) * 100, 1),
        }

        # Severity distribution within cluster
        if "priority_df" in cluster_df.columns:
            sev_dist = cluster_df["priority_df"].value_counts().sort_index()
            topic_info["severity_distribution"] = {
                str(int(k)): int(v) for k, v in sev_dist.items() if not pd.isna(k)
            }
            sev12 = cluster_df[cluster_df["priority_df"].isin([1, 2])]
            topic_info["high_severity_pct"] = round(len(sev12) / len(cluster_df) * 100, 1) if len(cluster_df) > 0 else 0

        # MTTR within cluster
        if "mttr_excl_hold" in cluster_df.columns:
            mttr = cluster_df["mttr_excl_hold"].dropna()
            if len(mttr) > 0:
                topic_info["avg_mttr"] = round(float(mttr.mean()), 2)
                topic_info["median_mttr"] = round(float(mttr.median()), 2)

        # Monthly trend for this topic
        if has_time:
            time_df = cluster_df.dropna(subset=["open_dttm"]).copy()
            if len(time_df) > 0:
                time_df = add_month_col(time_df)
                monthly = time_df["month"].value_counts().sort_index()
                topic_info["monthly_trend"] = [
                    {"month": str(m), "count": int(c)} for m, c in monthly.items()
                ]
                # Trend direction (compare last 2 months average vs first 2)
                if len(monthly) >= 4:
                    first_avg = float(monthly.iloc[:2].mean())
                    last_avg = float(monthly.iloc[-2:].mean())
                    if first_avg > 0:
                        change = ((last_avg - first_avg) / first_avg) * 100
                        topic_info["trend_direction"] = "increasing" if change > 15 else ("decreasing" if change < -15 else "stable")
                        topic_info["trend_change_pct"] = round(change, 1)

        # Sample descriptions
        sample = cluster_df["description"].dropna().head(5).tolist()
        topic_info["sample_descriptions"] = [str(s)[:150] for s in sample]

        topic_results.append(topic_info)

    # Unassigned tickets
    assigned_indices = set()
    for c in clusters.values():
        assigned_indices.update(c["indices"])
    unassigned_count = len(desc_df) - len(assigned_indices)

    return {
        "total_tickets_analyzed": len(desc_df),
        "topics_found": len(topic_results),
        "assigned_tickets": len(assigned_indices),
        "unassigned_tickets": unassigned_count,
        "unassigned_pct": round(unassigned_count / len(desc_df) * 100, 1),
        "topics": topic_results,
    }


# ═══════════════════════════════════════════════════════════
# 2. SIMILAR TICKET CLUSTERING & PROBLEM DETECTION
# ═══════════════════════════════════════════════════════════

def _text_signature(text, n=3):
    """Create a shingle-based signature for near-duplicate detection."""
    text = re.sub(r'[^a-z0-9\s]', '', str(text).lower().strip())
    words = text.split()
    if len(words) < n:
        return frozenset(words)
    shingles = set()
    for i in range(len(words) - n + 1):
        shingles.add(" ".join(words[i:i+n]))
    return frozenset(shingles)


def _jaccard_similarity(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def analyze_similar_tickets(df, similarity_threshold=0.45, min_group_size=3, max_groups=30):
    """
    Find clusters of similar tickets using text shingling and Jaccard similarity.
    Identifies potential problem tickets (recurring issues that should be grouped).
    """
    if "description" not in df.columns:
        return {"error": "No description column available"}

    work_df = df.dropna(subset=["description"]).copy()
    if len(work_df) < 10:
        return {"error": "Not enough tickets for similarity analysis"}

    # Limit to most recent tickets for performance (similarity is O(n²))
    if len(work_df) > 5000:
        work_df = work_df.sort_values("open_dttm", ascending=False).head(5000) if "open_dttm" in work_df.columns else work_df.tail(5000)

    # Create text signatures
    work_df["_sig"] = work_df["description"].apply(lambda x: _text_signature(x, n=3))
    # Also create a truncated key for exact/near-exact grouping
    work_df["_desc_key"] = work_df["description"].apply(
        lambda x: re.sub(r'[^a-z0-9\s]', '', str(x).lower().strip())[:100]
    )

    # Phase 1: Group exact/near-exact matches (fast)
    exact_groups = defaultdict(list)
    for idx, row in work_df.iterrows():
        exact_groups[row["_desc_key"]].append(idx)

    # Phase 2: Merge similar groups using Jaccard on signatures
    groups = []
    processed_keys = set()

    # Sort by group size descending so we process largest groups first
    sorted_keys = sorted(exact_groups.keys(), key=lambda k: len(exact_groups[k]), reverse=True)

    for key in sorted_keys:
        if key in processed_keys:
            continue
        indices = exact_groups[key]
        if len(indices) < 2:
            continue

        # Get representative signature
        rep_sig = work_df.loc[indices[0], "_sig"]

        # Try to merge with other groups
        merged_indices = list(indices)
        for other_key in sorted_keys:
            if other_key == key or other_key in processed_keys:
                continue
            other_sig = work_df.loc[exact_groups[other_key][0], "_sig"]
            if _jaccard_similarity(rep_sig, other_sig) >= similarity_threshold:
                merged_indices.extend(exact_groups[other_key])
                processed_keys.add(other_key)

        processed_keys.add(key)
        if len(merged_indices) >= min_group_size:
            groups.append(merged_indices)

    if not groups:
        return {
            "total_tickets_analyzed": len(work_df),
            "similar_groups_found": 0,
            "problem_candidates": 0,
            "groups": [],
            "message": "No significant similar ticket clusters found at current threshold"
        }

    # Sort by size
    groups.sort(key=len, reverse=True)
    groups = groups[:max_groups]

    # Build results
    group_results = []
    total_grouped = 0

    for i, group_indices in enumerate(groups):
        group_df = work_df.loc[group_indices]
        total_grouped += len(group_df)

        group_info = {
            "group_id": i + 1,
            "ticket_count": len(group_df),
            "pct_of_total": round(len(group_df) / len(work_df) * 100, 1),
        }

        # Representative description
        group_info["representative_description"] = str(group_df["description"].iloc[0])[:200]

        # Severity breakdown
        if "priority_df" in group_df.columns:
            sev = group_df["priority_df"].value_counts().sort_index()
            group_info["severity_distribution"] = {str(int(k)): int(v) for k, v in sev.items() if not pd.isna(k)}

        # MTTR stats
        if "mttr_excl_hold" in group_df.columns:
            mttr = group_df["mttr_excl_hold"].dropna()
            if len(mttr) > 0:
                group_info["avg_mttr"] = round(float(mttr.mean()), 2)
                group_info["total_hours_spent"] = round(float(mttr.sum()), 1)

        # Time span
        if "open_dttm" in group_df.columns:
            dates = group_df["open_dttm"].dropna()
            if len(dates) > 0:
                group_info["first_occurrence"] = dates.min().isoformat()
                group_info["last_occurrence"] = dates.max().isoformat()
                span = (dates.max() - dates.min()).days
                group_info["span_days"] = int(span)
                group_info["is_recurring"] = span > 7

        # Top hostnames
        if "hostname" in group_df.columns:
            top_hosts = group_df["hostname"].value_counts().head(5)
            group_info["top_hostnames"] = [
                {"hostname": str(h), "count": int(c)} for h, c in top_hosts.items()
            ]

        # Top applications
        if "business_application" in group_df.columns:
            top_apps = group_df["business_application"].value_counts().head(3)
            group_info["top_applications"] = [
                {"app": str(a), "count": int(c)} for a, c in top_apps.items()
            ]

        # Problem ticket candidate scoring
        score = 0
        if group_info.get("ticket_count", 0) >= 10:
            score += 3
        elif group_info.get("ticket_count", 0) >= 5:
            score += 2
        else:
            score += 1
        if group_info.get("is_recurring", False):
            score += 2
        if group_info.get("avg_mttr", 0) > 5:
            score += 2
        sev_dist = group_info.get("severity_distribution", {})
        if int(sev_dist.get("1", 0)) + int(sev_dist.get("2", 0)) > 0:
            score += 2

        group_info["problem_score"] = score
        group_info["is_problem_candidate"] = score >= 5

        group_results.append(group_info)

    problem_candidates = sum(1 for g in group_results if g.get("is_problem_candidate"))

    return {
        "total_tickets_analyzed": len(work_df),
        "similar_groups_found": len(group_results),
        "total_tickets_in_groups": total_grouped,
        "grouped_pct": round(total_grouped / len(work_df) * 100, 1),
        "problem_candidates": problem_candidates,
        "groups": group_results,
    }


# ═══════════════════════════════════════════════════════════
# 3. MTTR PREDICTION / ANALYSIS
# ═══════════════════════════════════════════════════════════

def analyze_mttr_prediction(df):
    """
    Analyze MTTR patterns and build prediction insights:
    - MTTR by category, severity, time-of-day, day-of-week
    - Identify outliers and high-MTTR segments
    - Predict expected MTTR ranges for different ticket types
    """
    if "mttr_excl_hold" not in df.columns:
        return {"error": "No mttr_excl_hold column available"}

    work_df = df.dropna(subset=["mttr_excl_hold"]).copy()
    if len(work_df) < 20:
        return {"error": "Not enough tickets with MTTR data"}

    mttr = work_df["mttr_excl_hold"]
    result = {
        "total_tickets_with_mttr": len(work_df),
        "overall_stats": {
            "mean": round(float(mttr.mean()), 2),
            "median": round(float(mttr.median()), 2),
            "std": round(float(mttr.std()), 2),
            "p25": round(float(mttr.quantile(0.25)), 2),
            "p75": round(float(mttr.quantile(0.75)), 2),
            "p90": round(float(mttr.quantile(0.90)), 2),
            "p95": round(float(mttr.quantile(0.95)), 2),
            "min": round(float(mttr.min()), 2),
            "max": round(float(mttr.max()), 2),
        },
    }

    # MTTR distribution buckets
    buckets = [0, 0.5, 1, 2, 4, 8, 16, 24, 48, 100, float('inf')]
    bucket_labels = ['<30m', '30m-1h', '1-2h', '2-4h', '4-8h', '8-16h', '16-24h', '1-2d', '2-4d', '4d+']
    hist = pd.cut(mttr, bins=buckets, labels=bucket_labels)
    dist = hist.value_counts().reindex(bucket_labels, fill_value=0)
    result["mttr_distribution"] = [
        {"bucket": str(b), "count": int(c), "pct": round(int(c) / len(work_df) * 100, 1)}
        for b, c in dist.items()
    ]

    # MTTR by severity
    if "priority_df" in work_df.columns:
        sev_mttr = work_df.groupby("priority_df")["mttr_excl_hold"].agg(
            ["mean", "median", "count", "std"]
        ).round(2)
        result["mttr_by_severity"] = [
            {
                "severity": int(k),
                "mean": round(float(v["mean"]), 2),
                "median": round(float(v["median"]), 2),
                "count": int(v["count"]),
                "std": round(float(v["std"]), 2) if pd.notna(v["std"]) else 0,
            }
            for k, v in sev_mttr.iterrows() if not pd.isna(k)
        ]

    # MTTR by category
    if "category" in work_df.columns:
        cat_mttr = work_df.groupby("category")["mttr_excl_hold"].agg(["mean", "median", "count"])
        cat_mttr = cat_mttr[cat_mttr["count"] >= 5].sort_values("mean", ascending=False).head(15)
        result["mttr_by_category"] = [
            {
                "category": str(k),
                "mean": round(float(v["mean"]), 2),
                "median": round(float(v["median"]), 2),
                "count": int(v["count"]),
            }
            for k, v in cat_mttr.iterrows()
        ]

    # MTTR by hour of day (when ticket was opened)
    if "open_dttm" in work_df.columns and pd.api.types.is_datetime64_any_dtype(work_df["open_dttm"]):
        time_df = work_df.dropna(subset=["open_dttm"]).copy()
        if len(time_df) > 0:
            time_df["_hour"] = time_df["open_dttm"].dt.hour
            hourly_mttr = time_df.groupby("_hour")["mttr_excl_hold"].agg(["mean", "count"]).round(2)
            result["mttr_by_hour"] = [
                {"hour": int(h), "mean_mttr": round(float(v["mean"]), 2), "count": int(v["count"])}
                for h, v in hourly_mttr.iterrows()
            ]

            # MTTR by day of week
            time_df["_dow"] = time_df["open_dttm"].dt.day_name()
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_mttr = time_df.groupby("_dow")["mttr_excl_hold"].agg(["mean", "count"]).round(2)
            result["mttr_by_day_of_week"] = [
                {"day": d, "mean_mttr": round(float(dow_mttr.loc[d, "mean"]), 2), "count": int(dow_mttr.loc[d, "count"])}
                for d in dow_order if d in dow_mttr.index
            ]

    # MTTR trend over months
    if "open_dttm" in work_df.columns and pd.api.types.is_datetime64_any_dtype(work_df["open_dttm"]):
        time_df = work_df.dropna(subset=["open_dttm"]).copy()
        if len(time_df) > 0:
            time_df = add_month_col(time_df)
            monthly_mttr = time_df.groupby("month")["mttr_excl_hold"].agg(["mean", "median", "count"]).round(2)
            result["mttr_monthly_trend"] = [
                {
                    "month": str(m),
                    "mean": round(float(v["mean"]), 2),
                    "median": round(float(v["median"]), 2),
                    "count": int(v["count"]),
                }
                for m, v in monthly_mttr.iterrows()
            ]
            # Overall trend direction
            if len(monthly_mttr) >= 4:
                first_half = float(monthly_mttr.iloc[:len(monthly_mttr)//2]["mean"].mean())
                second_half = float(monthly_mttr.iloc[len(monthly_mttr)//2:]["mean"].mean())
                if first_half > 0:
                    change = ((second_half - first_half) / first_half) * 100
                    result["mttr_overall_trend"] = "improving" if change < -10 else ("worsening" if change > 10 else "stable")
                    result["mttr_trend_change_pct"] = round(change, 1)

    # Outlier tickets (extremely high MTTR)
    p95 = float(mttr.quantile(0.95))
    outliers = work_df[work_df["mttr_excl_hold"] > p95].copy()
    if len(outliers) > 0:
        outlier_list = []
        for _, row in outliers.sort_values("mttr_excl_hold", ascending=False).head(10).iterrows():
            entry = {"mttr_hours": round(float(row["mttr_excl_hold"]), 2)}
            if "priority_df" in row.index and pd.notna(row["priority_df"]):
                entry["severity"] = int(row["priority_df"])
            if "description" in row.index and pd.notna(row["description"]):
                entry["description"] = str(row["description"])[:150]
            if "hostname" in row.index and pd.notna(row["hostname"]):
                entry["hostname"] = str(row["hostname"])
            if "business_application" in row.index and pd.notna(row["business_application"]):
                entry["application"] = str(row["business_application"])
            outlier_list.append(entry)
        result["outlier_tickets"] = outlier_list
        result["outlier_count"] = len(outliers)
        result["outlier_threshold_p95"] = round(p95, 2)

    # Predicted MTTR ranges by severity (simple percentile-based prediction)
    if "priority_df" in work_df.columns:
        predictions = []
        for sev in sorted(work_df["priority_df"].dropna().unique()):
            if pd.isna(sev):
                continue
            sev_df = work_df[work_df["priority_df"] == sev]["mttr_excl_hold"]
            if len(sev_df) >= 5:
                predictions.append({
                    "severity": int(sev),
                    "expected_range_low": round(float(sev_df.quantile(0.25)), 2),
                    "expected_range_high": round(float(sev_df.quantile(0.75)), 2),
                    "best_case": round(float(sev_df.quantile(0.10)), 2),
                    "worst_case": round(float(sev_df.quantile(0.90)), 2),
                    "sample_size": len(sev_df),
                })
        result["mttr_predictions_by_severity"] = predictions

    return result


# ═══════════════════════════════════════════════════════════
# COMBINED VESA ANALYSIS ENTRY POINT
# ═══════════════════════════════════════════════════════════

def perform_vesa_analysis(df):
    """Run all VESA analytics and return combined results."""
    return {
        "topic_clusters": analyze_topic_clusters(df),
        "similar_tickets": analyze_similar_tickets(df),
        "mttr_prediction": analyze_mttr_prediction(df),
    }
