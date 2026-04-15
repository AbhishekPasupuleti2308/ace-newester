"""
llm_helpers.py — Azure OpenAI generation, system prompts, deterministic answering.
"""
import re
import json
import pandas as pd
from config import azure_client, AZURE_OPENAI_DEPLOYMENT


# =========================
# System Prompts
# =========================

SYSTEM_PROMPT_GENERAL = """You are a precise IT operations data analyst for Kyndryl Bridge Intelligence.
You have access to a comprehensive context JSON containing data from MULTIPLE analysis sources:

1. **Incident Analysis**: Ticket data including severity, MTTR, hostnames, applications, automation opportunities, common issues from description/resolution fields
2. **EOL/EOS Analysis**: End-of-Life and End-of-Service device health data from inventory, with application mapping
3. **Business Insights**: Actionable insights and growth insights from Kyndryl Bridge, per-host and per-application.
4. **VESA Analytics**: Topic clustering, similar ticket detection, MTTR prediction patterns.

RULES:
1. Use ONLY numbers from the context. NEVER invent or estimate.
2. For direct metric questions, give the exact number first, then a thorough explanation.
3. For 'how is X calculated' questions, answer in 2-4 lines with the full methodology.
4. Format numbers with commas. Use currency symbols where relevant.
5. If a value is not in the context, say exactly: "Not available in the current analysis."
6. You can answer questions about ANY section — incidents, EOL/EOS, insights, common issues, automation, VESA topics, similar tickets, MTTR predictions, etc.
7. When asked about common issues or root causes, use the common_issues data which contains analysis of the description and resolution columns.
8. When asked about business insights, provide DETAILED analysis including the observation methodology, key findings with specific numbers, affected applications/hosts, identified symptoms, and the full situation-problem-recommendation breakdown.
9. When asked about automation opportunities: Explain that automation opportunity is calculated by excluding tickets already auto-remediated (closure codes) and excluding all MAINFRAME tickets (ostype = "MAINFRAME"), then identifying tickets with valid suggested_automata values. Tickets with "unknown" or "other requests" automata are mapped via their label field. Final exclusions are made for "other-handler", "unknown", and "other requests" automata types.
10. When asked about topic clusters or emerging trends, reference the VESA topic clustering data with trend directions.
11. When asked about similar or duplicate tickets, reference the similar ticket groups and problem candidates.
12. When asked about MTTR predictions, use the MTTR prediction data including by severity, category, time patterns.
13. Provide comprehensive, detailed answers with specific data points and actionable recommendations.

CALCULATION METHODS:
- Automation Opportunity %: 
  1. First, exclude all tickets with closure_code = "Remediation with Corrective Closure" or "Remediation with Validation Closure"
  1b. Exclude all tickets where ostype = "MAINFRAME" (mainframe tickets are not eligible for automation)
  2. From remaining tickets, select those with a non-blank suggested_automata value
  3. For tickets with suggested_automata = "unknown" or "other requests", map the "label" field to a suggested_automata using predefined mappings (e.g., "Password Issue / Reset" → "password reset", "Disk Space" → "disk-handler", etc.)
  4. Exclude tickets with suggested_automata = "other-handler", "unknown", or "other requests" (after mapping)
  5. Remaining tickets = automation opportunity. Percentage = (automatable tickets / tickets after closure and mainframe exclusion) × 100
- Downtime: Severity 1 & 2 tickets. Yearly hours = average of each month's mean MTTR × 12.
- Noise: Severity 3 & 4 tickets. Same projection as downtime.
- Noise Yearly cost: Projected yearly hours × $144/hr (fixed noise cost rate).
- Downtime Yearly cost: Projected yearly hours × industry downtime cost per hour.
- MTTR: Mean Time To Resolve, from mttr_excl_hold column, in hours.
- EOL: End of Life — device OS has passed its end-of-life date.
- EOS: End of Service — device OS has passed its end-of-service date."""


SYSTEM_PROMPT_EXEC_SUMMARY = """You are a senior IT operations consultant for Kyndryl Bridge Intelligence.
You generate an Executive Summary with exactly 3 TOP OBSERVATIONS based on the data provided.

The user may provide additional instructions to refine, change, or redirect the observations.
ALWAYS follow the user's instructions.

CATEGORIES you can draw from (pick whichever 3 are most impactful unless the user specifies otherwise):
   - AUTOMATION: Playbook adoption, automation opportunity percentage
   - DOWNTIME/COST: Top application downtime, projected cost savings
   - EOL/EOS RISK: End-of-life devices, applications running on EOL infrastructure
   - ACTIONABLE INSIGHTS: Growth insights, actionable insights from Kyndryl Bridge
   - COMMON ISSUES: Recurring failures, most common incident types
   - NOISE REDUCTION: Service request automation, severity 3&4 noise reduction
   - APPLICATION RISK: Top applications by incident volume, cost, or downtime
   - REASSIGNMENT/ROUTING: Ticket reassignment patterns
   - MTTR OPTIMIZATION: Mean time to resolve patterns, outlier hosts or apps
   - TOPIC CLUSTERS: Emerging incident trends, chronic vs spike patterns
   - SIMILAR TICKETS: Duplicate ticket groups, problem ticket candidates
   - MONITORING GAPS: Servers without monitoring

For each observation provide:
   - metric: A bold percentage or number
   - title: Short title
   - explanation: 2-3 sentence explanation
   - recommendation: A specific, actionable recommendation
   - detail_text: 5-8 sentence detailed analysis
   - estimated_outcome_low: Dollar value low estimate
   - estimated_outcome_high: Dollar value high estimate
   - outcome_label: Label for the outcome
   - category: The category

RESPONSE FORMAT — respond ONLY with valid JSON, no markdown fences:
{
  "observations": [
    {
      "metric": "50%",
      "title": "Automation Opportunity (52K Incidents)",
      "explanation": "...",
      "recommendation": "...",
      "detail_text": "...",
      "estimated_outcome_low": "$461K",
      "estimated_outcome_high": "$654K",
      "outcome_label": "Annual Cost Savings",
      "category": "AUTOMATION"
    }
  ]
}"""


# =========================
# Generation
# =========================

def generate_with_azure(prompt, max_output_tokens=1200, temperature=0.1, system_prompt=None):
    if azure_client is None:
        raise RuntimeError("Azure OpenAI not configured.")
    system = system_prompt or SYSTEM_PROMPT_GENERAL
    resp = azure_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    return resp.choices[0].message.content.strip()


# =========================
# Deterministic helpers
# =========================

def _get_df_for_type(df, analysis_type):
    if analysis_type == "non_managed" and "sso_ticket" in df.columns:
        return df[df["sso_ticket"].fillna("").astype(str).str.upper() == "N"]
    if analysis_type == "managed" and "sso_ticket" in df.columns:
        return df[df["sso_ticket"].fillna("").astype(str).str.upper() == "Y"]
    return df


def try_answer_deterministically(question, df, analysis_type):
    q = (question or "").strip().lower()
    m = re.search(r"\btop\s+(\d+)\b", q)

    if m and ("application" in q or "app" in q) and ("ticket" in q or "count" in q or "number" in q or "most" in q or "based on" in q):
        n = int(m.group(1))
        sub = _get_df_for_type(df, analysis_type)
        if "business_application" not in sub.columns:
            return "The `business_application` column is missing."
        vc = sub["business_application"].fillna("(blank)").astype(str).value_counts().head(n)
        lines = [f"Top {n} applications by ticket count ({analysis_type}):"]
        for i, (k, v) in enumerate(vc.items(), 1):
            lines.append(f"{i}. {k}: {int(v):,} tickets")
        return "\n".join(lines)

    if ("how many" in q or "count" in q or "number of" in q) and "ticket" in q and "for" in q and ("application" in q or "app" in q):
        parts = re.split(r"\bfor\b", question, flags=re.IGNORECASE)
        if len(parts) >= 2:
            app_name_raw = parts[-1].strip().strip(" ?.")
            if app_name_raw:
                sub = _get_df_for_type(df, analysis_type)
                if "business_application" not in sub.columns:
                    return "The `business_application` column is missing."
                count = int((sub["business_application"].fillna("").astype(str).str.strip().str.lower() == app_name_raw.strip().lower()).sum())
                return f"Ticket count for **{app_name_raw}** ({analysis_type}): **{count:,}**"

    return None
