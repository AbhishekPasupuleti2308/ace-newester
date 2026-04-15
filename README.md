# Kyndryl Bridge Intelligence — Unified Data Platform

Merged extraction + analysis platform that combines:
- **Frontend extraction** (cookie-based APIs for BAM, Insights)
- **Backend extraction** (Elasticsearch for Incidents, Changes, Service Requests, Config)
- **Analysis engine** (from incident-analysis-dashboard)

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

## Workflow

1. **Select Account** — search and pick from the account dropdown
2. **Enter Cookie** — paste your session cookie from browser DevTools
3. **Validate** — the app checks if your cookie grants access
4. **Select Datasets** — choose which data to extract:
   - **Business App Mapping** → Frontend API
   - **Incidents** → Backend (Elasticsearch)
   - **Service Requests** → Backend (Elasticsearch)
   - **Change Requests** → Backend (Elasticsearch)
   - **Config Processed (Enriched)** → Backend (Elasticsearch + EoES enrichment)
   - **Actionable Insights V1 Combined** → Frontend API
5. **Extract** — data is pulled in parallel, with download links
6. **Analyse** — run analysis on extracted datasets (incidents, changes, SRs)

## Dataset → Source Mapping

| Dataset | Source | Index / API |
|---------|--------|-------------|
| Business App Mapping | Frontend | `/api/aiops-selfservice/v1/download/business-applications-mapping` |
| Incidents | Backend | `cdi_incident_tickets_processed` |
| Service Requests | Backend | `cdi_service_requests_processed` |
| Change Requests | Backend | `cdi_changerequests_processed` |
| Config Enriched | Backend | `cdi_configurations_processed` + `delivery.inventory.common-data` |
| Actionable Insights | Frontend | `/api/aiops/v1/actionableInsights/*` |

## Notes

- VPN is required to access both frontend APIs and backend Elasticsearch
- Cookie expires periodically — re-paste when you get auth errors
- PPT generation has been removed from this version
