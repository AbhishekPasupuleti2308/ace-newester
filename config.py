"""
config.py — Constants, Azure OpenAI setup, industry costs.
"""
import os
try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

# =========================
# Azure OpenAI setup
# =========================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()

AZURE_CONFIGURED = bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT)

azure_client = None
if AZURE_CONFIGURED and AzureOpenAI is not None:
    azure_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )

# =========================
# Industry costs
# =========================
INDUSTRY_DOWNTIME_COSTS = {
    "brokerage_service": 6500000,
    "auto": 3000000,
    "energy": 2500000,
    "telecommunications": 2000000,
    "retail": 1100000,
    "enterprise": 1000000,
    "healthcare": 600000,
    "manufacturing": 300000,
    "media": 100000,
}

EUR_TO_USD = 1.2
NOISE_COST_PER_HOUR = 144  # Fixed cost rate for noise (Sev 3 & 4)


def convert_currency(amount_usd, currency="USD"):
    if currency == "EUR":
        return amount_usd / EUR_TO_USD
    return amount_usd
