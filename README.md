# The-Fact-Check-Agent
# 🔍 TruthLayer — Automated PDF Fact-Checking Agent

**TruthLayer** is an AI-powered automated fact-checking engine that reads PDF documents (marketing collateral, pitch decks, whitepapers, or strategy reports), extracts specific statistical, numerical, and date-based claims, cross-references them against live web data, and identifies inaccurate, outdated, or hallucinated claims.

---

## ✨ Features

- **📄 Robust PDF Parsing:** Reads and extracts raw text from single- or multi-page PDF documents.
- **🎯 Precise Claim Extraction:** Uses LLM structured outputs to identify concrete metrics, dates, percentages, and technical figures while ignoring generic fluff.
- **🌐 Real-Time Web Verification:** Leverages the Tavily Search API to execute targeted live web queries for every claim.
- **📊 Automatic Classification & Auditing:** Categorizes each extracted claim into one of three distinct statuses:
  - 🟢 **VERIFIED:** Stat/fact matches current live web data.
  - 🟡 **INACCURATE:** The claim is outdated, slightly misquoted, or missing context.
  - 🔴 **FALSE / UNMATCHED:** The claim is directly contradicted by live web evidence or is entirely fabricated.
- **💡 Real Fact Correction:** Automatically provides the accurate, current metric along with live web citations for quick verification.

---

## 🛠️ Tech Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **PDF Extraction:** `pypdf`
- **LLM Engine:** Google Gemini API (`google-genai` SDK)
- **Search Engine:** Tavily Web Search API
- **Language:** Python 3.10+

