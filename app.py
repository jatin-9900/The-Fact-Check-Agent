import streamlit as st
import pypdf
import json
import os
import requests
from typing import List, Dict, Any
from google import genai
from google.genai import types

# ------------------------------------------------------------------------------
# PAGE CONFIG & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="TruthLayer — AI Fact-Checking Engine",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #374151;
    }
    .status-verified { color: #10b981; font-weight: bold; }
    .status-inaccurate { color: #f59e0b; font-weight: bold; }
    .status-false { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SECRET HELPER FUNCTION
# ------------------------------------------------------------------------------
def get_secret(key_name: str) -> str:
    """Safely retrieves keys from environment or Streamlit secrets without throwing errors."""
    val = os.environ.get(key_name)
    if val:
        return val
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return ""

# ------------------------------------------------------------------------------
# API INITIALIZATION & SETUP
# ------------------------------------------------------------------------------
st.title("🔍 TruthLayer: Automated PDF Fact-Checker")
st.caption("Upload marketing content, whitepapers, or reports to cross-reference claims against live web data.")

# Retrieve default keys safely
default_gemini_key = get_secret("GEMINI_API_KEY")
default_tavily_key = get_secret("TAVILY_API_KEY")

with st.sidebar:
    st.header("🔑 API Credentials")
    
    gemini_api_key = st.text_input(
        "Gemini API Key", 
        value=default_gemini_key, 
        type="password"
    )
    
    tavily_api_key = st.text_input(
        "Tavily API Key", 
        value=default_tavily_key, 
        type="password"
    )
    
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. **Extracts** specific stats, dates, and technical figures.
    2. **Executes** live web searches for each claim.
    3. **Evaluates** evidence and flags inaccurate or false data.
    """)

# Guard against missing or empty keys
if not gemini_api_key or len(gemini_api_key.strip()) < 10 or not tavily_api_key or len(tavily_api_key.strip()) < 5:
    st.warning("⚠️ Please provide valid **Gemini API Key** and **Tavily API Key** in the sidebar to run the application.")
    st.stop()

# Initialize Gemini Client with active sidebar key
client = genai.Client(api_key=gemini_api_key.strip())

# ------------------------------------------------------------------------------
# CORE HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def extract_text_from_pdf(pdf_file) -> str:
    """Extracts raw text content from uploaded PDF file."""
    pdf_reader = pypdf.PdfReader(pdf_file)
    extracted_text = ""
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            extracted_text += f"\n--- Page {page_num + 1} ---\n" + text
    return extracted_text

def extract_claims_from_text(raw_text: str) -> List[Dict[str, str]]:
    """Uses Gemini to extract verifiable statistical, numerical, or factual claims."""
    prompt = f"""
    Analyze the following text and extract all specific, verifiable claims.
    Focus strictly on:
    - Statistics, percentages, and metrics.
    - Dates, timelines, and years.
    - Financial values, market sizes, and numbers.
    - Technical specifications and hard facts.

    Ignore generic marketing fluff, subjective opinions, or non-verifiable statements.

    Return a JSON array of objects with the exact schema:
    [
      {{
        "claim_id": 1,
        "original_text": "Exact sentence or snippet from text",
        "claim_statement": "Concise, stand-alone factual proposition to verify",
        "search_query": "Optimized web search query to check this claim"
      }}
    ]

    Document Text:
    {raw_text[:12000]}
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    
    try:
        claims = json.loads(response.text)
        return claims
    except Exception as e:
        st.error(f"Failed to parse extracted claims: {e}")
        return []

def execute_web_search(query: str) -> List[Dict[str, str]]:
    """Fetches live search results using Tavily Web Search API."""
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": tavily_api_key.strip(),
        "query": query,
        "search_depth": "basic",
        "max_results": 3
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            results = res.json().get("results", [])
            return [{"title": r.get("title"), "snippet": r.get("content"), "url": r.get("url")} for r in results]
    except Exception as e:
        st.warning(f"Search API error for query '{query}': {e}")
    return []

def verify_claim(claim: Dict[str, str], search_results: List[Dict[str, str]]) -> Dict[str, Any]:
    """Cross-references a claim against live web evidence using Gemini."""
    
    evidence_str = "\n".join([f"- [{r['title']}]({r['url']}): {r['snippet']}" for r in search_results])
    
    prompt = f"""
    You are an objective AI Fact-Checker. Evaluate the following claim against live web evidence.

    Claim to Verify: "{claim['claim_statement']}"
    Original Context in Document: "{claim['original_text']}"

    Live Web Evidence:
    {evidence_str if evidence_str else "No relevant web evidence was returned."}

    Classify the status into EXACTLY one of these three categories:
    1. VERIFIED: The live web evidence explicitly confirms the claim (stats, numbers, dates match).
    2. INACCURATE: The claim is outdated, slightly wrong, off by a notable margin, or misquoted.
    3. FALSE: The claim is contradicted by live evidence or is entirely unsubstantiated/fabricated.

    Return JSON with exact structure:
    {{
      "status": "VERIFIED" | "INACCURATE" | "FALSE",
      "explanation": "Clear 2-sentence explanation of why it is accurate or wrong.",
      "corrected_fact": "The actual current/true metric if inaccurate or false. If verified, state 'N/A'.",
      "confidence_score": 0.0 to 1.0
    }}
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    
    try:
        verification = json.loads(response.text)
        verification["evidence"] = search_results
        return verification
    except Exception as e:
        return {
            "status": "FALSE",
            "explanation": "Failed to parse verification response.",
            "corrected_fact": "Unknown",
            "confidence_score": 0.0,
            "evidence": []
        }

# ------------------------------------------------------------------------------
# FRONTEND INTERFACE & WORKFLOW
# ------------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload PDF Document for Automated Verification", type=["pdf"])

if uploaded_file is not None:
    st.info(f"📄 Processing `{uploaded_file.name}`...")
    
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
    
    if not pdf_text.strip():
        st.error("No readable text found in the uploaded PDF file.")
        st.stop()
        
    st.success("Text extracted successfully! Identifying factual claims...")
    
    with st.spinner("Analyzing document structure & extracting claims..."):
        claims = extract_claims_from_text(pdf_text)
    
    if not claims:
        st.warning("No clear statistical or factual claims were identified in this document.")
        st.stop()
        
    st.subheader(f"Found {len(claims)} Extractable Claims")
    
    # Progress Bar for Live Web Verification
    progress_bar = st.progress(0)
    results = []
    
    for idx, c in enumerate(claims):
        st.text(f"Verifying Claim {idx + 1}/{len(claims)}: {c['claim_statement'][:60]}...")
        
        # 1. Fetch live search evidence
        search_data = execute_web_search(c["search_query"])
        
        # 2. Verify claim against evidence
        verdict = verify_claim(c, search_data)
        
        results.append({
            "claim": c,
            "verdict": verdict
        })
        
        progress_bar.progress((idx + 1) / len(claims))
        
    progress_bar.empty()
    st.markdown("---")
    
    # --------------------------------------------------------------------------
    # SUMMARY DASHBOARD METRICS
    # --------------------------------------------------------------------------
    total_claims = len(results)
    verified_cnt = sum(1 for r in results if r['verdict']['status'] == 'VERIFIED')
    inaccurate_cnt = sum(1 for r in results if r['verdict']['status'] == 'INACCURATE')
    false_cnt = sum(1 for r in results if r['verdict']['status'] == 'FALSE')
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Claims Tested", total_claims)
    m2.metric("Verified (True)", verified_cnt)
    m3.metric("Inaccurate (Outdated)", inaccurate_cnt)
    m4.metric("False / Unverified", false_cnt)
    
    st.markdown("---")
    st.header("📋 Fact-Checking Verification Report")
    
    # Display Claim Breakdown
    for idx, item in enumerate(results):
        c = item["claim"]
        v = item["verdict"]
        status = v.get("status", "FALSE")
        
        if status == "VERIFIED":
            status_badge = "🟢 **VERIFIED**"
        elif status == "INACCURATE":
            status_badge = "🟡 **INACCURATE**"
        else:
            status_badge = "🔴 **FALSE / UNMATCHED**"
            
        with st.expander(f"Claim #{idx + 1}: {c['claim_statement']} — {status_badge}"):
            st.markdown(f"**Original Context in PDF:**")
            st.caption(f'"{c["original_text"]}"')
            
            st.markdown(f"**Verification Explanation:**")
            st.write(v.get("explanation", ""))
            
            if status in ["INACCURATE", "FALSE"]:
                st.markdown(f"**Corrected Fact / Current Metric:**")
                st.info(v.get("corrected_fact", "N/A"))
                
            st.markdown("**Live Web Citations:**")
            evidence_list = v.get("evidence", [])
            if evidence_list:
                for ev in evidence_list:
                    st.markdown(f"- [{ev['title']}]({ev['url']}) — *{ev['snippet']}*")
            else:
                st.write("No direct web citations returned.")