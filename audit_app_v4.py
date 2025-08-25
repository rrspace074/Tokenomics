import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os
import re
from openai import OpenAI
import base64
import random

# --- Helper: Safe OpenAI API key retrieval ---
def get_openai_api_key():
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    try:
        if hasattr(st, "secrets") and hasattr(st.secrets, "__getitem__"):
            openai_section = st.secrets.get("openai") if hasattr(st.secrets, "get") else None
            if openai_section and isinstance(openai_section, dict):
                return openai_section.get("api_key")
    except Exception:
        pass
    return None

# Custom CSS for TDeFi theme
st.set_page_config(
    page_title="Tokenomics Audit AI", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for TDeFi branding and theme
st.markdown("""
<style>
    .company-logo {
        position: relative;
        background: transparent;
        padding: 0;
        border: none;
        box-shadow: none;
        max-width: 220px;
        max-height: 80px;
        margin: 10px 0 0 0;
        display: block;
    }
    .company-logo img {
        max-width: 100%;
        max-height: 60px;
        height: auto;
        object-fit: contain;
        display: block;
    }
    
    .main-header {
        background: linear-gradient(90deg, #ffc107 0%, #fd7e14 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.3);
        display: block;
    }
    .main-header-inner { 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        gap: 1.2rem;
        width: 100%;
    }
    .main-title {
        font-weight: 800;
        font-size: 2.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .powered-wrap {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        font-size: 1rem;
        opacity: 0.95;
    }
    

    
    .tdefi-powered {
        text-align: center;
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: white;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        padding: 1.5rem;
        border-radius: 15px;
        border: 3px solid #ffc107;
        box-shadow: 0 8px 25px rgba(255, 193, 7, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .tdefi-powered::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 193, 7, 0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .tdefi-logo-text {
        display: inline-block;
        background: linear-gradient(90deg, #ffc107 0%, #fd7e14 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
        font-weight: 900;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B35 0%, #FFD700 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B35;
        color: white;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def _get_logo_data_uri():
    candidates = [
        "your_logo.png",
        "your_logo.jpeg",
        "your_logo.jpg",
        "logo.png",
        "logo.jpg",
        "logo.jpeg",
    ]
    for filename in candidates:
        try:
            if os.path.exists(filename):
                mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
                with open(filename, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                return f"data:{mime};base64,{b64}"
        except Exception:
            continue
    return None

# Company Logo (placed below header)



# Main Header
_logo_uri = _get_logo_data_uri()
st.markdown(
    f"""
<div class="main-header">
  <div class="main-header-inner">
    <div class="main-title">🧠 <span>Tokenomics Audit AI Tool</span></div>
    <div class="powered-wrap">
        <span>powered by</span>
        {f'<img src="{_logo_uri}" alt="Company Logo" style="height: 72px;" />' if _logo_uri else ''}
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# (Logo is now shown inside header; no separate placement below)

# Project Inputs
col1, col2, col3 = st.columns(3)
with col1:
    project_name = st.text_input("Project Name")
    total_supply_tokens = st.number_input("Total Token Supply", min_value=1, value=1_000_000_000)
    tge_price = st.number_input("Token Price at TGE (USD)", min_value=0.00001, value=0.05, step=0.01, format="%.5f")
with col2:
    liquidity_fund = st.number_input("Liquidity Fund (USD)", min_value=0.0, value=500000.0, step=10000.0)
    project_type = st.selectbox("Project Category", ["Gaming", "DeFi", "NFT", "Infrastructure"])
    user_base = st.number_input("Current User Base", min_value=0, value=10000)
with col3:
    user_growth_rate = st.number_input("Monthly User Growth Rate (%)", min_value=0.0, value=10.0) / 100
    monthly_burn = st.number_input("Monthly Incentive Spend (USD)", min_value=0.0, value=50000.0, step=1000.0)
    revenue = st.number_input("Annual Revenue (USD)", min_value=0.0, value=1000000.0, step=10000.0)

# Token Allocation Table
st.header("📊 Step 1: Token Allocation & Vesting Schedule")
st.markdown("Enter pool details in the table below:")

# Create a more compact table-like input
pool_names = st.text_area("Pool Names (comma-separated)", value="Private Sale, Public Sale, Team, Ecosystem, Advisors, Listing, Airdrop, Staking", height=80)
pools = [p.strip() for p in pool_names.split(",") if p.strip()]

if len(pools) > 0:
    # Create table headers
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.markdown("**Pool Name**")
    with col2:
        st.markdown("**Allocation %**")
    with col3:
        st.markdown("**Category**")
    with col4:
        st.markdown("**TGE Unlock %**")
    with col5:
        st.markdown("**Cliff (months)**")
    with col6:
        st.markdown("**Vesting (months)**")
    with col7:
        st.markdown("**Sellable at TGE**")
    
    allocations, tags, vesting_schedule = {}, {}, {}
    total_alloc = 0
    
    for i, pool in enumerate(pools):
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            st.markdown(f"**{pool}**")
        with col2:
            allocations[pool] = st.number_input(f"", 0.0, 100.0, 0.0, step=1.0, key=f"alloc_{i}")
            total_alloc += allocations[pool]
        with col3:
            tags[pool] = st.selectbox("", ["VC", "Community", "Team", "Liquidity", "Advisor", "Other"], key=f"tag_{i}")
        with col4:
            tge = st.number_input("", 0.0, 100.0, 0.0, step=1.0, key=f"tge_{i}")
        with col5:
            cliff = st.number_input("", 0, 48, 0, key=f"cliff_{i}")
        with col6:
            vesting = st.number_input("", 0, 72, 12, key=f"vesting_{i}")
        with col7:
            # Automatically determine sellable at TGE based on TGE unlock percentage
            sellable = tge > 0
            sellable_text = "✅ Yes" if sellable else "❌ No"
            color_code = "#28a745" if sellable else "#dc3545"
            st.markdown(f"<div style='text-align: center; color: {color_code}; font-weight: bold;'>{sellable_text}</div>", unsafe_allow_html=True)
        
        vesting_schedule[pool] = {"cliff": cliff, "vesting": vesting, "tge": tge, "sellable": sellable}
    
    if total_alloc != 100:
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ Allocation must total 100%. Current: {total_alloc:.2f}%
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    else:
        st.markdown(f"""
        <div class="success-box">
            ✅ Total allocation: {total_alloc:.0f}%
        </div>
        """, unsafe_allow_html=True)

# Build release schedule dataframe
months = list(range(72))
df = pd.DataFrame({"Month": months})
df["Monthly Release %"] = 0
df["Cumulative %"] = 0

for pool in pools:
    tge_amount = allocations[pool] * vesting_schedule[pool]["tge"] / 100
    cliff = vesting_schedule[pool]["cliff"]
    vest = vesting_schedule[pool]["vesting"]
    monthly_amount = (allocations[pool] - tge_amount) / vest if vest > 0 else 0
    
    release = [0] * 72
    release[0] += tge_amount
    for m in range(cliff + 1, cliff + vest + 1):
        if m < 72:
            release[m] += monthly_amount
    df[pool] = release
    df["Monthly Release %"] += df[pool]

df["Cumulative %"] = df["Monthly Release %"].cumsum()

st.markdown("""
<div class="success-box">
    ✅ Vesting schedule processed successfully.
</div>
""", unsafe_allow_html=True)
# --- Audit Metrics ---

def inflation_guard(df):
    inflation = []
    for year in range(1, 6):
        prev = df.loc[(year - 1) * 12, "Cumulative %"] if (year - 1) * 12 < len(df) else 0
        curr = df.loc[year * 12, "Cumulative %"] if year * 12 < len(df) else df["Cumulative %"].iloc[-1]
        rate = ((curr - prev) / max(prev, 1)) * 100
        inflation.append(round(rate, 2))
    return inflation

def shock_stopper(df):
    return {
        "0–5%": df[(df["Monthly Release %"] <= 5)].shape[0],
        "5–10%": df[(df["Monthly Release %"] > 5) & (df["Monthly Release %"] <= 10)].shape[0],
        "10–15%": df[(df["Monthly Release %"] > 10) & (df["Monthly Release %"] <= 15)].shape[0],
        "15%+": df[df["Monthly Release %"] > 15].shape[0]
    }

def governance_hhi(allocations):
    shares = [v / 100 for v in allocations.values()]
    return round(sum([(s * 100) ** 2 for s in shares]) / 10000, 3)

def liquidity_shield(total_supply, tge_price, liquidity_usd, allocations, vesting):
    tge_percent = sum([allocations[p] * (vesting[p]["tge"] / 100) for p in allocations if p in vesting])
    tokens_at_tge = total_supply * (tge_percent / 100)
    market_cap_at_tge = tokens_at_tge * tge_price
    return round(liquidity_usd / market_cap_at_tge, 2) if market_cap_at_tge > 0 else 0

def lockup_ratio(vesting):
    locked = sum([1 for p in vesting if vesting[p]["cliff"] >= 12])
    return round(locked / len(vesting), 2)

def vc_dominance(allocations, tags):
    return round(sum([allocations[p] for p in allocations if tags.get(p) == "VC"]) / 100, 2)

def community_index(allocations, tags):
    return round(sum([allocations[p] for p in allocations if tags.get(p) == "Community"]) / 100, 2)

def emission_taper(df):
    first_12m = df.loc[0:11, "Monthly Release %"].sum()
    last_12m = df.loc[-12:, "Monthly Release %"].sum()
    return round(first_12m / last_12m if last_12m else 0, 2)

# --- Monte Carlo Simulation for Survivability ---
def monte_carlo_simulation(df, user_base, growth, months, category):
    if category == "Gaming":
        arpu = 76 * 12 * 0.03
    elif category == "DeFi":
        arpu = 3300 * 0.03
    elif category == "NFT":
        arpu = 59 * 0.03
    else:
        arpu = 50

    simulations = []
    for _ in range(100):
        price = 1.0
        for m in range(min(months, len(df))):
            users = user_base * ((1 + growth) ** (m / 12))
            buy_pressure = users * arpu
            sell_pressure = df.loc[m, "Monthly Release %"] * total_supply_tokens * tge_price
            price_change = (buy_pressure - sell_pressure) / max(sell_pressure, 1)
            price *= (1 + price_change * 0.01)
        simulations.append(price)
    return simulations

# --- Game Theory Audit ---
def game_theory_audit(hhi, shield, inflation):
    score = 0
    if hhi < 0.15: score += 1
    if shield >= 1.0: score += 1
    if inflation[0] <= 300: score += 1
    if inflation[1] <= 100: score += 1
    if inflation[2] <= 50: score += 1
    label = "Excellent" if score >= 4 else "Moderate" if score == 3 else "Needs Improvement"
    return label, score
if st.button("🚀 Generate Audit Report"):
    inflation = inflation_guard(df)
    shock = shock_stopper(df)
    hhi = governance_hhi(allocations)
    shield = liquidity_shield(total_supply_tokens, tge_price, liquidity_fund, allocations, vesting_schedule)
    lock = lockup_ratio(vesting_schedule)
    vc = vc_dominance(allocations, tags)
    community = community_index(allocations, tags)
    taper = emission_taper(df)
    monte = monte_carlo_simulation(df, user_base, user_growth_rate, 24, project_type)
    game_label, game_score = game_theory_audit(hhi, shield, inflation)

    # Game Theory Audit Metric with TDeFi styling
    st.markdown(f"""
    <div class="metric-card">
        <h3>🎯 Game Theory Audit Score: {game_score}/5</h3>
        <p style="font-size: 1.2rem; color: #FF6B35;">{game_label}</p>
    </div>
    """, unsafe_allow_html=True)

    # Create three columns for charts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(range(1, 6), inflation, marker='o', color='#FF6B35', linewidth=2, markersize=8)
        ax1.set_title("📈 Inflation Guard", fontweight='bold', color='#FF6B35')
        ax1.set_facecolor('#f8f9fa')
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']
        ax2.bar(shock.keys(), shock.values(), color=colors)
        ax2.set_title("🛡️ Shock Stopper", fontweight='bold', color='#FF6B35')
        ax2.set_facecolor('#f8f9fa')
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        if len(set(monte)) > 1:
            ax3.hist(monte, bins=min(20, len(set(monte))), color='#FF6B35', alpha=0.7, edgecolor='#FF8C42')
            ax3.set_title("🎲 Monte Carlo Simulation", fontweight='bold', color='#FF6B35')
        else:
            ax3.bar(["Simulated"], [monte[0]], color='#FF6B35', alpha=0.7)
            ax3.set_title("📊 Simulation Output", fontweight='bold', color='#FF6B35')
        ax3.set_facecolor('#f8f9fa')
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

    # GPT prompt
    api_key = get_openai_api_key()
    if not api_key:
        st.markdown("""
        <div class="warning-box">
            ❌ Missing OpenAI API key. Set OPENAI_API_KEY env var or add st.secrets['openai']['api_key'].
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    client = OpenAI(api_key=api_key)
    
    with st.spinner("🤖 AI is analyzing your tokenomics..."):
        prompt = f"""
        You are a senior tokenomics analyst auditing the project '{project_name}'. You've reviewed over 150 token models.

        For each metric below, give an analytical breakdown in this format:

        1. State the actual metric result (range or value)
        2. Interpret it: If high → what risk it carries; if low → what strength it shows
        3. Explain the possible impact on price behavior or investor perception

        Metrics to report:
        - 🟠 YoY Inflation: List values for Year 1–5
        - 🔴 Supply Shock: Count of months in ranges: 0–5%, 5–10%, >10%
        - 🟡 Governance HHI: Show score and interpret decentralization risk
        - 🔵 Liquidity Shield Ratio: Value and implication for price support
        - 🔒 Lockup Ratio: Fraction of pools with 12+ month cliff
        - 💼 VC Dominance: Combined allocation % of VC-tagged pools
        - 👥 Community Control Index: % allocated to community-tagged pools
        - 📉 Emission Taper Score: Ratio of first 12M emission to last 12M
        - 🎲 Monte Carlo Survivability: Summary of price outcomes after 100 runs
        - 🧠 Game Theory Score: Score (0–5) and what it reveals about design resilience

        Guidelines:
        - Do NOT repeat the definitions of each metric
        - Be analytical, insightful, and precise
        - This is for institutional readers — avoid fluff
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a tokenomics audit analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        summary = response.choices[0].message.content

    # Display AI Analysis with TDeFi styling
    st.markdown("""
    <div class="metric-card">
        <h3>🤖 AI Tokenomics Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(summary)

    def clean_gpt_text(summary):
        lines = summary.splitlines()
        formatted = []
        for line in lines:
            line = re.sub(r'^#+', '', line).strip()
            line = re.sub(r'^[-*•]\\s+', '', line).strip()
            if line:
                formatted.append(line)
        return formatted

    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 14)
            self.cell(0, 10, f"Tokenomics Audit Report - {project_name}", ln=True, align="C")
            self.cell(0, 10, "Powered by TDeFi - TradeDog Token Growth Studio", ln=True, align="C")

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, "Report by TDeFi", 0, 0, "C")

    def save_fig_temp(fig, name):
        path = os.path.join(tempfile.gettempdir(), name)
        fig.savefig(path)
        return path

    # Images
    img1 = save_fig_temp(fig1, "inflation.png")
    img2 = save_fig_temp(fig2, "shock.png")
    img3 = save_fig_temp(fig3, "sim.png")

    # PDF
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Compute effective page width for safe wrapping
    effective_page_width = pdf.w - 2 * pdf.l_margin

    def sanitize_text(text):
        t = (text or "").strip()
        t = t.encode("latin-1", "ignore").decode("latin-1")
        return t

    def write_multiline(text, line_height=8):
        t = sanitize_text(text)
        if not t:
            return
        # Guard against extremely long unbroken tokens by inserting soft breaks
        max_token_len = 120
        tokens = t.split(" ")
        rebuilt = []
        for token in tokens:
            if len(token) > max_token_len:
                # Split long token into chunks with spaces so fpdf can wrap
                chunks = [token[i:i+max_token_len] for i in range(0, len(token), max_token_len)]
                rebuilt.extend(chunks)
            else:
                rebuilt.append(token)
        safe_text = " ".join(rebuilt)
        pdf.multi_cell(effective_page_width, line_height, safe_text)
        pdf.ln(1)

    for para in clean_gpt_text(summary):
        write_multiline(para)

    for fig_path in [img1, img2, img3]:
        pdf.image(fig_path, x=10, w=180)
        pdf.ln(5)

    # TDeFi Note
    pdf.set_font("Arial", "B", 12)
    pdf.multi_cell(0, 10, "Token Design by TDeFi")
    pdf.set_font("Arial", "", 11)
    tdefi_note = """
    Token engineering is not about distribution schedules or supply caps alone - it is the structured discipline of aligning incentives, behavior, and long-term value creation. At its core, it is the science of designing economic systems where every stakeholder, from early investors to late-stage contributors, is guided by aligned motivations. A well-engineered token system creates harmony between product usage, network growth, and token demand. Poor token design, on the other hand, often leads to value leakage, unsustainable emissions, and ultimately, failure of both the product and its economy. At TDeFi, we don't treat tokenomics as an afterthought. We build token models that function as economic engines - driven by utility, governed by logic, and sustained by real-world adoption.

    Designing for Demand, Not Just Distribution
    A common pitfall in tokenized ecosystems is the decoupling of token emissions from actual product usage or demand. When supply outpaces utility, it triggers sell pressure, user attrition, and a negative flywheel. Our approach at TDeFi ensures that token release schedules are dynamically tied to verifiable demand metrics - whether that's active users, protocol revenue, or ecosystem participation. We engineer feedback loops that reward value creation, not just speculation. Tokenization is a powerful instrument, but only when wielded with precision - designed not as a shortcut to liquidity, but as a long-term mechanism for decentralized ownership, utility alignment, and sustainable network growth.
    """
    for para in tdefi_note.strip().split("\n\n"):
        write_multiline(para)

    pdf_out = pdf.output(dest='S')
    if isinstance(pdf_out, (bytes, bytearray)):
        pdf_bytes = bytes(pdf_out)
    else:
        pdf_bytes = str(pdf_out).encode('latin-1', 'ignore')
    
    # Download button with TDeFi styling
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h3>📄 Download Complete Report</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.download_button(
        "📄 Download PDF Report", 
        data=pdf_bytes, 
        file_name=f"{project_name}_Audit_Report.pdf", 
        mime="application/pdf"
    )

# Powered by TDeFi footer
st.markdown("""
<div class="tdefi-powered">
    🚀 Powered by <strong>TDeFi</strong> - TradeDog Token Growth Studio
</div>
""", unsafe_allow_html=True)
