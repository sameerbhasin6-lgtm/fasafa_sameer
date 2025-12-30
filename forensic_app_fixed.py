import streamlit as st

# --- 1. SAFE IMPORTS & SETUP ---
try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    import numpy as np
    from pypdf import PdfReader
    import re
    import difflib
    from collections import Counter
    try:
        from textstat import textstat
        TEXTSTAT_AVAILABLE = True
    except ImportError:
        TEXTSTAT_AVAILABLE = False
    try:
        from textblob import TextBlob
        TEXTBLOB_AVAILABLE = True
    except ImportError:
        TEXTBLOB_AVAILABLE = False
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    st.error(f"❌ LIBRARY ERROR: {e}")
    st.warning("Please install: pip install streamlit pandas plotly matplotlib wordcloud scikit-learn pypdf openpyxl")
    st.stop()

st.set_page_config(page_title="Forensic & Credit Risk Master", page_icon="⚖️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #4e8cff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .risk-high { background-color: #ffe6e6; border-left: 5px solid #ff4b4b; padding: 15px; border-radius: 5px; }
    .risk-med { background-color: #fff8e6; border-left: 5px solid #ffa500; padding: 15px; border-radius: 5px; }
    .risk-low { background-color: #e6fffa; border-left: 5px solid #00cc96; padding: 15px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# MODULES & LOGIC
# ==========================================

class FinancialEngine:
    """Handles Credit Risk & Stress Testing"""
    def __init__(self):
        self.years = ['2019', '2020', '2021', '2022', '2023']
        self.financials = pd.DataFrame({
            "Year": self.years,
            "Revenue": [1000, 1100, 1050, 1300, 1450],
            "EBITDA": [200, 210, 180, 240, 250],
            "Net_Income": [80, 85, 40, 100, 110],
            "CFO": [180, 190, 120, 200, 150],
            "Total_Debt": [400, 450, 500, 600, 750],
            "Cash": [50, 60, 40, 50, 40],
            "Interest_Expense": [30, 32, 35, 45, 60]
        })
        self.financials['Net_Debt'] = self.financials['Total_Debt'] - self.financials['Cash']
        self.financials['Net_Leverage'] = self.financials['Net_Debt'] / self.financials['EBITDA'].replace(0, 1)
        self.financials['ICR'] = self.financials['EBITDA'] / self.financials['Interest_Expense'].replace(0, 1)

    def run_stress_test(self, rev_shock, margin_shock, rate_shock):
        base = self.financials.iloc[-1]
        new_rev = base['Revenue'] * (1 + rev_shock)
        new_ebitda = max(new_rev * ((base['EBITDA']/base['Revenue']) + margin_shock), 0.01)
        new_int = max(base['Interest_Expense'] + (base['Total_Debt'] * rate_shock), 0.01)
        new_lev = base['Net_Debt'] / new_ebitda
        new_icr = new_ebitda / new_int
        return {"EBITDA": new_ebitda, "Leverage": new_lev, "ICR": new_icr}

class QualitativeForensics:
    """Handles NLP & Intent Detection"""
    def __init__(self, text):
        self.text = text.lower() if text else ""

    def analyze_manipulation_risk(self):
        hedges = ['approximately', 'estimated', 'possibly', 'might', 'suggests', 'assumed']
        one_timers = ['exceptional', 'one-time', 'non-recurring', 'special item']
        h_count = sum(self.text.count(w) for w in hedges)
        o_count = sum(self.text.count(w) for w in one_timers)
        
        score = 0
        reasons = []
        if h_count > 10: score += 1; reasons.append("Excessive Hedging (Vagueness)")
        if o_count > 3: score += 1; reasons.append("Repeated 'One-Time' Items")
        
        if not reasons:
            reasons.append("No significant manipulation signals detected")
        
        rating = "HIGH" if score >= 2 else "MODERATE" if score == 1 else "LOW"
        color = "#ff4b4b" if score >= 2 else "#ffa500" if score == 1 else "#00cc96"
        return {"Rating": rating, "Color": color, "Drivers": reasons}

    def analyze_justification(self):
        triggers = ['because', 'due to', 'owing to', 'despite', 'attributed to', 'driven by']
        words = self.text.split()
        if len(words) == 0:
            return {"Level": "N/A", "Score": 0}
        count = sum(1 for w in words if w in triggers)
        density = (count / len(words)) * 1000
        level = "HIGH" if density > 15 else "MEDIUM" if density > 8 else "LOW"
        return {"Level": level, "Score": density}

    def detect_boilerplate(self, prev_text):
        if not prev_text or not self.text: 
            return "N/A - No comparison available"
        ratio = difflib.SequenceMatcher(None, self.text[:5000], prev_text[:5000].lower()).ratio()
        if ratio > 0.95: return "CRITICAL: Lazy Disclosure (Copy-Paste)"
        elif ratio > 0.85: return "HIGH: Boilerplate Repetition"
        return "LOW: Fresh Language"

@st.cache_data
def load_red_flag_dictionary(uploaded_file=None):
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            if 'Word' in df.columns:
                df['Word'] = df['Word'].astype(str).str.lower().str.strip()
                return df
        except Exception as e:
            st.warning(f"Could not load custom dictionary: {e}")
    
    # Fallback Data
    data = {
        "Word": ["contingent", "estimate", "fluctuate", "litigation", "claim", "uncertainty", 
                 "material", "adverse", "going concern", "restatement", "write-off", "impairment",
                 "restructuring", "provision", "exceptional", "alleged", "dispute", "risk"],
        "Category": ["Uncertainty", "Uncertainty", "Volatility", "Legal", "Legal", "Uncertainty", 
                     "Materiality", "Negative", "Viability", "Accounting", "Loss", "Loss",
                     "Restructuring", "Accounting", "Loss", "Legal", "Legal", "Uncertainty"]
    }
    return pd.DataFrame(data)

@st.cache_data
def extract_text_fast(file):
    text = ""
    try:
        reader = PdfReader(file)
        for i in range(min(50, len(reader.pages))):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text += page_text + "\n"
        return text if text else "Unable to extract text from PDF"
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def analyze_metrics(text, red_flag_df):
    if not text or len(text) < 100: 
        return None
    
    words = re.findall(r'\w+', text.lower())
    total_words = max(len(words), 1)
    
    # Fog Index
    fog = 0
    if TEXTSTAT_AVAILABLE:
        try:
            fog = textstat.gunning_fog(text[:5000])  # Limit text size
        except:
            fog = 0
    
    # Sentiment
    sent = 0
    if TEXTBLOB_AVAILABLE:
        try:
            sent = TextBlob(text[:5000]).sentiment.polarity
        except:
            sent = 0
    
    # Complex Words (simple heuristic if textstat unavailable)
    if TEXTSTAT_AVAILABLE:
        complex_words = [w for w in words if textstat.syllable_count(w) >= 3]
    else:
        complex_words = [w for w in words if len(w) > 10]  # Fallback heuristic
    
    complex_pct = (len(complex_words) / total_words) * 100
    
    # Red Flag Matching
    red_set = set(red_flag_df['Word'].unique()) if not red_flag_df.empty else set()
    matched = [w for w in words if w in red_set]
    
    word_to_cat = pd.Series(red_flag_df.Category.values, index=red_flag_df.Word).to_dict() if not red_flag_df.empty else {}
    cats = [word_to_cat.get(w, "Unknown") for w in matched]
    
    return {
        "fog": fog, 
        "sentiment": sent, 
        "complex_words": complex_words[:100],  # Limit for performance
        "complex_pct": complex_pct, 
        "matched_words": matched, 
        "matched_categories": cats,
        "total_words": total_words
    }

# ==========================================
# MAIN UI
# ==========================================
fin_engine = FinancialEngine()

with st.sidebar:
    st.title("📂 Inputs")
    file_curr = st.file_uploader("Current Year Report (PDF)", type="pdf")
    file_prev = st.file_uploader("Previous Year Report (PDF)", type="pdf")
    uploaded_dict = st.file_uploader("Custom Red Flags (CSV/Excel)", type=["csv", "xlsx"])
    
    st.markdown("---")
    st.header("Financial Smoke Test")
    net_income = st.number_input("Net Income (Cr)", value=0.0, step=10.0)
    cash_flow = st.number_input("Cash Flow (Cr)", value=0.0, step=10.0)

if file_curr:
    with st.spinner("Processing document..."):
        # 1. PROCESS DATA
        red_flags_df = load_red_flag_dictionary(uploaded_dict)
        text_curr = extract_text_fast(file_curr)
        
        if "Error" in text_curr or "Unable" in text_curr:
            st.error(text_curr)
            st.stop()
        
        metrics_curr = analyze_metrics(text_curr, red_flags_df)
        
        if metrics_curr is None:
            st.error("Could not analyze document - text too short or extraction failed")
            st.stop()
        
        text_prev = extract_text_fast(file_prev) if file_prev else None
        
    st.title("⚖️ Forensic & Credit Risk Master")
    
    # Accruals Alert
    if net_income > 0 and cash_flow > 0 and net_income > (cash_flow * 1.5):
        st.warning(f"⚠️ **ACCRUALS ALERT:** Net Income ({net_income:.0f}) > 1.5x Cash Flow ({cash_flow:.0f}). Check earnings quality.")

    # --- TABS FOR DIFFERENT PAGES ---
    tab1, tab2, tab3 = st.tabs(["1️⃣ Forensic Snapshot", "2️⃣ Regression Analysis", "3️⃣ Credit & Qualitative"])

    # ==========================================
    # PAGE 1: FORENSIC SNAPSHOT
    # ==========================================
    with tab1:
        st.subheader("📊 Document Complexity & Red Flags")
        
        # Row 1: Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Fog Index (Complexity)", f"{metrics_curr['fog']:.1f}", "Target: <18")
        c2.metric("Complex Word %", f"{metrics_curr['complex_pct']:.1f}%", "Long/Technical Words")
        c3.metric("Sentiment Score", f"{metrics_curr['sentiment']:.2f}", "-1 (Neg) to +1 (Pos)")
        
        st.markdown("---")
        
        # Row 2: Word Cloud & Pie Chart
        col_cloud, col_stats = st.columns([2, 1])
        
        with col_cloud:
            st.markdown("##### 🚩 Red Flag Word Cloud")
            if metrics_curr['matched_words'] and len(metrics_curr['matched_words']) > 0:
                try:
                    wc = WordCloud(
                        background_color="white", 
                        colormap="Reds", 
                        width=600, 
                        height=350, 
                        collocations=False,
                        min_font_size=10
                    ).generate(" ".join(metrics_curr['matched_words']))
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Could not generate word cloud: {e}")
            else:
                st.info("✅ No red flag words detected in document")

        with col_stats:
            st.markdown("##### 🚨 Top Red Flags")
            if metrics_curr['matched_words'] and len(metrics_curr['matched_words']) > 0:
                counts = Counter(metrics_curr['matched_words']).most_common(10)
                df_counts = pd.DataFrame(counts, columns=["Word", "Frequency"])
                st.dataframe(df_counts, hide_index=True, use_container_width=True, height=180)
                
                # Pie Chart
                st.markdown("##### 📊 By Category")
                cat_counts = Counter(metrics_curr['matched_categories'])
                if cat_counts:
                    fig_pie = px.pie(
                        names=list(cat_counts.keys()), 
                        values=list(cat_counts.values()), 
                        hole=0.4
                    )
                    fig_pie.update_layout(
                        margin=dict(t=0, b=0, l=0, r=0), 
                        height=180, 
                        showlegend=True,
                        legend=dict(font=dict(size=9))
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.success("✅ Document appears clean")

    # ==========================================
    # PAGE 2: REGRESSION ANALYSIS
    # ==========================================
    with tab2:
        st.subheader("📈 Industry Benchmark Regression")
        c_desc, c_chart = st.columns([1, 2])
        
        with c_desc:
            st.markdown("""
            **How to read this:**
            
            We compare your document against a simulated industry baseline of 50 companies.
            
            * **X-Axis:** Reading difficulty (Fog Index)
            * **Y-Axis:** Fraud Risk Score
            * **Blue Line:** Normal industry trend
            * **Red X:** Your document
            
            **📍 Interpretation:**
            - **Above the line:** Unjustified complexity = Higher risk
            - **On the line:** Normal for industry
            - **Below the line:** Clearer than average
            """)
            
        with c_chart:
            # Generate Benchmark Data
            np.random.seed(42)
            bench_fog = np.random.normal(16, 3, 50)
            bench_risk = (bench_fog * 2.5) + np.random.normal(0, 8, 50)
            df_bench = pd.DataFrame({"Fog Index": bench_fog, "Risk Score": bench_risk})
            
            # Train Model
            model = LinearRegression()
            model.fit(df_bench[["Fog Index"]], df_bench["Risk Score"])
            pred = model.predict([[metrics_curr['fog']]])[0]
            
            # Plot
            fig_reg = px.scatter(
                df_bench, 
                x="Fog Index", 
                y="Risk Score", 
                opacity=0.4, 
                title="Complexity vs. Fraud Risk (Industry Benchmark)"
            )
            line_x = np.linspace(df_bench["Fog Index"].min(), df_bench["Fog Index"].max(), 100).reshape(-1, 1)
            fig_reg.add_traces(
                go.Scatter(
                    x=line_x.flatten(), 
                    y=model.predict(line_x), 
                    mode='lines', 
                    name='Industry Trend',
                    line=dict(color='blue', width=2)
                )
            )
            fig_reg.add_traces(
                go.Scatter(
                    x=[metrics_curr['fog']], 
                    y=[pred], 
                    mode='markers+text', 
                    marker=dict(color='red', size=15, symbol='x'), 
                    name='Your Document', 
                    text=["YOUR DOC"], 
                    textposition="top center"
                )
            )
            fig_reg.update_layout(height=400)
            st.plotly_chart(fig_reg, use_container_width=True)

    # ==========================================
    # PAGE 3: CREDIT & QUALITATIVE
    # ==========================================
    with tab3:
        st.subheader("💰 Credit Risk & Qualitative Analysis")
        
        # 1. Credit Snapshot
        latest = fin_engine.financials.iloc[-1]
        risk_score = 0
        if latest['Net_Leverage'] > 4: risk_score += 1
        if latest['ICR'] < 3: risk_score += 1
        
        lvl = "HIGH" if risk_score >= 2 else "MEDIUM" if risk_score == 1 else "LOW"
        color = "#ff4b4b" if lvl == "HIGH" else "#ffa500" if lvl == "MEDIUM" else "#00cc96"
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(
                f'<div style="background-color:{color}20; padding:20px; border-left:5px solid {color}; border-radius:8px;">'
                f'<h3 style="margin:0;">Credit Risk Level</h3>'
                f'<h1 style="color:{color}; margin:10px 0;">{lvl}</h1></div>', 
                unsafe_allow_html=True
            )
        with c2:
            st.markdown("**📊 Key Metrics**")
            st.write(f"- **Net Leverage:** {latest['Net_Leverage']:.2f}x (Target: <4.0x)")
            st.write(f"- **Interest Coverage:** {latest['ICR']:.2f}x (Target: >3.0x)")
            st.write(f"- **Net Debt:** ₹{latest['Net_Debt']:.0f} Cr")
            st.caption("⚠️ Based on simulated data. Replace with real financial extraction.")

        st.markdown("---")
        
        # 2. Qualitative Forensics
        ql = QualitativeForensics(text_curr)
        risk = ql.analyze_manipulation_risk()
        
        col_qual1, col_qual2 = st.columns(2)
        with col_qual1:
            st.markdown(f"**🔍 Manipulation Risk: {risk['Rating']}**")
            for d in risk["Drivers"]: 
                st.markdown(f"• {d}")
            
        with col_qual2:
            jd = ql.analyze_justification()
            st.metric("Justification Density", f"{jd['Score']:.1f}", jd['Level'])
            st.caption("High density may indicate defensive/explanatory tone")
            
        # 3. Tone Drift
        if text_prev:
            st.markdown("---")
            st.subheader("🔄 Year-over-Year Language Drift")
            bp = ql.detect_boilerplate(text_prev)
            
            if "CRITICAL" in bp:
                st.error(f"🚨 {bp}")
            elif "HIGH" in bp:
                st.warning(f"⚠️ {bp}")
            else:
                st.success(f"✅ {bp}")

else:
    st.info("👆 Please upload a PDF document to begin forensic analysis")
    
    # Show sample capabilities
    st.markdown("---")
    st.subheader("🎯 What This Tool Does")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📊 Forensic Analysis**")
        st.write("- Document complexity (Fog Index)")
        st.write("- Red flag word detection")
        st.write("- Sentiment analysis")
        st.write("- Word cloud visualization")
    
    with col2:
        st.markdown("**📈 Benchmarking**")
        st.write("- Industry comparison")
        st.write("- Regression analysis")
        st.write("- Risk scoring")
        st.write("- Anomaly detection")
    
    with col3:
        st.markdown("**💰 Credit Risk**")
        st.write("- Leverage analysis")
        st.write("- Manipulation detection")
        st.write("- YoY comparison")
        st.write("- Stress testing")