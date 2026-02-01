# app.py - INVESTA AI: Business Analyzer (Updated for your exact dataset)
# Run: streamlit run app.py (place train.csv in same folder)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import time

# Page config
st.set_page_config(page_title="Investa AI", page_icon="💰", layout="wide")

# Premium CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
html, body { font-family: 'Plus Jakarta Sans', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.header { background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%); padding: 3rem 2rem; border-radius: 30px; color: white; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.2); }
.investa-title { font-size: 3.5rem; font-weight: 800; letter-spacing: -2px; margin: 0; }
.slogan { font-size: 1.3rem; opacity: 0.95; margin-top: 0.5rem; }
.metric-card { background: rgba(255,255,255,0.95); padding: 2rem; border-radius: 20px; box-shadow: 0 15px 40px rgba(0,0,0,0.1); border: 1px solid rgba(255,255,255,0.3); }
</style>
""", unsafe_allow_html=True)

# Header with Slogan
st.markdown("""
<div class="header">
    <h1 class="investa-title">INVESTA AI</h1>
    <p class="slogan">🔮 Precision Business Intelligence | Where Data Meets Destiny</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    return df

df = load_data()

# Train model for your exact columns
@st.cache_data
def train_model(df):
    le_domain = LabelEncoder()
    le_market = LabelEncoder()
    le_exp = LabelEncoder()
    le_loc = LabelEncoder()
    le_risk = LabelEncoder()
    
    df['DomainCode'] = le_domain.fit_transform(df['Startup_Domain'])
    df['MarketCode'] = le_market.fit_transform(df['Market_Size'])
    df['ExpCode'] = le_exp.fit_transform(df['Experience_Level'])
    df['LocCode'] = le_loc.fit_transform(df['Location'])
    
    features = ['Initial_Capital', 'DomainCode', 'MarketCode', 'Expected_Monthly_Revenue', 
                'Operational_Cost', 'Team_Size', 'ExpCode', 'LocCode']
    X = df[features]
    y = (df['Decision'] == 'Invest').astype(int)
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model, {'domain': le_domain, 'market': le_market, 'exp': le_exp, 'loc': le_loc}

model, encoders = train_model(df)

# Main interface - Your 8 inputs
st.subheader("📊 Business Analysis Engine")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Core Business**")
    domain = st.selectbox("Startup Domain", df['Startup_Domain'].unique(), key="domain")
    capital = st.number_input("Initial Capital", value=500000, key="capital")

with col2:
    st.markdown("**Financials**")
    market_size = st.selectbox("Market Size", df['Market_Size'].unique(), key="market")
    revenue = st.number_input("Expected Monthly Revenue", value=120000, key="revenue")
    costs = st.number_input("Operational Cost", value=80000, key="costs")

with col3:
    st.markdown("**Operations**")
    team_size = st.slider("Team Size", 1, 100, 15, key="team")
    exp_level = st.selectbox("Experience Level", df['Experience_Level'].unique(), key="exp")
    location = st.selectbox("Location", df['Location'].unique(), key="loc")

if st.button("🚀 ANALYZE INVESTMENT VIABILITY", type="primary", use_container_width=True):
    with st.spinner("Running AI Investment Analysis..."):
        time.sleep(1)
        
        # Predict using your exact dataset structure
        codes = {
            'domain': encoders['domain'].transform([domain])[0],
            'market': encoders['market'].transform([market_size])[0],
            'exp': encoders['exp'].transform([exp_level])[0],
            'loc': encoders['loc'].transform([location])[0]
        }
        
        input_data = np.array([[capital, codes['domain'], codes['market'], revenue, costs,
                               team_size, codes['exp'], codes['loc']]])
        
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0].max() * 100
        
        # Business metrics
        profit_margin = ((revenue - costs) / revenue * 100) if revenue > 0 else 0
        runway = capital / costs if costs > 0 else float('inf')
        growth_score = min(10, (profit_margin/10 + team_size/20 + confidence/20))
        
    # Results dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        color = "🟢 INVEST" if prediction else "🔴 AVOID"
        st.metric("AI Decision", color, f"{confidence:.1f}% Confidence")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Profit Margin", f"{profit_margin:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Runway (Months)", f"{runway:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Growth Score", f"{growth_score:.1f}/10")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(values=[confidence/100, 1-confidence/100], 
                    names=["Invest", "Avoid"], 
                    title="Investment Probability")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        runway_data = {'Months': range(1, min(25, int(runway)+1)), 
                      'Capital': [capital - i*costs for i in range(min(24, int(runway)))]}
        fig2 = px.line(pd.DataFrame(runway_data), title="Cash Runway")
        st.plotly_chart(fig2, use_container_width=True)

# Dataset preview
with st.expander("📈 Dataset Preview (train.csv)"):
    st.dataframe(df.head(10), use_container_width=True)
    st.caption("Powered by your real business data from Textile, Steel, IT Services, Food Processing...")

st.markdown("---")
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8);'>INVESTA AI 🔮 | Precision Business Intelligence</p>", unsafe_allow_html=True)
