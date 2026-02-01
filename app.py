import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- PAGE CONFIG ---
st.set_page_config(page_title="Investa AI", page_icon="🚀", layout="wide")

# --- CUSTOM CSS (GLASSMORPHISM & DASHBOARD STYLING) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #F8F9FB; }
    
    /* Card Styling */
    .metric-card {
        background: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #E0E0E0;
        text-align: center; color: white;
    }
    .card-title { font-size: 14px; opacity: 0.9; font-weight: 600; margin-bottom: 10px; }
    .card-value { font-size: 24px; font-weight: 800; }
    
    /* Color Themes for Cards */
    .blue-card { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .orange-card { background: linear-gradient(135deg, #f8a5c2 0%, #f7d794 100%); }
    .red-card { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .purple-card { background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%); }
    .green-card { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    
    /* Decision Button Visual */
    .decision-box {
        background: #27ae60; color: white; padding: 25px; 
        border-radius: 50px; text-align: center; font-weight: 800; font-size: 22px;
        box-shadow: 0 10px 20px rgba(39, 174, 96, 0.3); margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ML WORKFLOW & DATA LOADING ---
@st.cache_data
def process_ml():
    try:
        df = pd.read_csv('train.csv')
        
        # Preprocessing
        le = LabelEncoder()
        df['Domain_Code'] = le.fit_transform(df['Startup_Domain'])
        df['Loc_Code'] = le.fit_transform(df['Location'])
        df['Exp_Code'] = le.fit_transform(df['Experience_Level'])
        
        # 1. Supervised Learning (Random Forest)
        X = df[['Initial_Capital', 'Domain_Code', 'Loc_Code', 'Exp_Code', 'Team_Size']]
        y = df['Decision']
        model = RandomForestClassifier(n_estimators=100).fit(X, y)
        
        # 2. Unsupervised Learning (K-Means Clustering)
        scaler = StandardScaler()
        cluster_data = scaler.fit_transform(df[['Growth_Score', 'Profit_Margin']])
        kmeans = KMeans(n_clusters=3, n_init=10).fit(cluster_data)
        df['Cluster'] = kmeans.labels_
        
        return df, model, le, kmeans, scaler
    except Exception as e:
        st.error(f"Error loading train.csv: {e}")
        return pd.DataFrame(), None, None, None, None

df, model, le, kmeans, scaler = process_ml()

# --- SIDEBAR INPUTS ---
st.sidebar.header("📥 Startup Inputs")
domain = st.sidebar.selectbox("Startup Domain", df['Startup_Domain'].unique() if not df.empty else ["Tech"])
capital = st.sidebar.number_input("Initial Capital (₹)", value=500000)
market_size = st.sidebar.number_input("Market Size", value=12000)
rev = st.sidebar.number_input("Monthly Revenue (₹)", value=140000)
cost = st.sidebar.number_input("Operational Cost (₹)", value=140000)
team = st.sidebar.slider("Team Size", 1, 100, 5)
exp = st.sidebar.selectbox("Experience Level", df['Experience_Level'].unique() if not df.empty else ["Intermediate"])
loc = st.sidebar.selectbox("Location", df['Location'].unique() if not df.empty else ["Tier 1"])

# --- CALCULATIONS ---
profit = rev - cost
margin = (profit / rev * 100) if rev != 0 else 0
# Logic-based Growth Score (0-10)
raw_score = (market_size/10000 * 3) + (margin/10 * 4) + (team/10 * 3)
growth_score = min(max(raw_score, 0), 10)

# --- DASHBOARD UI ---
st.title("📊 Investa Dashboard")

if not df.empty:
    # Top Row: Metric Cards (Based on Image)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card blue-card"><div class="card-title">Initial Capital</div><div class="card-value">₹{capital:,}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card orange-card"><div class="card-title">Market Size</div><div class="card-value">{market_size:,}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card red-card"><div class="card-title">Team Size</div><div class="card-value">{team}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card purple-card"><div class="card-title">Summary (Yearly Rev)</div><div class="card-value">₹{rev*12:,}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Second Row
    m5, m6, m7, m8 = st.columns(4)
    with m5:
        st.markdown(f'<div class="metric-card green-card" style="background: #2ecc71"><div class="card-title">Pred. Monthly Revenue</div><div class="card-value">₹{rev:,}</div></div>', unsafe_allow_html=True)
    with m6:
        st.markdown(f'<div class="metric-card" style="background: #9b59b6"><div class="card-title">Operational Cost</div><div class="card-value">₹{cost:,}</div></div>', unsafe_allow_html=True)
    with m7:
        st.markdown(f'<div class="metric-card blue-card"><div class="card-title">Profit Margin</div><div class="card-value">{margin:.1f}%</div></div>', unsafe_allow_html=True)
    with m8:
        st.markdown(f'<div class="metric-card green-card"><div class="card-title">Growth Score</div><div class="card-value">{growth_score:.1f} / 10</div></div>', unsafe_allow_html=True)

    st.divider()

    # Third Row: Visualization & Decision
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("📈 Performance Analysis")
        chart_data = pd.DataFrame({
            "Category": ["Revenue", "Cost", "Profit"],
            "Amount": [rev, cost, profit]
        })
        fig = px.bar(chart_data, x="Category", y="Amount", color="Category", 
                     color_discrete_sequence=["#2ecc71", "#e74c3c", "#3498db"])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("🤖 AI Decision")
        # Prediction
        d_code = le.transform([domain])[0]
        l_code = le.transform([loc])[0]
        e_code = le.transform([exp])[0]
        prediction = model.predict([[capital, d_code, l_code, e_code, team]])[0]
        
        risk = "Low" if growth_score > 7 else "Medium" if growth_score > 4 else "High"
        
        st.write(f"**Risk Level:** {risk}")
        st.progress(growth_score/10)
        
        st.markdown(f'<div class="decision-box">{prediction.upper()}</div>', unsafe_allow_html=True)

    # K-Means Section
    st.subheader("🎯 Market Segmentation (K-Means)")
    fig_cluster = px.scatter(df, x="Growth_Score", y="Profit_Margin", color="Cluster", 
                             title="Startup Clusters based on Performance",
                             template="plotly_white")
    st.plotly_chart(fig_cluster, use_container_width=True)

else:
    st.warning("Please ensure 'train.csv' is in the same folder as app.py")
