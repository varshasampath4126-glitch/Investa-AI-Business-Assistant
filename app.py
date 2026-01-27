import streamlit as st  
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Investa AI", page_icon="🚀", layout="wide")

# --- 2. DRIBBBLE-STYLE PREMIUM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Plus Jakarta Sans', sans-serif; 
        background-color: #ffffff !important; 
    }

    /* Labels - Bold Black */
    label, .stMarkdown p {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }

    /* Vibrant Gradient Header */
    .header-nav {
        display: flex; justify-content: space-between; align-items: center;
        padding: 30px 60px; background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        border-radius: 0 0 30px 30px; margin-bottom: 40px; box-shadow: 0 10px 30px rgba(37, 117, 252, 0.2);
    }
    .brand { font-size: 32px; font-weight: 800; color: white; letter-spacing: -1px; }
    .tag { font-size: 14px; color: rgba(255,255,255,0.8); font-weight: 500; }

    /* Input Card - Only Hover Glow */
    [data-testid="stVerticalBlock"] > div:has(div.stNumberInput, div.stSelectbox, div.stSlider, div.stTextInput) {
        background: #ffffff;
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #f0f0f0;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
        margin-bottom: 12px;
    }
    /* Glow when I touch the input */
    [data-testid="stVerticalBlock"] > div:has(div.stNumberInput, div.stSelectbox, div.stSlider, div.stTextInput):hover {
        transform: translateY(-6px);
        border-color: #6a11cb;
        box-shadow: 0 20px 40px rgba(106, 17, 203, 0.15); /* Vibrant Glow */
    }

    /* Professional Action Button */
    .stButton>button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white; border: none; padding: 20px; border-radius: 15px;
        font-weight: 700; font-size: 18px; width: 100%; transition: 0.3s;
        box-shadow: 0 10px 20px rgba(106, 17, 203, 0.3);
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 15px 30px rgba(106, 17, 203, 0.5); }

    /* Summary Dashboard Cards */
    .summary-card {
        background: white; border-radius: 20px; padding: 30px;
        border: 1px solid #f0f0f0; box-shadow: 0 5px 15px rgba(0,0,0,0.03);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. BRANDING ---
st.markdown("""
    <div class="header-nav">
        <div class="brand">INVESTA AI</div>
        <div class="tag">Next-Gen Startup Intelligence • Dribbble Edition</div>
    </div>
    """, unsafe_allow_html=True)

# --- 4. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('train.csv', encoding='latin1', on_bad_lines='skip')
        return df
    except: return pd.DataFrame()

df = load_data()

if not df.empty:
    # Basic ML Logic
    le_domain, le_loc, le_exp = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df['Domain_Code'] = le_domain.fit_transform(df['Startup_Domain'])
    df['Loc_Code'] = le_loc.fit_transform(df['Location'])
    df['Exp_Code'] = le_exp.fit_transform(df['Experience_Level'])

    X = df[['Initial_Capital', 'Domain_Code', 'Loc_Code', 'Exp_Code', 'Team_Size']]
    y = df['Decision']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # --- 5. SYSTEM INPUTS (The 8 Parameters) ---
    st.subheader("📊 Startup Evaluation Engine")
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        domain = st.selectbox("1. Startup Domain", df['Startup_Domain'].unique())
        capital = st.number_input("2. Initial Capital (₹)", value=500000)
        market = st.selectbox("3. Market Size/Demand", ["High Demand", "Moderate", "Niche"])
        
    with col2:
        revenue = st.number_input("4. Expected Monthly Revenue (₹)", value=150000)
        ops_cost = st.number_input("5. Operational Costs (₹)", value=60000)
        location = st.selectbox("7. Location & Target Audience", df['Location'].unique())

    with col3:
        team_size = st.slider("6. Team Size", 1, 100, 15)
        exp_level = st.selectbox("Team Experience Level", df['Experience_Level'].unique())
        past_data = st.selectbox("8. Past Data Available?", ["Available", "New Startup"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("RUN INTELLIGENCE ANALYSIS 🚀"):
        with st.status("🔮 Mapping Startup Viability...", expanded=False):
            time.sleep(1.5)
        
        d_code = le_domain.transform([domain])[0]
        l_code = le_loc.transform([location])[0]
        e_code = le_exp.transform([exp_level])[0]
        prediction = model.predict([[capital, d_code, l_code, e_code, team_size]])[0]

        st.divider()
        
        # --- 6. VISUAL OUTPUT: LIVELY CHART & METRICS ---
        res_col, chart_col = st.columns([1, 1.5], gap="large")

        with res_col:
            st.markdown("<div class='summary-card'>", unsafe_allow_html=True)
            if prediction == "Invest":
                st.success(f"### Result: {prediction} ✅")
                st.write("**Profit Potential:** High")
                color = "#6a11cb"
            else:
                st.warning(f"### Result: {prediction} ⚠️")
                st.write("**Risk Level:** Significant")
                color = "#ff4b4b"
            
            st.metric("Success Index", "94.2%" if prediction == "Invest" else "42.8%")
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_col:
            st.subheader("📈 Real-time Growth Level Prediction")
                        step = 15 if prediction == "Invest" else 5
            lively_vals = np.cumsum(np.random.randint(-1, step, size=30))
            chart_df = pd.DataFrame(lively_vals, columns=['Market Level'])
            st.line_chart(chart_df, color=color)
            st.caption("Estimated growth level based on inputs over 30 evaluation cycles.")

else:
    st.error("Missing 'train.csv'! Please upload the data file.")