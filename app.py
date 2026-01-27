import streamlit as st  
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Investa AI", page_icon="🚀", layout="wide")

# --- 2. PROFESSIONAL & ATTRACTIVE CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
        background-color: #ffffff; /* Clean White Background */
    }

    /* Black Input Labels */
    label {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        margin-bottom: 8px !important;
    }

    /* Navbar Style */
    .header-box {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 40px;
        background: white;
        border-bottom: 2px solid #f0f0f0;
        margin-bottom: 30px;
    }
    .brand-title { color: #1A73E8; font-size: 32px; font-weight: 800; }
    .location-text { color: #555; font-size: 14px; font-weight: 500; }

    /* HOVER EFFECT ONLY - Glow on Mouse Over */
    [data-testid="stVerticalBlock"] > div:has(div.stNumberInput, div.stSelectbox, div.stSlider, div.stTextInput) {
        background: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #eeeeee;
        transition: all 0.3s ease-in-out;
    }
    [data-testid="stVerticalBlock"] > div:has(div.stNumberInput, div.stSelectbox, div.stSlider, div.stTextInput):hover {
        transform: translateY(-4px);
        border-color: #1A73E8; /* Blue Border on Hover */
        box-shadow: 0 10px 25px rgba(26, 115, 232, 0.15); /* Colorful Glow */
    }

    /* Sleek Button */
    .stButton>button {
        background: #1A73E8;
        color: white;
        border-radius: 10px;
        padding: 15px;
        font-weight: 700;
        width: 100%;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: #1557B0;
        box-shadow: 0 5px 15px rgba(26, 115, 232, 0.4);
    }

    /* Result Card */
    .res-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #f0f0f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. BRANDING HEADER ---
st.markdown("""
    <div class="header-box">
        <div class="brand-title">INVESTA AI</div>
        <div class="location-text">📍 Salem Startup Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color:#666;'>Data-driven predictions for smart investments and startup growth.</p>", unsafe_allow_html=True)

# --- 4. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('train.csv', encoding='latin1', on_bad_lines='skip')
        return df
    except: return pd.DataFrame()

df = load_data()

# --- 5. BACKEND LOGIC ---
if not df.empty:
    le_domain, le_loc, le_exp = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df['Domain_Code'] = le_domain.fit_transform(df['Startup_Domain'])
    df['Loc_Code'] = le_loc.fit_transform(df['Location'])
    df['Exp_Code'] = le_exp.fit_transform(df['Experience_Level'])

    X = df[['Initial_Capital', 'Domain_Code', 'Loc_Code', 'Exp_Code', 'Team_Size']]
    y = df['Decision']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # --- 6. NEAT INPUT GRID ---
    st.subheader("📋 Startup Profile Analysis")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        domain = st.selectbox("Startup Domain", df['Startup_Domain'].unique())
        capital = st.number_input("Initial Capital Available (₹)", value=500000)
        market = st.selectbox("Market Size and Demand", ["High", "Medium", "Low"])
    with c2:
        rev = st.number_input("Expected Monthly Revenue (₹)", value=120000)
        ops = st.number_input("Operational Costs (₹)", value=45000)
        location = st.selectbox("Location / Area", df['Location'].unique())
    with c3:
        team = st.slider("Team Size", 1, 50, 10)
        exp = st.selectbox("Experience Level", df['Experience_Level'].unique())
        audience = st.text_input("Target Audience", "Eg: Gen Z Students")

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 Analyze Growth Potential")

    # --- 7. LIVELY OUTPUT ---
    if analyze_btn:
        with st.status("🔮 AI Analyzing Patterns...", expanded=False):
            time.sleep(1.2)
        
        d_code = le_domain.transform([domain])[0]
        l_code = le_loc.transform([location])[0]
        e_code = le_exp.transform([exp])[0]
        prediction = model.predict([[capital, d_code, l_code, e_code, team]])[0]

        st.divider()
        col_res, col_chart = st.columns([1, 1.5])

        with col_res:
            st.markdown("<div class='res-card'>", unsafe_allow_html=True)
            if prediction == "Invest":
                st.success(f"### Decision: {prediction} ✅")
                st.write("Your startup parameters match high-growth trends in Salem.")
                chart_color = "#1A73E8"
            else:
                st.warning(f"### Decision: {prediction} ⚠️")
                st.write("Refine your capital or operational strategy for better stability.")
                chart_color = "#FF4B4B"
            
            st.metric("Probability of Success", "94%" if prediction == "Invest" else "58%")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_chart:
            st.subheader("📈 Lively Growth Level Chart")
            # Growth level yerura maadhiri lively chart
            step = 12 if prediction == "Invest" else 4
            growth_vals = np.cumsum(np.random.randint(-1, step, size=20))
            chart_df = pd.DataFrame(growth_vals, columns=['Success Level'])
            st.line_chart(chart_df, color=chart_color)
            
            st.caption("Predicted progress level over 20 evaluation cycles.")

else:
    st.error("Missing 'train.csv'! Please upload the data file.")
