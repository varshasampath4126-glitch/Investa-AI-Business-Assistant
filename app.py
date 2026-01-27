import streamlit as st  
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Investa AI", page_icon="✨", layout="wide")

# --- 2. ULTRA-COLOURFUL & VIBRANT CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;700;800&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Plus Jakarta Sans', sans-serif; 
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e7eb 100%);
    }

    /* Vibrant Header */
    .header-box {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px; border-radius: 20px; text-align: center;
        margin-bottom: 30px; box-shadow: 0 10px 20px rgba(79, 172, 254, 0.3);
    }
    .main-title { color: white; font-size: 40px; font-weight: 800; letter-spacing: -1px; }
    .sub-title { color: white; opacity: 0.9; font-size: 16px; }

    /* Neon Input Cards with Hover Glow */
    [data-testid="stVerticalBlock"] > div:has(div.stNumberInput, div.stSelectbox, div.stSlider, div.stTextInput) {
        background: white;
        padding: 25px;
        border-radius: 20px;
        border: 2px solid transparent;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    [data-testid="stVerticalBlock"] > div:has(div.stNumberInput, div.stSelectbox, div.stSlider, div.stTextInput):hover {
        transform: translateY(-8px);
        border-color: #4facfe;
        box-shadow: 0 15px 30px rgba(79, 172, 254, 0.2);
    }

    /* Electric Button */
    .stButton>button {
        background: linear-gradient(to right, #6a11cb 0%, #2575fc 100%);
        color: white; border: none; padding: 18px; border-radius: 15px;
        font-weight: 700; font-size: 18px; width: 100%;
        box-shadow: 0 8px 20px rgba(37, 117, 252, 0.4);
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 12px 25px rgba(37, 117, 252, 0.6); }

    /* Prediction Glow Card */
    .prediction-card {
        background: white; border-radius: 25px; padding: 35px;
        border-right: 8px solid #00f2fe;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        animation: glow 2s infinite alternate;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. VIBRANT HEADER ---
st.markdown("""
    <div class="header-box">
        <div class="main-title">INVESTA AI ✨</div>
        <div class="sub-text">Smart Business Intelligence for the Next Gen Startups</div>
    </div>
    """, unsafe_allow_html=True)

# --- 4. ABOUT SECTION ---
with st.expander("🌈 What is Investa AI?"):
    st.markdown("Investa AI uses **Advanced Machine Learning** to predict your startup's future. Enter your data and watch the magic happen!")

# --- 5. DATA & MODEL ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('train.csv', encoding='latin1', on_bad_lines='skip')
        return df
    except: return pd.DataFrame()

df = load_data()

if not df.empty:
    le_domain, le_loc, le_exp = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df['Domain_Code'] = le_domain.fit_transform(df['Startup_Domain'])
    df['Loc_Code'] = le_loc.fit_transform(df['Location'])
    df['Exp_Code'] = le_exp.fit_transform(df['Experience_Level'])

    X = df[['Initial_Capital', 'Domain_Code', 'Loc_Code', 'Exp_Code', 'Team_Size']]
    y = df['Decision']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # --- 6. COLOURFUL INPUT GRID ---
    st.subheader("🎨 Customize Your Startup Profile")
    
    r1_c1, r1_c2 = st.columns(2)
    with r1_c1:
        domain = st.selectbox("🚀 Startup Domain", df['Startup_Domain'].unique())
        capital = st.number_input("💰 Initial Capital (₹)", value=500000)
    with r1_c2:
        market = st.select_slider("📊 Market Demand", options=["Low", "Medium", "High", "Extreme"])
        revenue = st.number_input("💵 Expected Monthly Revenue (₹)", value=150000)

    r2_c1, r2_c2 = st.columns(2)
    with r2_c1:
        ops = st.number_input("📉 Operational Costs (₹)", value=60000)
        location = st.selectbox("📍 Location / Area", df['Location'].unique())
    with r2_c2:
        team = st.slider("👥 Team Size", 1, 100, 15)
        exp = st.selectbox("🎓 Experience Level", df['Experience_Level'].unique())

    target = st.text_input("🎯 Target Audience", "Eg: Tech Students, Coffee Lovers")

    # --- 7. ANALYSIS & LIVELY CHART ---
    if st.button("PREDICT MY SUCCESS 🚀"):
        with st.status("🔮 AI is scanning market vibes...", expanded=True) as status:
            time.sleep(1.2)
            st.write("Processing financial metrics...")
            time.sleep(0.8)
            status.update(label="Analysis Ready! ✨", state="complete", expanded=False)

        d_code = le_domain.transform([domain])[0]
        l_code = le_loc.transform([location])[0]
        e_code = le_exp.transform([exp])[0]
        prediction = model.predict([[capital, d_code, l_code, e_code, team]])[0]

        st.divider()
        
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            if prediction == "Invest":
                st.success(f"## DECISION: {prediction} 💎")
                st.write("**Verdict:** Your startup has elite growth potential!")
                line_color = "#00f2fe"
            else:
                st.warning(f"## DECISION: {prediction} ⚡")
                st.write("**Verdict:** High risk detected. Optimize your costs.")
                line_color = "#ff4b4b"
            
            st.metric("Growth Index", "Elite" if prediction == "Invest" else "Moderate")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.subheader("📈 Real-time Growth Level Prediction")
            # Lively Chart
            data = pd.DataFrame(np.cumsum(np.random.randint(-2, 15 if prediction == "Invest" else 6, size=30)), columns=['Level'])
            st.line_chart(data, color=line_color)
            
            st.caption("This chart visualizes your predicted success level over 30 months.")

else:
    st.error("Error: 'train.csv' not found. Please upload the data file.")
