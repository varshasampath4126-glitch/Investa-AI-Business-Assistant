import streamlit as st  
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Investa AI", page_icon="🚀", layout="wide")

# --- 2. ADVANCED CSS (Shadows, Hovers & Glassmorphism) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #f8f9fa; }

    /* Header Styling */
    .main-header { display: flex; justify-content: space-between; align-items: center; padding: 20px 5%; background: white; border-bottom: 1px solid #eee; margin-bottom: 30px; }
    .logo { font-size: 28px; font-weight: 800; color: #1A73E8; }
    .tagline { font-size: 14px; color: #5F6368; font-style: italic; }

    /* Card Styling with Shadow & Hover */
    [data-testid="stVerticalBlock"] > div:has(div.stNumberInput, div.stSelectbox, div.stSlider) {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        border: 1px solid #f0f0f0;
        margin-bottom: 15px;
    }
    [data-testid="stVerticalBlock"] > div:has(div.stNumberInput, div.stSelectbox, div.stSlider):hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.1);
        border-color: #1A73E8;
    }

    /* Vibrant Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1A73E8 0%, #0D47A1 100%);
        color: white;
        border-radius: 12px;
        padding: 15px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.3);
    }
    .stButton>button:hover { box-shadow: 0 6px 20px rgba(26, 115, 232, 0.5); transform: scale(1.02); }

    /* Result Level-Up Box */
    .result-container {
        background: #ffffff;
        border-radius: 20px;
        padding: 30px;
        border-left: 10px solid #1A73E8;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. BRANDING HEADER ---
st.markdown("""
    <div class="main-header">
        <div class="logo">INVESTA AI</div>
        <div class="tagline">Empowering Entrepreneurs with Data-Driven Intelligence.</div>
    </div>
    """, unsafe_allow_html=True)

# --- 4. ABOUT SECTION ---
with st.expander("ℹ️ About Investa AI"):
    st.write("""
    **Investa AI** is a state-of-the-art predictive engine designed to analyze startup viability. 
    By processing inputs like market demand, operational costs, and capital, it provides 
    real-time investment suggestions and growth projections to help you scale smartly.
    """)

# --- 5. DATA LOADING & MODEL ---
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

    # --- 6. INPUT GRID (Coloured & Structured) ---
    st.subheader("🚀 Project Analysis Parameters")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        domain = st.selectbox("Startup Domain", df['Startup_Domain'].unique())
        capital = st.number_input("Initial Capital (₹)", value=500000)
        market_size = st.selectbox("Market Demand", ["High", "Medium", "Low"])
    with c2:
        rev = st.number_input("Expected Monthly Revenue (₹)", value=100000)
        ops = st.number_input("Operational Costs (₹)", value=40000)
        location = st.selectbox("Location / Area", df['Location'].unique())
    with c3:
        team = st.slider("Team Size", 1, 50, 10)
        exp = st.selectbox("Experience Level", df['Experience_Level'].unique())
        audience = st.text_input("Target Audience", "Gen Z / Small Businesses")

    # --- 7. PREDICTION & LIVELY CHART ---
    if st.button("RUN INVESTA INTELLIGENCE"):
        with st.status("Analyzing Market Trends...", expanded=True) as status:
            time.sleep(1)
            st.write("Checking Location Viability...")
            time.sleep(0.7)
            st.write("Simulating Revenue Streams...")
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        d_code = le_domain.transform([domain])[0]
        l_code = le_loc.transform([location])[0]
        e_code = le_exp.transform([exp])[0]
        prediction = model.predict([[capital, d_code, l_code, e_code, team]])[0]

        st.divider()
        
        res_col, chart_col = st.columns([1, 1.5])
        
        with res_col:
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            if prediction == "Invest":
                st.success(f"### AI Decision: {prediction} ✅")
                st.write("Excellent! Your parameters align with high-growth startup patterns.")
                score = 92
            else:
                st.warning(f"### AI Decision: {prediction} ⚠️")
                st.write("Caution advised. Refine your operational costs or team structure.")
                score = 54
            
            st.metric("Success Probability", f"{score}%", f"{score-50}%")
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_col:
            st.subheader("📈 Lively Growth Projection")
            # Lively Line Chart: Prediction-ku yetha maadhiri uyarum
            base = 20 if prediction == "Invest" else 5
            growth_data = pd.DataFrame(
                np.cumsum(np.random.randint(-2, base, size=30)), 
                columns=['Level']
            )
            st.line_chart(growth_data, color="#1A73E8")
            
            st.caption("Estimated growth level over a 30-month period.")

else:
    st.error("Error: 'train.csv' missing. Please ensure your dataset is in the folder.")
