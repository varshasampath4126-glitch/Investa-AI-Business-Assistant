import streamlit as st  
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Investa AI", page_icon="📈", layout="wide")

# --- 2. PREMIUM GLASS UI CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    
    /* Navbar Style */
    .nav-bar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 15px 30px; background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px); border-bottom: 1px solid rgba(255,255,255,0.3);
        border-radius: 0 0 20px 20px; margin-bottom: 30px;
    }
    .brand { font-size: 32px; font-weight: 800; color: #00468C; letter-spacing: -1px; }
    .loc { font-size: 14px; color: #555; background: #e0eaff; padding: 5px 15px; border-radius: 20px; }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(10px);
        border-radius: 25px; padding: 25px; border: 1px solid rgba(255,255,255,0.5);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
    }
    
    /* Level Up Animation */
    @keyframes slideUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
    .result-area { animation: slideUp 0.8s ease-out; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CUSTOM NAVBAR ---
st.markdown("""
    <div class="nav-bar">
        <div class="brand">INVESTA AI</div>
        <div class="loc">📍 Salem, TN</div>
    </div>
    """, unsafe_allow_html=True)

# --- 4. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('train.csv', encoding='latin1', on_bad_lines='skip')
        return df
    except:
        return pd.DataFrame()

df = load_data()

# --- 5. BACKEND ---
if not df.empty:
    le_domain, le_loc, le_exp = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df['Domain_Code'] = le_domain.fit_transform(df['Startup_Domain'])
    df['Loc_Code'] = le_loc.fit_transform(df['Location'])
    df['Exp_Code'] = le_exp.fit_transform(df['Experience_Level'])

    X = df[['Initial_Capital', 'Domain_Code', 'Loc_Code', 'Exp_Code', 'Team_Size']]
    y = df['Decision']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # --- MAIN UI ---
    col_input, col_result = st.columns([1, 1.4], gap="large")

    with col_input:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("✨ Startup Profile")
        domain = st.selectbox("Select Domain", df['Startup_Domain'].unique())
        location = st.selectbox("Salem Location", df['Location'].unique())
        capital = st.number_input("Seed Capital (₹)", min_value=10000, value=500000, step=50000)
        team = st.slider("Team Strength", 1, 50, 12)
        exp = st.selectbox("Exp Level", df['Experience_Level'].unique())
        
        analyze_btn = st.button("🚀 Analyze Growth Potential")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if analyze_btn:
            # Cute Loading Level
            with st.status("🔍 AI is mapping Salem's market trends...", expanded=True) as status:
                time.sleep(1)
                st.write("📊 Crunching startup data...")
                time.sleep(0.8)
                st.write("📈 Generating growth curves...")
                status.update(label="Analysis Complete! ✅", state="complete", expanded=False)
            
            # Predict
            d_code = le_domain.transform([domain])[0]
            l_code = le_loc.transform([location])[0]
            e_code = le_exp.transform([exp])[0]
            prediction = model.predict([[capital, d_code, l_code, e_code, team]])[0]

            st.markdown("<div class='result-area'>", unsafe_allow_html=True)
            
            # Prediction Card
            if prediction == "Invest":
                st.toast("Brilliant! Leveling up... 🚀")
                st.markdown(f"### AI Suggestion: <span style='color:#008000'>{prediction} ✨</span>", unsafe_allow_html=True)
                # Success Growth Path
                chart_data = pd.DataFrame(np.cumsum(np.random.randint(5, 15, size=20)), columns=['Market Strength'])
            else:
                st.markdown(f"### AI Suggestion: <span style='color:#FF8C00'>{prediction} 💡</span>", unsafe_allow_html=True)
                # Moderate Growth Path
                chart_data = pd.DataFrame(np.cumsum(np.random.randint(-2, 8, size=20)), columns=['Market Strength'])

            # Visual Chart
            st.line_chart(chart_data, color="#00468C", use_container_width=True)
            
            
            # Aesthetic Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Market Fit", "High" if team > 8 else "Mid")
            m2.metric("Risk Score", "Low" if prediction == "Invest" else "Mid")
            m3.metric("Scalability", "Top Tier" if capital > 200000 else "Stable")
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='text-align: center; padding-top: 50px; color: #777;'>
                    <h3>Welcome to the Future of Salem Startups 🏙️</h3>
                    <p>Enter your details on the left to see your growth projection level up!</p>
                </div>
            """, unsafe_allow_html=True)

else:
    st.error("Unga 'train.csv' file-ai check pannunga!")
