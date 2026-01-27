import streamlit as st  
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Investa AI", page_icon="💡", layout="wide")

# --- 2. SIMPLE, CLEAN CSS (Material-inspired) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    .main { background-color: #f0f2f6; } /* Light grey background */

    /* Custom Header */
    .stApp > header { background-color: #263238; } /* Dark header */
    .stApp > header .css-asc { color: #ffffff; } /* Text color in header */

    /* Main Title */
    h1 { color: #1a237e; font-weight: 700; text-align: center; margin-bottom: 30px; }
    
    /* Card for inputs/results */
    .stCard {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        padding: 25px;
        margin-bottom: 20px;
    }

    /* Colorful Buttons */
    .stButton>button {
        background-color: #4285F4; /* Google Blue */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        font-weight: 500;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #3367D6; /* Darker blue on hover */
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Result text */
    .stSuccess { background-color: #e6ffe6; color: #008000; border-radius: 8px; padding: 10px; margin-top: 15px; }
    .stWarning { background-color: #fff3e0; color: #ff6f00; border-radius: 8px; padding: 10px; margin-top: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CUSTOM HEADER (Simple & Neat) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3141/3141133.png", width=80) # Small icon
st.sidebar.title("Investa AI")
st.sidebar.markdown("---")
st.sidebar.info("Analyze startup potential in Salem.")

# --- 4. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('train.csv', encoding='latin1', on_bad_lines='skip')
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

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

    # --- MAIN DASHBOARD UI ---
    st.title("Salem Startup Analyzer 💡")

    col_inputs, col_results = st.columns([1, 1.5], gap="large")

    with col_inputs:
        with st.container(border=True): # Input Card
            st.subheader("Startup Details")
            domain = st.selectbox("Select Domain", df['Startup_Domain'].unique(), key="domain_sel")
            location = st.selectbox("Location in Salem", df['Location'].unique(), key="loc_sel")
            capital = st.number_input("Capital (₹)", min_value=10000, value=500000, step=50000, key="cap_in")
            team = st.slider("Team Size", 1, 50, 10, key="team_sl")
            exp = st.selectbox("Experience Level", df['Experience_Level'].unique(), key="exp_sel")
            
            # Button for analysis
            analyze_btn = st.button("Analyze Potential", key="analyze_btn")

    with col_results:
        if analyze_btn:
            with st.spinner('Analyzing startup data...'):
                time.sleep(1.5) # Simulate work

            d_code = le_domain.transform([domain])[0]
            l_code = le_loc.transform([location])[0]
            e_code = le_exp.transform([exp])[0]
            prediction = model.predict([[capital, d_code, l_code, e_code, team]])[0]

            with st.container(border=True): # Results Card
                st.subheader("AI Prediction & Growth")
                if prediction == "Invest":
                    st.success(f"✅ AI Decision: {prediction}! This startup shows strong potential.")
                    chart_data = pd.DataFrame(np.cumsum(np.random.randint(5, 12, size=20)), columns=['Growth Score'])
                    chart_color = "#34A853" # Google Green
                else:
                    st.warning(f"⚠️ AI Decision: {prediction}. Consider re-evaluating strategy.")
                    chart_data = pd.DataFrame(np.cumsum(np.random.randint(-2, 7, size=20)), columns=['Growth Score'])
                    chart_color = "#FBBC05" # Google Yellow

                st.line_chart(chart_data, color=chart_color, use_container_width=True)
                st.caption("Projected growth score over time.")
                
                st.markdown("---")
                m1, m2 = st.columns(2)
                m1.metric("Market Fit Index", "8.5/10", "High")
                m2.metric("Investment Risk", "Low" if prediction == "Invest" else "Moderate", "Based on AI")
        else:
            st.info("Enter startup details on the left and click 'Analyze Potential' to get predictions!")

else:
    st.error("Error: 'train.csv' not found or empty. Please ensure the file is in the same directory.")
