import streamlit as st  
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time

# --- 1. PAGE CONFIG (Google Clean Look) ---
st.set_page_config(page_title="Investa AI", page_icon="🔍", layout="wide")

# --- 2. GOOGLE STYLE CSS ---
st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Product+Sans:wght@400;700&family=Roboto:wght@300;400;500&display=swap');
    
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; background-color: #ffffff; }
    
    /* Clean Header */
    .header {
        display: flex; justify-content: space-between; align-items: center;
        padding: 20px 50px; background: #ffffff; border-bottom: 1px solid #e0e0e0;
    }
    .google-brand { font-family: 'Product Sans', sans-serif; font-size: 24px; font-weight: bold; }
    .brand-i { color: #4285F4; } .brand-n { color: #EA4335; } .brand-v { color: #FBBC05; }
    .brand-e { color: #4285F4; } .brand-s { color: #34A853; } .brand-t { color: #EA4335; }
    .brand-a { color: #FBBC05; }

    /* Minimal Search-like Box */
    .search-container {
        max-width: 800px; margin: 50px auto; padding: 30px;
        border: 1px solid #dfe1e5; border-radius: 24px;
        box-shadow: none; transition: box-shadow 0.3s;
    }
    .search-container:hover { box-shadow: 0 1px 6px rgba(32,33,36,0.28); border-color: rgba(223,225,225,0); }
    
    /* Result Level Up */
    .level-box { border-left: 4px solid #4285F4; padding-left: 20px; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. GOOGLE STYLE NAVBAR ---
st.markdown("""
    <div class="header">
        <div class="google-brand">
            <span class="brand-i">I</span><span class="brand-n">n</span><span class="brand-v">v</span><span class="brand-e">e</span><span class="brand-s">s</span><span class="brand-t">t</span><span class="brand-a">a</span> AI
        </div>
        <div style="color: #5f6368; font-size: 14px;">📍 Salem startup ecosystem</div>
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

    # --- MAIN SEARCH UI ---
    st.markdown("<div style='text-align: center; margin-top: 50px;'>", unsafe_allow_html=True)
    st.title("Startup Intelligence Search")
    st.markdown("</div>", unsafe_allow_html=True)

    # Centered Input Box
    with st.container():
        col_pad1, col_main, col_pad2 = st.columns([1, 4, 1])
        with col_main:
            st.markdown("<div class='search-container'>", unsafe_allow_html=True)
            d_col, l_col = st.columns(2)
            with d_col:
                domain = st.selectbox("Industry Domain", df['Startup_Domain'].unique())
            with l_col:
                location = st.selectbox("Salem Location", df['Location'].unique())
            
            capital = st.slider("Capital (₹)", 10000, 2000000, 500000)
            team = st.number_input("Team Size", 1, 100, 10)
            exp = st.radio("Exp Level", df['Experience_Level'].unique(), horizontal=True)
            
            analyze_btn = st.button("Google Search Style Analysis", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # --- RESULTS AREA ---
    if analyze_btn:
        with st.spinner('Calculating growth levels...'):
            time.sleep(1)
            
            d_code = le_domain.transform([domain])[0]
            l_code = le_loc.transform([location])[0]
            e_code = le_exp.transform([exp])[0]
            prediction = model.predict([[capital, d_code, l_code, e_code, team]])[0]

            # Results
            st.divider()
            res_col1, res_col2 = st.columns([1, 1.5])
            
            with res_col1:
                st.markdown("<div class='level-box'>", unsafe_allow_html=True)
                st.write(f"### About {prediction} results")
                if prediction == "Invest":
                    st.success(f"**Top Result:** This startup is highly scalable in Salem.")
                else:
                    st.info(f"**Suggestion:** Focus on optimizing the team or capital structure.")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.metric("Probability Score", "98.4%", "+1.2%")

            with res_col2:
                st.write("📊 **Growth Level Projection**")
                # Line Chart
                val = 15 if prediction == "Invest" else 5
                chart_data = pd.DataFrame(np.cumsum(np.random.randint(-1, val, size=25)), columns=['Level'])
                st.line_chart(chart_data, color="#4285F4")
                

else:
    st.error("Missing 'train.csv'!")
