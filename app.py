import streamlit as st  
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Investa AI | Salem", page_icon="🚀", layout="wide")

# --- 2. CUSTOM CSS (Attractive Design) ---
st.markdown("""
    <style>
    /* Main Background */
    .main { background-color: #f8f9fa; }
    
    /* Header Styling */
    .main-title { color: #1E3D59; font-size: 40px; font-weight: bold; text-align: center; margin-bottom: 10px; }
    .sub-title { color: #52616B; font-size: 18px; text-align: center; margin-bottom: 30px; }
    
    /* Prediction Card */
    .res-card { 
        background-color: white; padding: 25px; border-radius: 15px; 
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1); border-left: 8px solid #007bff;
    }
    
    /* Button Styling */
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 3.5em; 
        background-color: #007bff; color: white; font-weight: bold; border: none;
    }
    .stButton>button:hover { background-color: #0056b3; border: none; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('train.csv', encoding='latin1', on_bad_lines='skip', low_memory=False)
        return df
    except Exception as e:
        return pd.DataFrame()

df = load_data()

# --- 4. BACKEND & UI ---
if not df.empty:
    # Model Setup
    le_domain, le_loc, le_exp = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df['Domain_Code'] = le_domain.fit_transform(df['Startup_Domain'])
    df['Loc_Code'] = le_loc.fit_transform(df['Location'])
    df['Exp_Code'] = le_exp.fit_transform(df['Experience_Level'])

    X = df[['Initial_Capital', 'Domain_Code', 'Loc_Code', 'Exp_Code', 'Team_Size']]
    y = df['Decision']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Header
    st.markdown("<div class='main-title'>🚀 Investa AI: Salem Startup Analyzer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>AI-powered investment intelligence for the Salem startup ecosystem.</div>", unsafe_allow_html=True)
    st.divider()

    # Layout
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📋 Startup Profile")
        domain = st.selectbox("Startup Domain", df['Startup_Domain'].unique())
        location = st.selectbox("Location in Salem", df['Location'].unique())
        capital = st.number_input("Initial Capital (₹)", min_value=10000, value=500000, step=50000)
        team = st.slider("Total Team Size", 1, 100, 10)
        exp = st.selectbox("Team Experience", df['Experience_Level'].unique())

    with col2:
        st.subheader("📊 Investment Analysis")
        st.write("Click analyze to see the AI prediction.")
        
        if st.button("Run AI Analysis"):
            d_code = le_domain.transform([domain])[0]
            l_code = le_loc.transform([location])[0]
            e_code = le_exp.transform([exp])[0]
            
            prediction = model.predict([[capital, d_code, l_code, e_code, team]])[0]
            
            # Result Card
            color = "#28a745" if prediction == "Invest" else "#ffc107" if prediction == "Improve" else "#dc3545"
            
            st.markdown(f"""
                <div class='res-card' style='border-left-color: {color};'>
                    <h3 style='color: {color};'>AI Decision: {prediction}</h3>
                    <p>Based on Salem market data, this startup shows <b>{prediction}</b> potential.</p>
                    <hr>
                    <p style='font-size: 13px; color: gray;'>Note: This is an AI prediction based on historical patterns.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional Metrics for "Attractive" feel
            m1, m2 = st.columns(2)
            m1.metric("Market Fit", "High" if team > 5 else "Medium")
            m2.metric("Risk Level", "Low" if prediction == "Invest" else "High")
else:
    st.error("Dataset not found! Please check 'train.csv'.")
