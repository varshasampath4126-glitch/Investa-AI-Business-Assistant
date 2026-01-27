import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="Investa AI", page_icon="🚀", layout="wide")

# --- CUSTOM CSS (HTML) ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .prediction-card { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    .header-text { text-align: center; color: #1e3d59; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    return df

try:
    df = load_data()
    
    # Preprocessing
    le_domain = LabelEncoder()
    le_loc = LabelEncoder()
    le_exp = LabelEncoder()
    
    df['Domain_Code'] = le_domain.fit_transform(df['Startup_Domain'])
    df['Loc_Code'] = le_loc.fit_transform(df['Location'])
    df['Exp_Code'] = le_exp.fit_transform(df['Experience_Level'])

    # Model for Decision Prediction
    X = df[['Initial_Capital', 'Domain_Code', 'Loc_Code', 'Exp_Code', 'Team_Size']]
    y = df['Decision']
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    # --- UI LAYOUT ---
    st.markdown("<h1 class='header-text'>🚀 Investa AI: Salem Startup Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Salem-la irukkara startup-oda profit matrum risk-ai AI moolama analyze pannunga.</p>", unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📋 Startup Details")
        domain = st.selectbox("Startup Domain", df['Startup_Domain'].unique())
        location = st.selectbox("Location in Salem", df['Location'].unique())
        capital = st.number_input("Initial Capital (₹)", min_value=10000, value=500000)
        revenue = st.number_input("Expected Monthly Revenue (₹)", min_value=5000, value=80000)
        cost = st.number_input("Operational Cost (₹)", min_value=1000, value=40000)
        team = st.slider("Team Size", 1, 100, 10)
        exp = st.selectbox("Team Experience Level", df['Experience_Level'].unique())

    with col2:
        st.subheader("📊 Analysis Result")
        if st.button("Analyze Investment"):
            # Math logic
            profit_margin = ((revenue - cost) / revenue) * 100
            growth_score = round((revenue / (capital/12)) * 10, 1)
            if growth_score > 10: growth_score = 9.5 # Capping for realism
            
            # ML Prediction
            d_code = le_domain.transform([domain])[0]
            l_code = le_loc.transform([location])[0]
            e_code = le_exp.transform([exp])[0]
            
            prediction = model.predict([[capital, d_code, l_code, e_code, team]])[0]
            
            # Result Display
            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            
            # Decision Color Logic
            if prediction == "Invest": color = "green"
            elif prediction == "Improve": color = "orange"
            else: color = "red"
            
            st.markdown(f"### Decision: <span style='color:{color}'>{prediction}</span>", unsafe_allow_html=True)
            st.metric("Profit Margin", f"{profit_margin:.1f}%")
            st.metric("Growth Score", f"{growth_score}/10")
            
            risk = "Low" if profit_margin > 25 else "Medium" if profit_margin > 10 else "High"
            st.info(f"Predicted Risk Level: **{risk}**")
            
            if prediction == "Invest":
                st.write(f"💡 **AI Suggestion:** Indha startup-kku ₹{capital * 1.2:,.0f} varai invest seiyalam.")
            
            st.markdown("</div>", unsafe_allow_html=True)

except FileNotFoundError:
    st.error("Error: 'train.csv' file-ai system-la kandu pudikka mudiyala. Check pannunga!")