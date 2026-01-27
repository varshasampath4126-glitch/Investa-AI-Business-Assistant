import streamlit as st  
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="Investa AI", page_icon="🚀", layout="wide")

# --- LOAD DATA (Robust Fix for Reader Errors) ---
@st.cache_data
def load_data():
    try:
        # Latin-1 encoding matrum low_memory=False pottadhala reader error varadhu
        df = pd.read_csv('train.csv', encoding='latin1', on_bad_lines='skip', low_memory=False)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

df = load_data()

# --- BACKEND LOGIC ---
if not df.empty:
    try:
        # Categorical columns-ai number-ah maatha LabelEncoder
        le_domain = LabelEncoder()
        le_loc = LabelEncoder()
        le_exp = LabelEncoder()
        
        df['Domain_Code'] = le_domain.fit_transform(df['Startup_Domain'])
        df['Loc_Code'] = le_loc.fit_transform(df['Location'])
        df['Exp_Code'] = le_exp.fit_transform(df['Experience_Level'])

        # Features (X) matrum Target (y)
        X = df[['Initial_Capital', 'Domain_Code', 'Loc_Code', 'Exp_Code', 'Team_Size']]
        y = df['Decision']
        
        # ML Model Training
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # --- FRONTEND UI ---
        st.title("🚀 Investa AI: Salem Startup Analyzer")
        st.write("Analyze startups in Salem using AI-driven insights.")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            domain = st.selectbox("Startup Domain", df['Startup_Domain'].unique())
            location = st.selectbox("Location in Salem", df['Location'].unique())
            capital = st.number_input("Initial Capital (₹)", min_value=10000, value=500000)
            team = st.slider("Team Size", 1, 100, 10)
            exp = st.selectbox("Team Experience Level", df['Experience_Level'].unique())

        with col2:
            if st.button("Analyze Investment"):
                # User input-ai model-ukku yetha maadhiri maathuradhu
                d_code = le_domain.transform([domain])[0]
                l_code = le_loc.transform([location])[0]
                e_code = le_exp.transform([exp])[0]
                
                prediction = model.predict([[capital, d_code, l_code, e_code, team]])[0]
                
                # UI Result Display
                st.subheader("Result")
                if prediction == "Invest":
                    st.success(f"✅ AI Suggestion: **{prediction}**")
                else:
                    st.warning(f"⚠️ AI Suggestion: **{prediction}**")
    except Exception as e:
        st.error(f"Processing Error: {e}")
else:
    st.error("No data found in 'train.csv'. Please check the file.")
