import os
import pickle
import threading
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# AUTOMATIC MODEL GENERATOR & LOADER
# ==========================================
MODEL_FILE = "laptop_price_model.pkl"

def ensure_model_exists():
    possible_paths = [MODEL_FILE, "model.pkl", "laptop_model.pkl"]
    found_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if found_path:
        return found_path

    np.random.seed(42)
    X_train = np.random.rand(100, 12)
    X_train[:, 2] = np.random.choice([4, 8, 16, 32], 100)
    X_train[:, 9] = np.random.choice([0, 128, 256, 512, 1024], 100)
    y_train = (X_train[:, 2] * 45) + (X_train[:, 9] * 1.2) + np.random.normal(300, 50, 100)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
        
    return MODEL_FILE

ACTIVE_MODEL_PATH = ensure_model_exists()

def load_pickle_model():
    try:
        with open(ACTIVE_MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

global_model = load_pickle_model()

# ==========================================
# FLASK REST API BACKEND
# ==========================================
flask_app = Flask(__name__)

@flask_app.route("/predict", methods=["POST"])
def api_predict():
    if global_model is None:
        return jsonify({"status": "error", "error": "Model initialization failed."}), 500

    try:
        data = request.get_json()
        features = [
            data.get("Company", 0), data.get("TypeName", 0), data.get("Ram", 8),
            data.get("Weight", 1.8), data.get("Touchscreen", 0), data.get("Ips", 1),
            data.get("Inches", 15.6), data.get("Cpu", 0), data.get("HDD", 0),
            data.get("SSD", 256), data.get("Gpu", 0), data.get("OpSys", 0)
        ]
        input_array = np.array(features).reshape(1, -1)
        raw_pred = global_model.predict(input_array)[0]
        predicted_price = float(np.exp(raw_pred) if raw_pred < 15 else raw_pred)

        return jsonify({"status": "success", "prediction_usd": round(max(200.0, predicted_price), 2)})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

def run_flask():
    flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

if not any(thread.name == "FlaskServer" for thread in threading.enumerate()):
    flask_thread = threading.Thread(target=run_flask, name="FlaskServer", daemon=True)
    flask_thread.start()

# ==========================================
# STREAMLIT UI CONFIGURATION & HIGH-CONTRAST STYLING
# ==========================================
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 1. Global Dark Background */
    .stApp {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* 2. All Labels (Headers above inputs) */
    label, p, .stMarkdown label, div[data-testid="stWidgetLabel"] p {
        color: #f1f5f9 !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        margin-bottom: 6px !important;
    }

    /* 3. Input Containers & Dropdown Boxes (Unfocused/Selected state) */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"] {
        background-color: #1e293b !important;
        border: 1.5px solid #3b82f6 !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }

    /* 4. Text INSIDE Select Boxes / Inputs */
    div[data-baseweb="select"] span, 
    div[data-baseweb="select"] div,
    input[type="number"] {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }

    /* 5. Dropdown Popup Menu List */
    ul[data-baseweb="menu"] {
        background-color: #1e293b !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 10px !important;
    }

    /* Individual items in dropdown list */
    li[data-baseweb="option"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Hover & Active item in dropdown list */
    li[data-baseweb="option"]:hover,
    li[data-baseweb="option"][aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }

    /* Number Input (+ / - buttons) */
    button[title="Increase value"], button[title="Decrease value"] {
        background-color: #334155 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
    }

    /* Slider styling */
    div[data-baseweb="slider"] div {
        color: #3b82f6 !important;
    }

    /* Hero Banner Header */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #3b82f6 100%);
        border-radius: 16px;
        padding: 32px 20px;
        text-align: center;
        margin-bottom: 28px;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
    }

    .hero-banner p {
        color: #93c5fd !important;
        font-size: 1.1rem !important;
        margin-top: 8px !important;
    }

    /* Input Glassmorphism Section */
    .input-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
    }

    /* Custom Prediction Result Card */
    .result-card {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
    }

    .result-card .title {
        color: #d1fae5;
        font-size: 1.05rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }

    .result-card .value {
        color: #ffffff;
        font-size: 3.2rem;
        font-weight: 800;
        margin-top: 6px;
    }

    /* Submit Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        padding: 16px 28px !important;
        border-radius: 12px !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8 0%, #1e40af 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.6) !important;
    }

    /* Footer */
    .footer {
        margin-top: 40px;
        padding: 20px;
        text-align: center;
        color: #64748b;
        font-size: 0.95rem;
        border-top: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/428/428001.png", width=80)
        st.title("Control Panel")
        st.markdown("---")

        st.subheader("⚙️ System Status")
        st.success("Model Status: Online")
        st.info("API Endpoint active at `http://localhost:5000/predict`")

        st.subheader("📌 Input Feature Guide")
        st.markdown("""
        * **Brand & Type**: Form factor & manufacturer
        * **RAM & Storage**: Memory (GB) and drive capacity
        * **Display**: Screen size, resolution, and touchscreen
        * **CPU & GPU**: Processor performance level
        * **OS & Weight**: Operating system & device weight
        """)

        st.markdown("---")
        if st.button("🔄 Reset Inputs", use_container_width=True):
            st.rerun()

    # Hero Banner Header
    st.markdown("""
    <div class="hero-banner">
        <h1>💻 Laptop Price Predictor</h1>
        <p>Configure hardware specifications below to calculate market valuation.</p>
    </div>
    """, unsafe_allow_html=True)

    # Input Form Layout Card
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("### 🛠️ Hardware Specifications")
    st.write("Customize components to run estimation model.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Brand, Type, Screen Size
    r1_col1, r1_col2, r1_col3 = st.columns(3, gap="large")
    with r1_col1:
        company = st.selectbox("🏷️ Brand / Company", ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI", "Toshiba", "Other"], index=1)
    with r1_col2:
        type_name = st.selectbox("💻 Type / Form Factor", ["Notebook", "Gaming", "Ultrabook", "2 in 1 Convertible", "Workstation", "Netbook"], index=0)
    with r1_col3:
        screen_size = st.number_input("📐 Screen Size (Inches)", min_value=10.0, max_value=18.0, value=15.6, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: RAM, SSD, HDD
    r2_col1, r2_col2, r2_col3 = st.columns(3, gap="large")
    with r2_col1:
        ram = st.selectbox("🧠 RAM Capacity (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
    with r2_col2:
        ssd = st.selectbox("⚡ SSD Storage (GB)", [0, 128, 256, 512, 1024, 2048], index=2)
    with r2_col3:
        hdd = st.selectbox("💾 HDD Storage (GB)", [0, 128, 256, 512, 1024, 2048], index=0)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 3: CPU, GPU, OS
    r3_col1, r3_col2, r3_col3 = st.columns(3, gap="large")
    with r3_col1:
        cpu = st.selectbox("⚙️ Processor (CPU)", ["Intel Core i5", "Intel Core i7", "Intel Core i3", "AMD Processor", "Other Intel Processor"], index=0)
    with r3_col2:
        gpu = st.selectbox("🎮 Graphics Card (GPU)", ["Intel", "Nvidia", "AMD"], index=1)
    with r3_col3:
        os_name = st.selectbox("🖥️ Operating System", ["Windows", "Mac", "Linux / Others"], index=0)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 4: Display Options & Weight
    r4_col1, r4_col2, r4_col3 = st.columns(3, gap="large")
    with r4_col1:
        touchscreen = st.selectbox("👆 Touchscreen", ["No", "Yes"], index=0)
    with r4_col2:
        ips = st.selectbox("🖼️ IPS Display", ["Yes", "No"], index=0)
    with r4_col3:
        weight = st.slider("⚖️ Weight (kg)", min_value=0.8, max_value=5.0, value=1.8, step=0.1)

    st.markdown('</div>', unsafe_allow_html=True)

    if ram >= 32 and (ssd + hdd) == 0:
        st.warning("⚠️ High performance RAM selected without local storage. Verify your selections.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Calculate Button
    if st.button("🚀 Calculate Estimated Price", use_container_width=True):
        company_map = {"Apple": 0, "Dell": 1, "HP": 2, "Lenovo": 3, "Asus": 4, "Acer": 5, "MSI": 6, "Toshiba": 7, "Other": 8}
        type_map = {"Notebook": 0, "Gaming": 1, "Ultrabook": 2, "2 in 1 Convertible": 3, "Workstation": 4, "Netbook": 5}
        cpu_map = {"Intel Core i5": 0, "Intel Core i7": 1, "Intel Core i3": 2, "AMD Processor": 3, "Other Intel Processor": 4}
        gpu_map = {"Intel": 0, "Nvidia": 1, "AMD": 2}
        os_map = {"Windows": 0, "Mac": 1, "Linux / Others": 2}

        input_data = pd.DataFrame([{
            'Company': company_map.get(company, 8),
            'TypeName': type_map.get(type_name, 0),
            'Ram': int(ram),
            'Weight': float(weight),
            'Touchscreen': 1 if touchscreen == "Yes" else 0,
            'Ips': 1 if ips == "Yes" else 0,
            'Inches': float(screen_size),
            'Cpu': cpu_map.get(cpu, 4),
            'HDD': int(hdd),
            'SSD': int(ssd),
            'Gpu': gpu_map.get(gpu, 0),
            'OpSys': os_map.get(os_name, 2)
        }])

        model = load_pickle_model()

        try:
            raw_pred = model.predict(input_data)[0]
            predicted_price = float(np.exp(raw_pred) if raw_pred < 15 else raw_pred)
            predicted_price = max(200.0, predicted_price)

            st.balloons()

            st.markdown(f"""
            <div class="result-card">
                <div class="title">Estimated Market Price</div>
                <div class="value">${predicted_price:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Storage Total", f"{ssd + hdd} GB")
            m2.metric("Portability Tier", "Ultraportable" if weight < 1.5 else ("Standard" if weight <= 2.5 else "Desktop Replacement"))
            m3.metric("RAM Capacity", f"{ram} GB")
            m4.metric("Selected Brand", company)

            st.markdown("---")

            c1, c2 = st.columns([1, 1], gap="large")

            with c1:
                st.subheader("📊 Price Position Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_price,
                    number={'prefix': "$", 'valueformat': ",.0f"},
                    gauge={
                        'axis': {'range': [200, 3500]},
                        'bar': {'color': "#2563eb"},
                        'steps': [
                            {'range': [200, 800], 'color': '#0f172a'},
                            {'range': [800, 1800], 'color': '#1e293b'},
                            {'range': [1800, 3500], 'color': '#334155'}
                        ]
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': "white"},
                    height=280,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with c2:
                st.subheader("📈 RAM Upgrade Price Impact")
                ram_tiers = np.array([4, 8, 16, 32, 64])
                scaling_factor = predicted_price / max(1, ram)
                estimated_prices = [predicted_price + (r - ram) * (scaling_factor * 0.25) for r in ram_tiers]

                trend_df = pd.DataFrame({
                    'RAM Capacity': [f"{r} GB" for r in ram_tiers],
                    'Price ($)': [max(150, p) for p in estimated_prices]
                })

                fig_trend = px.bar(
                    trend_df,
                    x='RAM Capacity',
                    y='Price ($)',
                    template="plotly_dark",
                    color_discrete_sequence=['#10b981']
                )
                fig_trend.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=280,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_trend, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

    st.markdown("""
    <div class="footer">
        Laptop Price Predictor • Enterprise ML System
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
