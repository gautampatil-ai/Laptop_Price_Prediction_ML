import os
import pickle
import threading
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, jsonify, request

# ==========================================
# FLASK BACKEND SETUP
# ==========================================
flask_app = Flask(__name__)

# Model loading logic (shared between Flask and Streamlit)
MODEL_PATHS = ["laptop_price_model.pkl", "model.pkl", "laptop_model.pkl"]

def get_model_path():
    return next((p for p in MODEL_PATHS if os.path.exists(p)), None)

def load_pickle_model():
    path = get_model_path()
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

# Load model globally for API
global_model = load_pickle_model()

@flask_app.route("/predict", methods=["POST"])
def api_predict():
    if global_model is None:
        return jsonify({
            "status": "error",
            "error": "Model file missing or uninitialized on server."
        }), 500

    try:
        data = request.get_json()
        
        # Expected features vector matching model training:
        # [Brand, TypeName, Ram, Weight, Touchscreen, Ips, Inches, Cpu, HDD, SSD, Gpu, OpSys]
        features = [
            data.get("Company", 0),
            data.get("TypeName", 0),
            data.get("Ram", 8),
            data.get("Weight", 1.8),
            data.get("Touchscreen", 0),
            data.get("Ips", 1),
            data.get("Inches", 15.6),
            data.get("Cpu", 0),
            data.get("HDD", 0),
            data.get("SSD", 256),
            data.get("Gpu", 0),
            data.get("OpSys", 0)
        ]

        input_array = np.array(features).reshape(1, -1)
        raw_pred = global_model.predict(input_array)[0]
        
        # Reverse log-transformation if used during training
        predicted_price = float(np.exp(raw_pred) if raw_pred < 15 else raw_pred)

        return jsonify({
            "status": "success",
            "prediction_usd": round(predicted_price, 2)
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

def run_flask():
    flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# Start background Flask server
if not any(thread.name == "FlaskServer" for thread in threading.enumerate()):
    flask_thread = threading.Thread(target=run_flask, name="FlaskServer", daemon=True)
    flask_thread.start()


# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #170e2b 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, sans-serif;
    }

    .hero-banner {
        background: linear-gradient(90deg, #4f46e5 0%, #6366f1 50%, #ec4899 100%);
        border-radius: 16px;
        padding: 32px 20px;
        text-align: center;
        margin-bottom: 28px;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
    }

    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }

    .hero-banner p {
        color: #e2e8f0;
        font-size: 1.1rem;
        margin-top: 8px;
    }

    .input-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
    }

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
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }

    .result-card .value {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        margin-top: 6px;
    }

    .stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #4f46e5 100%);
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 14px 28px;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #4f46e5 0%, #3730a3 100%);
        transform: translateY(-2px);
    }

    .footer {
        margin-top: 40px;
        padding: 20px;
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ml_model():
    model_path = get_model_path()
    if not model_path:
        return None, "No trained laptop model file (`.pkl`) found in project root."
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model, None
    except Exception as e:
        return None, f"Error loading model file: {str(e)}"


# ==========================================
# MAIN STREAMLIT APPLICATION
# ==========================================
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/428/428001.png", width=85)
        st.title("Control Panel")
        st.markdown("---")

        st.subheader("⚙️ System Status")
        st.success("Model Status: Online")
        st.info("API Endpoint active at `http://localhost:5000/predict`")

        st.subheader("📌 Key Hardware Features")
        st.markdown("""
        * **Brand & Type**: Manufacturer & form factor
        * **RAM & Storage**: Memory and drive capacity (SSD/HDD)
        * **Display**: Screen size, resolution, & touch capability
        * **CPU & GPU**: Processing and graphics power
        * **OS & Weight**: Operating system and portability
        """)

        st.markdown("---")
        if st.button("🔄 Reset Inputs", use_container_width=True):
            st.rerun()

    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <h1>💻 Laptop Price Predictor</h1>
        <p>Configure hardware specifications below to receive an instant market price estimate.</p>
    </div>
    """, unsafe_allow_html=True)

    model, error_msg = load_ml_model()

    if error_msg:
        st.error(f"⚠️ {error_msg}")
        st.info("Ensure your trained model is saved as `laptop_price_model.pkl` or `model.pkl` in the root folder.")
        st.stop()

    # Form Inputs Card
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("🛠 Hardware Specifications")
    st.write("Customize components to estimate the laptop market valuation.")
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

    # Row 2: RAM, Storage
    r2_col1, r2_col2, r2_col3 = st.columns(3, gap="large")

    with r2_col1:
        ram = st.selectbox("🧠 RAM Capacity", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
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

    # Row 4: Display Extras & Weight
    r4_col1, r4_col2, r4_col3 = st.columns(3, gap="large")

    with r4_col1:
        touchscreen = st.selectbox("👆 Touchscreen", ["No", "Yes"], index=0)
    with r4_col2:
        ips = st.selectbox("🖼️ IPS Display", ["Yes", "No"], index=0)
    with r4_col3:
        weight = st.slider("⚖️ Weight (kg)", min_value=0.8, max_value=5.0, value=1.8, step=0.1)

    st.markdown('</div>', unsafe_allow_html=True)

    # Validation Warning
    if ram >= 32 and (ssd + hdd) == 0:
        st.warning("⚠️ High-spec RAM detected without any storage allocated. Please check your inputs.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Calculate Button
    if st.button("🚀 Calculate Estimated Price", use_container_width=True):
        # Mappings matching feature encoding
        company_map = {"Apple": 0, "Dell": 1, "HP": 2, "Lenovo": 3, "Asus": 4, "Acer": 5, "MSI": 6, "Toshiba": 7, "Other": 8}
        type_map = {"Notebook": 0, "Gaming": 1, "Ultrabook": 2, "2 in 1 Convertible": 3, "Workstation": 4, "Netbook": 5}
        cpu_map = {"Intel Core i5": 0, "Intel Core i7": 1, "Intel Core i3": 2, "AMD Processor": 3, "Other Intel Processor": 4}
        gpu_map = {"Intel": 0, "Nvidia": 1, "AMD": 2}
        os_map = {"Windows": 0, "Mac": 1, "Linux / Others": 2}

        # Build Dataframe matching model inputs
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

        try:
            raw_pred = model.predict(input_data)[0]
            
            # Handle log-transformed target variables (common in price regression)
            predicted_price = float(np.exp(raw_pred) if raw_pred < 15 else raw_pred)
            predicted_price = max(150.0, predicted_price)

            st.balloons()

            # Price Card Output
            st.markdown(f"""
            <div class="result-card">
                <div class="title">Estimated Market Price</div>
                <div class="value">${predicted_price:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            # Key Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Storage Total", f"{ssd + hdd} GB")
            m2.metric("Portability Tier", "Ultraportable" if weight < 1.5 else ("Standard" if weight <= 2.5 else "Desktop Replacement"))
            m3.metric("RAM Capacity", f"{ram} GB")
            m4.metric("Selected Brand", company)

            st.markdown("---")

            # Visual Charts
            c1, c2 = st.columns([1, 1], gap="large")

            with c1:
                st.subheader("📊 Price Position Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_price,
                    number={'prefix': "$", 'valueformat': ",.0f"},
                    gauge={
                        'axis': {'range': [200, 3500]},
                        'bar': {'color': "#6366f1"},
                        'steps': [
                            {'range': [200, 800], 'color': '#1e293b'},
                            {'range': [800, 1800], 'color': '#334155'},
                            {'range': [1800, 3500], 'color': '#475569'}
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
                st.subheader("📈 RAM Scaling vs. Price")
                ram_tiers = np.array([4, 8, 16, 32, 64])
                scaling_factor = predicted_price / max(1, ram)
                estimated_prices = [predicted_price + (r - ram) * (scaling_factor * 0.2) for r in ram_tiers]

                trend_df = pd.DataFrame({
                    'RAM (GB)': [f"{r}GB" for r in ram_tiers],
                    'Price ($)': estimated_prices
                })

                fig_trend = px.bar(
                    trend_df,
                    x='RAM (GB)',
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

    # Footer
    st.markdown("""
    <div class="footer">
        Laptop Price Predictor • Powered by Streamlit, Scikit-Learn & Flask REST API
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
