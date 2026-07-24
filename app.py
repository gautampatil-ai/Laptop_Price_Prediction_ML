import os
import pickle
import numpy as np
from flask import Flask, jsonify, render_template_string, request

# ==============================================================================
# FLASK APP INITIALIZATION
# ==============================================================================
app = Flask(__name__)

# Load the Decision Tree model using pickle
MODEL_PATH = "Decision_Tree_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f" SUCCESS: Model loaded from '{MODEL_PATH}'.")
    except Exception as e:
        print(f" ERROR: Failed to load model file. Details: {e}")
else:
    print(
        f" WARNING: '{MODEL_PATH}' missing. Place it in the root directory."
    )

# ==============================================================================
# ENTERPRISE UI TEMPLATE (HTML5 + Glassmorphism CSS3 + Async JavaScript)
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Prediction System | Enterprise Interface</title>
    <!-- Google Fonts & FontAwesome Icons -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #170e2b 100%);
            --glass-card: rgba(15, 23, 42, 0.65);
            --glass-border: rgba(255, 255, 255, 0.1);
            --primary-accent: #6366f1;
            --primary-accent-glow: rgba(99, 102, 241, 0.35);
            --secondary-accent: #ec4899;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --input-bg: rgba(2, 6, 23, 0.6);
            --radius-lg: 24px;
            --radius-md: 12px;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
            overflow-x: hidden;
            position: relative;
        }

        /* Ambient Animated Lighting Orbs */
        .ambient-light {
            position: fixed;
            border-radius: 50%;
            filter: blur(120px);
            z-index: 0;
            pointer-events: none;
            animation: orbFloat 12s infinite alternate ease-in-out;
        }

        .orb-1 {
            width: 450px;
            height: 450px;
            background: var(--primary-accent-glow);
            top: -10%;
            left: -10%;
        }

        .orb-2 {
            width: 400px;
            height: 400px;
            background: rgba(236, 72, 153, 0.25);
            bottom: -10%;
            right: -10%;
            animation-delay: -6s;
        }

        @keyframes orbFloat {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(40px, 40px) scale(1.15); }
        }

        /* Main Container Card */
        .app-container {
            width: 100%;
            max-width: 950px;
            background: var(--glass-card);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            padding: 3.5rem 3rem;
            box-shadow: 0 30px 60px -12px rgba(0, 0, 0, 0.6);
            position: relative;
            z-index: 10;
            animation: appEntry 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        @keyframes appEntry {
            from { opacity: 0; transform: translateY(20px) scale(0.98); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        /* Header Section */
        .app-header {
            text-align: center;
            margin-bottom: 3rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(99, 102, 241, 0.12);
            border: 1px solid rgba(99, 102, 241, 0.25);
            color: #a5b4fc;
            padding: 0.4rem 1.2rem;
            border-radius: 50px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            margin-bottom: 1.2rem;
        }

        .app-header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.6rem;
            letter-spacing: -0.5px;
        }

        .app-header p {
            color: var(--text-secondary);
            font-size: 1rem;
            font-weight: 400;
        }

        /* Input Grid */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.75rem;
            margin-bottom: 2.5rem;
        }

        .input-card {
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }

        .input-card label {
            font-size: 0.875rem;
            font-weight: 600;
            color: #cbd5e1;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .input-card label i {
            color: var(--primary-accent);
        }

        .input-control {
            width: 100%;
            padding: 0.9rem 1.1rem;
            background: var(--input-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .input-control:focus {
            border-color: var(--primary-accent);
            box-shadow: 0 0 0 4px var(--primary-accent-glow);
            background: rgba(2, 6, 23, 0.85);
        }

        .input-control select option {
            background-color: #020617;
            color: #ffffff;
        }

        /* Submit Button */
        .btn-submit {
            width: 100%;
            padding: 1.1rem;
            background: linear-gradient(135deg, var(--primary-accent) 0%, var(--secondary-accent) 100%);
            border: none;
            border-radius: var(--radius-md);
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px -5px rgba(236, 72, 153, 0.45);
            opacity: 0.96;
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        /* Prediction Result Box */
        .result-container {
            margin-top: 2rem;
            padding: 1.75rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(236, 72, 153, 0.05) 100%);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: var(--radius-md);
            text-align: center;
            display: none;
            animation: resultReveal 0.5s ease forwards;
        }

        @keyframes resultReveal {
            from { opacity: 0; transform: translateY(10px) scale(0.97); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        .result-container h4 {
            font-size: 0.85rem;
            color: #a5b4fc;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 0.4rem;
        }

        .result-value {
            font-size: 2.25rem;
            font-weight: 800;
            color: #ffffff;
            text-shadow: 0 0 20px rgba(255, 255, 255, 0.25);
        }

        /* Loading Spinner */
        .spinner {
            display: none;
            width: 22px;
            height: 22px;
            border: 3px solid rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            border-top-color: #ffffff;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Responsive Breakpoints */
        @media (max-width: 640px) {
            .app-container { padding: 2rem 1.5rem; }
            .app-header h1 { font-size: 1.85rem; }
            .form-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

    <div class="ambient-light orb-1"></div>
    <div class="ambient-light orb-2"></div>

    <div class="app-container">
        <div class="app-header">
            <div class="badge">
                <i class="fa-solid fa-microchip"></i> Decision Tree Engine
            </div>
            <h1>AI Inference Hub</h1>
            <p>Provide feature inputs below for accurate real-time classification/regression.</p>
        </div>

        <form id="predictionForm">
            <div class="form-grid">
                <div class="input-card">
                    <label for="age"><i class="fa-solid fa-user"></i> Age</label>
                    <input type="number" id="age" name="Age" class="input-control" placeholder="e.g. 30" min="1" max="120" required>
                </div>

                <div class="input-card">
                    <label for="gender"><i class="fa-solid fa-venus-mars"></i> Gender</label>
                    <select id="gender" name="Gender" class="input-control" required>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>

                <div class="input-card">
                    <label for="region"><i class="fa-solid fa-globe"></i> Region Code</label>
                    <input type="number" id="region" name="Region" class="input-control" placeholder="e.g. 1" required>
                </div>

                <div class="input-card">
                    <label for="occupation"><i class="fa-solid fa-briefcase"></i> Occupation Code</label>
                    <input type="number" id="occupation" name="Occupation" class="input-control" placeholder="e.g. 2" required>
                </div>

                <div class="input-card">
                    <label for="income"><i class="fa-solid fa-wallet"></i> Income</label>
                    <input type="number" id="income" name="Income" class="input-control" placeholder="e.g. 65000" step="any" required>
                </div>
            </div>

            <button type="submit" class="btn-submit" id="submitBtn">
                <span id="btnText"><i class="fa-solid fa-bolt"></i> Generate Prediction</span>
                <div class="spinner" id="btnSpinner"></div>
            </button>
        </form>

        <div class="result-container" id="resultContainer">
            <h4>Model Output Result</h4>
            <div class="result-value" id="resultValue">--</div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const submitBtn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const btnSpinner = document.getElementById('btnSpinner');
            const resultContainer = document.getElementById('resultContainer');
            const resultValue = document.getElementById('resultValue');

            // Set UI Loading state
            btnText.style.display = 'none';
            btnSpinner.style.display = 'block';
            submitBtn.disabled = true;

            const payload = {
                Age: parseFloat(document.getElementById('age').value),
                Gender: parseInt(document.getElementById('gender').value),
                Region: parseFloat(document.getElementById('region').value),
                Occupation: parseFloat(document.getElementById('occupation').value),
                Income: parseFloat(document.getElementById('income').value)
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                setTimeout(() => {
                    btnText.style.display = 'flex';
                    btnSpinner.style.display = 'none';
                    submitBtn.disabled = false;

                    if (data.status === 'success') {
                        resultValue.innerText = data.prediction;
                        resultContainer.style.display = 'block';
                    } else {
                        alert('Error processing prediction: ' + data.error);
                    }
                }, 300);

            } catch (err) {
                btnText.style.display = 'flex';
                btnSpinner.style.display = 'none';
                submitBtn.disabled = false;
                alert('Connection error. Failed to reach model prediction server.');
            }
        });
    </script>
</body>
</html>
"""

# ==============================================================================
# ROUTE HANDLERS
# ==============================================================================
@app.route("/")
def index():
    """Renders the main glassmorphism prediction dashboard."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    """Receives JSON payload, executes model inference via pickle, and returns output."""
    if model is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "Model file missing or uninitialized on server.",
                }
            ),
            500,
        )

    try:
        data = request.get_json()

        # Build feature vector matching model inputs: [Age, Gender, Region, Occupation, Income]
        features = [
            data.get("Age", 0),
            data.get("Gender", 0),
            data.get("Region", 0),
            data.get("Occupation", 0),
            data.get("Income", 0),
        ]

        # Reshape for scikit-learn standard input format
        input_array = np.array(features).reshape(1, -1)

        # Run prediction
        raw_pred = model.predict(input_array)[0]

        # Format output string
        formatted_pred = str(raw_pred).upper()

        return jsonify({"status": "success", "prediction": formatted_pred})

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
