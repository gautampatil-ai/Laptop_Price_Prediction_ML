import os
import pickle
import numpy as np
from flask import Flask, jsonify, render_template_string, request

# Initialize Flask application
app = Flask(__name__)

# Load the trained Decision Tree model via pickle
MODEL_PATH = "Decision_Tree_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Decision Tree model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model file: {e}")
else:
    print(
        f"⚠️ Warning: '{MODEL_PATH}' was not found. Please place the model file in the root directory."
    )

# Embedded HTML, Glassmorphism CSS, Keyframe Animations, & JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Prediction Dashboard</title>
    <!-- Google Fonts & FontAwesome -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.12);
            --accent-glow: #6366f1;
            --accent-pink: #ec4899;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --card-radius: 20px;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
            overflow-x: hidden;
            position: relative;
        }

        /* Ambient Glowing Background Animations */
        .ambient-orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(100px);
            z-index: 0;
            pointer-events: none;
            animation: pulseGlow 10s infinite alternate ease-in-out;
        }

        .orb-1 {
            width: 400px;
            height: 400px;
            background: rgba(99, 102, 241, 0.35);
            top: -50px;
            left: -50px;
        }

        .orb-2 {
            width: 350px;
            height: 350px;
            background: rgba(236, 72, 153, 0.3);
            bottom: -50px;
            right: -50px;
            animation-delay: -5s;
        }

        @keyframes pulseGlow {
            0% { transform: scale(1) translate(0, 0); }
            100% { transform: scale(1.2) translate(30px, 30px); }
        }

        /* Glassmorphism Container */
        .container {
            width: 100%;
            max-width: 900px;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: var(--card-radius);
            padding: 3rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            position: relative;
            z-index: 10;
            animation: slideUpFade 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        @keyframes slideUpFade {
            from {
                opacity: 0;
                transform: translateY(30px) scale(0.98);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        /* Header Styling */
        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .header-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #a5b4fc;
            padding: 0.4rem 1rem;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 1rem;
        }

        /* Form Layout Grid */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .input-group label {
            font-size: 0.9rem;
            font-weight: 600;
            color: #cbd5e1;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .input-group label i {
            color: var(--accent-glow);
        }

        .input-group input, .input-group select {
            width: 100%;
            padding: 0.85rem 1rem;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            color: #ffffff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--accent-glow);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
            background: rgba(15, 23, 42, 0.8);
        }

        .input-group select option {
            background-color: #0f172a;
            color: #ffffff;
        }

        /* Interactive Submit Button */
        .submit-btn {
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, var(--accent-glow) 0%, var(--accent-pink) 100%);
            border: none;
            border-radius: 12px;
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(236, 72, 153, 0.5);
            opacity: 0.95;
        }

        .submit-btn:active {
            transform: translateY(0);
        }

        /* Result Animation Box */
        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 16px;
            text-align: center;
            display: none;
            animation: fadeInScale 0.5s ease forwards;
        }

        @keyframes fadeInScale {
            from {
                opacity: 0;
                transform: scale(0.95);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }

        .result-box h3 {
            font-size: 0.9rem;
            color: #a5b4fc;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 0.5rem;
        }

        .result-box .prediction-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #ffffff;
            text-shadow: 0 0 15px rgba(255, 255, 255, 0.3);
        }

        /* Loading Spinner */
        .spinner {
            display: none;
            width: 22px;
            height: 22px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #ffffff;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Responsive Formatting */
        @media (max-width: 600px) {
            .container { padding: 1.5rem; }
            .header h1 { font-size: 1.8rem; }
            .form-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

    <div class="ambient-orb orb-1"></div>
    <div class="ambient-orb orb-2"></div>

    <div class="container">
        <div class="header">
            <div class="header-badge">
                <i class="fa-solid fa-brain"></i> Machine Learning Intelligence
            </div>
            <h1>Decision Tree Predictor</h1>
            <p>Enter your structured inputs below to execute real-time model inferences.</p>
        </div>

        <form id="predictionForm">
            <div class="form-grid">
                <div class="input-group">
                    <label><i class="fa-solid fa-user"></i> Age</label>
                    <input type="number" id="age" name="Age" placeholder="e.g. 25" min="1" max="100" required>
                </div>

                <div class="input-group">
                    <label><i class="fa-solid fa-venus-mars"></i> Gender</label>
                    <select id="gender" name="Gender" required>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>

                <div class="input-group">
                    <label><i class="fa-solid fa-earth-americas"></i> Region</label>
                    <input type="number" id="region" name="Region" placeholder="Region code (e.g. 0, 1, 2)" required>
                </div>

                <div class="input-group">
                    <label><i class="fa-solid fa-briefcase"></i> Occupation</label>
                    <input type="number" id="occupation" name="Occupation" placeholder="Occupation code" required>
                </div>

                <div class="input-group">
                    <label><i class="fa-solid fa-wallet"></i> Income Level</label>
                    <input type="number" id="income" name="Income" placeholder="e.g. 50000" step="any" required>
                </div>
            </div>

            <button type="submit" class="submit-btn" id="submitBtn">
                <span id="btnText"><i class="fa-solid fa-bolt"></i> Generate Prediction</span>
                <div class="spinner" id="btnSpinner"></div>
            </button>
        </form>

        <div class="result-box" id="resultBox">
            <h3>Prediction Result</h3>
            <div class="prediction-value" id="predictionResult">--</div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const submitBtn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const btnSpinner = document.getElementById('btnSpinner');
            const resultBox = document.getElementById('resultBox');
            const predictionResult = document.getElementById('predictionResult');

            // UI Loading state
            btnText.style.display = 'none';
            btnSpinner.style.display = 'block';
            submitBtn.disabled = true;

            const formData = {
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
                    body: JSON.stringify(formData)
                });

                const data = await response.json();

                // Artificial delay for smooth interaction design
                setTimeout(() => {
                    btnText.style.display = 'flex';
                    btnSpinner.style.display = 'none';
                    submitBtn.disabled = false;

                    if (data.status === 'success') {
                        predictionResult.innerText = data.prediction;
                        resultBox.style.display = 'block';
                    } else {
                        alert('Prediction Failed: ' + data.error);
                    }
                }, 400);

            } catch (error) {
                btnText.style.display = 'flex';
                btnSpinner.style.display = 'none';
                submitBtn.disabled = false;
                alert('Connection Error: Unable to complete prediction request.');
            }
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Render the front-end dynamic single page dashboard."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    """AJAX endpoint for evaluating input features through the loaded Decision Tree model[cite: 1]."""
    if model is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "Model file 'Decision_Tree_model.pkl' is missing on server.",
                }
            ),
            500,
        )

    try:
        data = request.get_json()

        # Extract features according to feature order
        features = [
            data.get("Age", 0),
            data.get("Gender", 0),
            data.get("Region", 0),
            data.get("Occupation", 0),
            data.get("Income", 0),
        ]

        # Reshape input to array for scikit-learn
        input_array = np.array(features).reshape(1, -1)

        # Run inference using the loaded model
        raw_prediction = model.predict(input_array)[0]

        # Format string output
        prediction_output = str(raw_prediction).title()

        return jsonify({"status": "success", "prediction": prediction_output})

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
