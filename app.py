import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- 1. SYNTHETIC DATA GENERATION (LRO Diviner & Chandrayaan-2 DFSAR Analog) ---
np.random.seed(42)
n_samples = 1200

# Surface features
surface_temp_k = np.random.uniform(40, 220, n_samples)  # Temperature in Kelvin
cpr_value = np.random.uniform(0.1, 1.8, n_samples)  # Circular Polarization Ratio
elevation_m = np.random.uniform(-4000, 2000, n_samples)  # Crater depth/elevation
albedo = np.random.uniform(0.05, 0.45, n_samples)  # Reflectance

# Physics-based ice existence target (Cold traps + high CPR + high albedo)
ice_presence = (
    (surface_temp_k < 110) & (cpr_value > 0.85) & (albedo > 0.20)
).astype(int)

df = pd.DataFrame(
    {
        "Temperature_K": surface_temp_k,
        "CPR_Value": cpr_value,
        "Elevation_m": elevation_m,
        "Albedo": albedo,
        "Ice_Detected": ice_presence,
    }
)

# Train baseline ML Model
X = df[["Temperature_K", "CPR_Value", "Elevation_m", "Albedo"]]
y = df["Ice_Detected"]
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# --- TAB 1: MODEL PREDICTION LOGIC ---
def predict_lunar_ice(temp, cpr, elev, albedo_val):
    input_data = pd.DataFrame([[temp, cpr, elev, albedo_val]], columns=X.columns)
    prob = model.predict_proba(input_data)[0][1]
    prediction = "ICE DETECTED" if prob > 0.5 else "NO ICE DETECTED"

    status = f"### Status: **{prediction}**\n**Confidence Score:** {prob * 100:.1f}%"

    # Gauge Visualization
    fig, ax = plt.subplots(figsize=(5, 2.5))
    color = "skyblue" if prob > 0.5 else "orange"
    ax.barh(["Probability"], [prob * 100], color=color, height=0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Water Ice Probability (%)")
    ax.set_title("Subsurface Water Ice Prediction")
    plt.tight_layout()

    return status, fig


# --- TAB 2: DATASET & GEOGRAPHIC ANALYTICS ---
def plot_scatter():
    fig, ax = plt.subplots(figsize=(7, 4))
    scatter = ax.scatter(
        df["Temperature_K"],
        df["CPR_Value"],
        c=df["Ice_Detected"],
        cmap="coolwarm",
        alpha=0.7,
        edgecolors="k",
        s=30,
    )
    ax.set_xlabel("Surface Temperature (K)")
    ax.set_ylabel("Circular Polarization Ratio (CPR)")
    ax.set_title("LRO Diviner vs Chandrayaan-2 Radar Signature Scatter Plot")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("0 = Regolith, 1 = Water Ice")
    plt.tight_layout()
    return fig


# --- TAB 3: SUBSURFACE DEPTH PROFILER ---
def generate_depth_profile(surface_temp):
    depths = np.linspace(0, 2, 50)  # Depth in meters
    # Simplified lunar thermal skin depth model
    temperatures = surface_temp + (100 - surface_temp) * (1 - np.exp(-depths / 0.4))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(temperatures, depths, color="cyan", linewidth=2)
    ax.axvline(
        x=110, color="red", linestyle="--", label="Ice Stability Threshold (110 K)"
    )
    ax.invert_yaxis()
    ax.set_xlabel("Subsurface Temperature (K)")
    ax.set_ylabel("Depth Below Regolith (m)")
    ax.set_title("Lunar South Pole Thermal Depth Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# --- TAB 4: SIMULATION & LANDING SAFETY ---
def assess_landing_site(slope_deg, cpr, ice_prob):
    safety_score = 100 - (slope_deg * 2.5) + (ice_prob * 30) - (cpr * 10)
    safety_score = max(0, min(100, safety_score))

    if safety_score > 70:
        recommendation = "OPTIMAL SITE: High Resource Potential & Safe Slope"
    elif safety_score > 40:
        recommendation = "MODERATE SITE: Caution Advised During Descent"
    else:
        recommendation = (
            "HAZARDOUS SITE: Excessive Slope or Unfavorable Terrain"
        )

    return f"### Landing Site Safety Index: {safety_score:.1f}/100\n**Assessment:** {recommendation}"


# --- GRADIO INTERFACE LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft(), title="TechQuest 2026 - Lunar Ice AI") as demo:
    gr.Markdown(
        """
    # 🌙 Lunar Polar Water Ice Detection Dashboard
    ### **TechQuest 2026 Exhibition Project**
    *Multi-sensor remote sensing & machine learning pipeline analyzing permanently shadowed lunar craters.*
    """
    )

    with gr.Tabs():
        # TAB 1
        with gr.TabItem("🎯 Real-Time Detector"):
            gr.Markdown("### Input Remote Sensing Parameters")
            with gr.Row():
                with gr.Column():
                    temp_input = gr.Slider(
                        40,
                        220,
                        value=85,
                        label="Surface Temperature (Kelvin)",
                        info="LRO Diviner Thermal Radiometer",
                    )
                    cpr_input = gr.Slider(
                        0.1,
                        1.8,
                        value=1.1,
                        label="Radar CPR Value",
                        info="Chandrayaan-2 DFSAR Radar",
                    )
                    elev_input = gr.Slider(
                        -4000,
                        1000,
                        value=-2500,
                        label="Elevation / Crater Depth (m)",
                        info="LOLA Altimeter",
                    )
                    albedo_input = gr.Slider(
                        0.05,
                        0.45,
                        value=0.28,
                        label="Surface Albedo",
                        info="Optical Reflectance",
                    )
                    predict_btn = gr.Button("Analyze Satellite Data", variant="primary")

                with gr.Column():
                    output_status = gr.Markdown("Click 'Analyze' to run model...")
                    output_plot = gr.Plot()

            predict_btn.click(
                predict_lunar_ice,
                inputs=[temp_input, cpr_input, elev_input, albedo_input],
                outputs=[output_status, output_plot],
            )

        # TAB 2
        with gr.TabItem("📊 Multi-Sensor Analytics"):
            gr.Markdown("### Correlation Between CPR, Temperature, and Ice Signatures")
            scatter_btn = gr.Button("Generate Scatter Plot")
            scatter_plot = gr.Plot()
            scatter_btn.click(plot_scatter, outputs=scatter_plot)

        # TAB 3
        with gr.TabItem("❄️ Subsurface Depth Profiler"):
            gr.Markdown("### Subsurface Regolith Temperature Gradient")
            prof_temp = gr.Slider(
                40, 180, value=75, label="Surface Skin Temperature (K)"
            )
            prof_btn = gr.Button("Simulate Subsurface Depth Profile")
            prof_plot = gr.Plot()
            prof_btn.click(
                generate_depth_profile, inputs=prof_temp, outputs=prof_plot
            )

        # TAB 4
        with gr.TabItem("🚀 Rover Landing Safety"):
            gr.Markdown("### Target Landing Site Viability Evaluator")
            slope_input = gr.Slider(
                0, 35, value=12, label="Local Terrain Slope (Degrees)"
            )
            cpr_land = gr.Slider(
                0.1, 1.8, value=1.0, label="Local CPR Heterogeneity"
            )
            ice_prob_land = gr.Slider(
                0.0, 1.0, value=0.85, label="Predicted Ice Probability"
            )
            eval_btn = gr.Button("Evaluate Site Risk", variant="primary")
            landing_output = gr.Markdown()

            eval_btn.click(
                assess_landing_site,
                inputs=[slope_input, cpr_land, ice_prob_land],
                outputs=landing_output,
            )
import os

# Get port assigned by Render, default to 7860 if local
port = int(os.environ.get("PORT", 7860)) 
demo.launch(server_name="0.0.0.0", server_port=port)


