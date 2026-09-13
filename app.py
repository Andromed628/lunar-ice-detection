import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# ==============================================================================
# 1. LOCAL TOPOGRAPHICAL LUNAR BASEMAP GENERATOR
# ==============================================================================
LOCAL_LUNAR_MAP_PATH = "lunar_polar_topography.png"

def generate_local_lunar_basemap():
    """Generates a high-contrast lunar topographic surface map locally."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    # Create synthetic lunar polar terrain map (Elevation Grid)
    grid_size = 200
    x = np.linspace(-35, 35, grid_size)
    y = np.linspace(-35, 35, grid_size)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # Mathematical representation of craters and polar topographic relief
    R = np.sqrt(X_grid**2 + Y_grid**2)
    Z = np.sin(R / 3) * np.cos(X_grid / 5) + np.sin(Y_grid / 4) * 0.5
    
    # Add key crater impact basins (Shackleton, Cabeus)
    crater_shackleton = np.exp(-((X_grid - 0)**2 + (Y_grid - 0.1)**2) / 4) * -2.5
    crater_cabeus = np.exp(-((X_grid - (-20))**2 + (Y_grid - 15)**2) / 15) * -3.0
    Z += crater_shackleton + crater_cabeus

    # Render topographic elevation colormap
    ax.imshow(Z, extent=[-35, 35, -35, 35], cmap='twilight_shifted', origin='lower')
    
    # Polar Reticle Overlay
    for r in [5, 15, 25, 35]:
        circle = patches.Circle((0, 0), r, fill=False, color='#38bdf8', linestyle=':', alpha=0.4)
        ax.add_patch(circle)

    ax.scatter([0], [0], color='#facc15', s=80, marker='*', label="South Pole (-90°)")
    ax.set_title("LOLA TOPOGRAPHIC RELIEF BASEMAP", color='#f8fafc', fontsize=9, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(LOCAL_LUNAR_MAP_PATH, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=150)
    plt.close()

generate_local_lunar_basemap()

# ==============================================================================
# 2. SYNTHETIC DATA GENERATION & MULTI-MODEL AI TRAINING
# ==============================================================================
np.random.seed(42)
n_samples = 4000

# Surface features
surface_temp_k = np.random.uniform(40, 220, n_samples)      # LRO Diviner Temp (K)
cpr_value = np.random.uniform(0.1, 1.8, n_samples)           # Chandrayaan-2 DFSAR Radar CPR
elevation_m = np.random.uniform(-4000, 2000, n_samples)     # LOLA Elevation / Crater Depth
albedo = np.random.uniform(0.05, 0.45, n_samples)            # LOLA Optical Reflectance
neutron_data = np.random.uniform(10, 100, n_samples)        # Lunar Prospector Neutrons (cps)

# Physics Ground Truth Target
ice_presence = (
    (surface_temp_k < 110) & (cpr_value > 0.85) & (albedo > 0.20) & (neutron_data < 45)
).astype(int)

df = pd.DataFrame({
    'Temperature_K': surface_temp_k,
    'CPR_Value': cpr_value,
    'Elevation_m': elevation_m,
    'Albedo': albedo,
    'Neutron_cps': neutron_data,
    'Ice_Detected': ice_presence
})

X = df[['Temperature_K', 'CPR_Value', 'Elevation_m', 'Albedo', 'Neutron_cps']]
y = df['Ice_Detected']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multi-Model Suite
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

nn_model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42).fit(X_train, y_train)
knn_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

# Preset Craters with Polar Coordinates
PRESET_SITES = {
    "Shackleton Crater (-89.9°S, 0.0°E) [PSR Target]": (-89.9, 0.0),
    "Faustini Crater (-87.3°S, 77.0°E) [Cold Trap]": (-87.3, 77.0),
    "Shoemaker Crater (-88.1°S, 45.0°E) [Deep Shadow]": (-88.1, 45.0),
    "Cabeus Crater (-84.9°S, 35.5°W) [LCROSS Target]": (-84.9, -35.5),
    "Tycho Crater (-43.3°S, 11.2°W) [Sunlit Region]": (-43.3, -11.2),
    "Mare Tranquillitatis (0.7°N, 23.5°E) [Equatorial]": (0.7, 23.5)
}

# ==============================================================================
# 3. HELPER & ANALYTICS FUNCTIONS
# ==============================================================================

# Preset Telemetry & Target Reticle
def ingest_preset_telemetry(preset_name, lat, lon):
    if preset_name != "Custom Coordinates":
        lat, lon = PRESET_SITES[preset_name]

    dist_from_pole = abs(-90.0 - lat)
    
    if dist_from_pole < 3.0:
        temp_k = float(np.round(42.0 + (dist_from_pole * 20.0) + np.random.normal(0, 2), 1))
        cpr = float(np.round(1.25 - (dist_from_pole * 0.1) + np.random.normal(0, 0.05), 2))
        neutron = float(np.round(25.0 + (dist_from_pole * 8.0) + np.random.normal(0, 3), 1))
        albedo_val = 0.28
        elev = -2500.0
    else:
        temp_k = float(np.round(130.0 + (dist_from_pole * 5.0), 1))
        cpr = float(np.round(0.35 + np.random.normal(0, 0.05), 2))
        neutron = float(np.round(85.0 + np.random.normal(0, 4), 1))
        albedo_val = 0.09
        elev = 500.0

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    
    for r in [1, 5, 10, 30]:
        circle = patches.Circle((0, 0), r, fill=False, color='#38bdf8', linestyle=':', alpha=0.5)
        ax.add_patch(circle)
    
    x_pos = (90 + lat) * np.sin(np.radians(lon))
    y_pos = (90 + lat) * np.cos(np.radians(lon))
    
    ax.scatter([0], [0], color='#facc15', s=120, marker='*', label="Lunar South Pole (-90°)")
    ax.scatter([x_pos], [y_pos], color='#00e676', s=180, zorder=5, edgecolors='white', label="Selected Target")
    
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)
    ax.set_title("POLAR TARGET SATELLITE RETICLE", color='#f8fafc', fontsize=10, fontweight='bold')
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.legend(loc='lower left', fontsize=8, facecolor='#0f172a', labelcolor='white', edgecolor='#334155')
    plt.tight_layout()

    telemetry_stream = (
        f"--- SENSOR TELEMETRY STREAM ---\n"
        f"• Target Coordinates : {lat}° N/S, {lon}° E/W\n"
        f"• Surface Temp (LRO Diviner) : {temp_k} K ({-273.15 + temp_k:.1f} °C)\n"
        f"• Radar CPR (Chandrayaan-2 DFSAR) : {cpr}\n"
        f"• Elevation (LOLA Altimeter) : {elev} m\n"
        f"• Reflectance Albedo (LRO LOLA) : {albedo_val}\n"
        f"• Epithermal Neutrons (Lunar Prospector) : {neutron} cps"
    )

    return telemetry_stream, fig, temp_k, cpr, elev, albedo_val, neutron, lat, lon

# Primary Predictor
def predict_lunar_ice(temp, cpr, elev, albedo_val, neutron=25.0):
    input_data = pd.DataFrame([[temp, cpr, elev, albedo_val, neutron]], columns=X.columns)
    prob = model.predict_proba(input_data)[0][1]
    prediction = "ICE DETECTED" if prob > 0.5 else "NO ICE DETECTED"

    status = f"### Status: **{prediction}**\n**Confidence Score:** {prob * 100:.1f}%"

    # Gauge Visualization
    fig, ax = plt.subplots(figsize=(5, 2.5))
    color = "#06b6d4" if prob > 0.5 else "#f97316"
    ax.barh(["Probability"], [prob * 100], color=color, height=0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Water Ice Probability (%)")
    ax.set_title("Subsurface Water Ice Prediction")
    plt.tight_layout()

    # Feature Importance Plot
    fig_feat, ax_feat = plt.subplots(figsize=(5, 2.5))
    fig_feat.patch.set_facecolor('#0f172a')
    ax_feat.set_facecolor('#1e293b')
    features = ['Temperature', 'Radar CPR', 'Elevation', 'Albedo', 'Neutron Flux']
    importances = model.feature_importances_
    ax_feat.barh(features, importances, color='#38bdf8')
    ax_feat.set_xlabel("Relative Feature Weight", color='#94a3b8', fontsize=8)
    ax_feat.set_title("Random Forest Feature Importance", color='#f8fafc', fontsize=9, fontweight='bold')
    ax_feat.tick_params(colors='#f8fafc', labelsize=8)
    plt.tight_layout()

    return status, fig, fig_feat

# Multi-Sensor Analytics Plot
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

# Subsurface Depth Profiler + Thermal Skin Model
def generate_depth_profile(surface_temp):
    depths = np.linspace(0, 2, 50)  # Depth in meters
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

# Rover Landing Assessment & Multi-Model Table
def assess_landing_site(slope_deg, cpr, ice_prob, temp_k=85.0, elev=-2500.0, albedo_val=0.28, neutron_val=25.0):
    safety_score = 100 - (slope_deg * 2.5) + (ice_prob * 30) - (cpr * 10)
    safety_score = max(0, min(100, safety_score))

    if safety_score > 70:
        recommendation = "OPTIMAL SITE: High Resource Potential & Safe Slope"
    elif safety_score > 40:
        recommendation = "MODERATE SITE: Caution Advised During Descent"
    else:
        recommendation = "HAZARDOUS SITE: Excessive Slope or Unfavorable Terrain"

    report = f"### Landing Site Safety Index: {safety_score:.1f}/100\n**Assessment:** {recommendation}"

    # Multi-Model Verification Comparison
    input_data = pd.DataFrame([[temp_k, cpr, elev, albedo_val, neutron_val]], columns=X.columns)
    p_rf = model.predict_proba(input_data)[0][1] * 100
    p_nn = nn_model.predict_proba(input_data)[0][1] * 100
    p_knn = knn_model.predict_proba(input_data)[0][1] * 100

    comparison_df = pd.DataFrame({
        "Algorithm Model": ["Random Forest Classifier", "Artificial Neural Network (MLP)", "K-Nearest Neighbors (KNN)"],
        "Ice Confidence (%)": [f"{p_rf:.1f}%", f"{p_nn:.1f}%", f"{p_knn:.1f}%"],
        "Prediction Status": ["Ice Verified" if p > 50 else "Dry Regolith" for p in [p_rf, p_nn, p_knn]]
    })

    return report, comparison_df

# Interactive Manual Physics Simulation
def run_manual_simulation(temp_sim, cpr_sim, neutron_sim, albedo_sim):
    input_data = pd.DataFrame([[temp_sim, cpr_sim, -2500.0, albedo_sim, neutron_sim]], columns=X.columns)
    prob = model.predict_proba(input_data)[0][1] * 100
    
    status = "VERIFIED DEPOSIT" if prob > 70 else ("TRACE FROST" if prob > 35 else "DRY REGOLITH")
    
    res = (
        f"--- MANUAL OVERRIDE AI PREDICTION ---\n"
        f"• Random Forest Confidence Score : {prob:.1f}%\n"
        f"• Classified Surface Condition   : {status}\n"
        f"• Physical Compatibility Check   : " + 
        ("High (Consistent with cold trap volatile retention)" if (temp_sim < 110 and cpr_sim > 0.9) else "Low / Discarded (Thermal sublimation threshold exceeded)")
    )
    return res

# ==============================================================================
# 4. GRADIO DASHBOARD INTERFACE
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft(), title="TechQuest 2026 - Lunar Ice AI") as demo:
    gr.Markdown(
        """
    # 🌙 Lunar Polar Water Ice Detection Dashboard
    ### **TechQuest 2026 Exhibition Project**
    *Multi-sensor remote sensing & machine learning pipeline analyzing permanently shadowed lunar craters.*
    """
    )

    with gr.Tabs():
        # TAB 1: REAL-TIME DETECTOR & TELEMETRY
        with gr.TabItem("🎯 Real-Time Detector"):
            gr.Markdown("### Input Remote Sensing Parameters & Satellite Preset Ingestion")
            with gr.Row():
                with gr.Column():
                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESET_SITES.keys()) + ["Custom Coordinates"],
                        value="Shackleton Crater (-89.9°S, 0.0°E) [PSR Target]",
                        label="Target Crater / Region Preset"
                    )
                    lat_input = gr.Number(value=-89.9, label="Latitude (-90° to 90°)")
                    lon_input = gr.Number(value=0.0, label="Longitude (-180° to 180°)")
                    ingest_btn = gr.Button("📡 Ingest Satellite Preset Telemetry", variant="secondary")

                    temp_input = gr.Slider(40, 220, value=85, label="Surface Temperature (Kelvin)", info="LRO Diviner Radiometer")
                    cpr_input = gr.Slider(0.1, 1.8, value=1.1, label="Radar CPR Value", info="Chandrayaan-2 DFSAR Radar")
                    elev_input = gr.Slider(-4000, 1000, value=-2500, label="Elevation / Crater Depth (m)", info="LOLA Altimeter")
                    albedo_input = gr.Slider(0.05, 0.45, value=0.28, label="Surface Albedo", info="Optical Reflectance")
                    neutron_input = gr.Slider(10, 100, value=25, label="Epithermal Neutrons (cps)", info="Lunar Prospector")

                    predict_btn = gr.Button("Analyze Satellite Data", variant="primary")

                    gr.Markdown("#### Lunar Elevation Relief Basemap")
                    gr.Image(value=LOCAL_LUNAR_MAP_PATH, label="Topographic Basemap", show_label=True)

                with gr.Column():
                    reticle_plot = gr.Plot(label="Polar Target Reticle Overlay")
                    telemetry_out = gr.Textbox(label="Ingested Telemetry Stream", lines=6)
                    output_status = gr.Markdown("Click 'Analyze' to run model...")
                    output_plot = gr.Plot(label="Confidence Meter")
                    feature_plot = gr.Plot(label="Model Decision Weights")

            ingest_btn.click(
                ingest_preset_telemetry,
                inputs=[preset_dropdown, lat_input, lon_input],
                outputs=[telemetry_out, reticle_plot, temp_input, cpr_input, elev_input, albedo_input, neutron_input, lat_input, lon_input]
            )

            predict_btn.click(
                predict_lunar_ice,
                inputs=[temp_input, cpr_input, elev_input, albedo_input, neutron_input],
                outputs=[output_status, output_plot, feature_plot],
            )

        # TAB 2: MULTI-SENSOR ANALYTICS
        with gr.TabItem("📊 Multi-Sensor Analytics"):
            gr.Markdown("### Correlation Between CPR, Temperature, and Ice Signatures")
            scatter_btn = gr.Button("Generate Scatter Plot", variant="primary")
            scatter_plot = gr.Plot()
            scatter_btn.click(plot_scatter, outputs=scatter_plot)

        # TAB 3: SUBSURFACE DEPTH PROFILER
        with gr.TabItem("❄️ Subsurface Depth Profiler"):
            gr.Markdown("### Subsurface Regolith Temperature Gradient & Thermal Skin Depth")
            prof_temp = gr.Slider(40, 180, value=75, label="Surface Skin Temperature (K)")
            prof_btn = gr.Button("Simulate Subsurface Depth Profile", variant="primary")
            prof_plot = gr.Plot()
            prof_btn.click(generate_depth_profile, inputs=prof_temp, outputs=prof_plot)

        # TAB 4: ROVER LANDING SAFETY & SIMULATOR
        with gr.TabItem("🚀 Rover Landing Safety & Simulator"):
            gr.Markdown("### Target Landing Site Viability Evaluator & Stress-Test Override")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Landing Site Evaluation")
                    slope_input = gr.Slider(0, 35, value=12, label="Local Terrain Slope (Degrees)")
                    cpr_land = gr.Slider(0.1, 1.8, value=1.0, label="Local CPR Heterogeneity")
                    ice_prob_land = gr.Slider(0.0, 1.0, value=0.85, label="Predicted Ice Probability")
                    eval_btn = gr.Button("Evaluate Site Risk", variant="primary")
                    landing_output = gr.Markdown()
                    model_table_output = gr.Dataframe(label="Multi-Model Verification Table")

                with gr.Column():
                    gr.Markdown("#### Manual Physics Override Simulator")
                    sim_temp = gr.Slider(30, 300, value=85, label="Surface Temperature (K)")
                    sim_cpr = gr.Slider(0.1, 2.0, value=1.2, label="Radar CPR")
                    sim_neutron = gr.Slider(5, 120, value=25, label="Epithermal Neutrons (cps)")
                    sim_albedo = gr.Slider(0.01, 0.4, value=0.22, label="Surface Albedo")
                    btn_sim = gr.Button("⚡ Run Manual Simulation", variant="secondary")
                    sim_out = gr.Textbox(label="Real-Time Override Evaluation", lines=6)

            eval_btn.click(
                assess_landing_site,
                inputs=[slope_input, cpr_land, ice_prob_land, temp_input, elev_input, albedo_input, neutron_input],
                outputs=[landing_output, model_table_output],
            )

            btn_sim.click(
                run_manual_simulation,
                inputs=[sim_temp, sim_cpr, sim_neutron, sim_albedo],
                outputs=[sim_out]
            )

# ==============================================================================
# 5. NETWORK BINDING FOR RENDER DEPLOYMENT
# ==============================================================================
import os

# Get port assigned by Render, default to 7860 if local
port = int(os.environ.get("PORT", 7860)) 
demo.launch(server_name="0.0.0.0", server_port=port)


