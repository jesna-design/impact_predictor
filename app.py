import gradio as gr
import math
import json
import os
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

# --- Global Constants ---
J_TO_MT = 2.39e-20
MT_TO_J = 4.184e15
MT_TO_KT = 1000
P_ATM = 101.325
G_EARTH = 9.81

DENSITIES = {
    "C-type (Carbonaceous)": 1300,
    "S-type (Stony)": 2700,
    "M-type (Metallic)": 5000,
    "Rocky": 3000
}
DEFAULT_DENSITY = DENSITIES["Rocky"]
JOULES_PER_MEGATON = 4.184e15
DAMAGE_SCALING_CONSTANT_KM = 3.5

DUST_EJECTA_PER_J = 1e-12
SOOT_EJECTA_PER_J = 1e-14
BASELINE_NPP = 1.0
NPP_REDUCTION_PER_DEGREE_C = 0.05
NPP_REDUCTION_PER_DIMMING = 0.5

DAMAGE_CRITERIA_KPA = {
    "Complete Destruction (CRATER)": 70.0,
    "Severe Structural Damage": 35.0,
    "Moderate Structural Damage": 14.0,
    "Heavy Glass/Minor Structural": 7.0,
}

SEISMIC_EFFICIENCY = 1e-4
K1 = 0.82
NU = 0.22

RHO_WATER = 1000
K_CAVITY = 0.5
K_WAVE = 0.25
DEFAULT_WATER_DEPTH_M = 50.0

# --- Data Loading and Utility Functions (ROBUST VERSION) ---

def load_data(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Loads and flattens asteroid data from multiple JSON files, using robust error handling."""
    all_asteroids = []

    for path in file_paths:
        if not os.path.exists(path):
            continue

        try:
            with open(path, 'r') as f:
                # 1. Check if file is empty before loading JSON
                f.seek(0)
                first_char = f.read(1)
                if not first_char:
                    continue
                f.seek(0)

                # 2. Attempt to load the JSON data
                data = json.load(f)

            neo_by_date = data.get('near_earth_objects', {})

            if neo_by_date:
                for date_data in neo_by_date.values():
                    all_asteroids.extend(date_data)

        except json.JSONDecodeError as e:
            # Catches files that contain non-JSON content
            print(f" WARNING: Skipping file {path}. JSON Decode Error: {e}")
            continue
        except Exception as e:
            # Catches other potential read/key errors
            print(f" WARNING: Skipping file {path}. General Error during parsing: {e}")
            continue

    if not all_asteroids:
        # Fallback to mock data only if NO files contributed any valid asteroid data
        print(" WARNING: No valid data loaded. Using Apophis mock data for initialization.")
        all_asteroids.append({
            'name': '99942 Apophis',
            'neo_reference_id': '2000042',
            'is_potentially_hazardous_asteroid': True,
            'estimated_diameter': {'meters': {'estimated_diameter_min': '320.0', 'estimated_diameter_max': '400.0'}},
            'close_approach_data': [{'relative_velocity': {'kilometers_per_second': '7.42',}, 'miss_distance': {'kilometers': '31000.0'}}]
        })
    return all_asteroids

def find_asteroid_data(asteroid_name: str, asteroid_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Finds the asteroid by name or ID."""
    normalized_name = asteroid_name.strip().upper().replace('(', '').replace(')', '').replace('-', ' ')
    for asteroid in asteroid_list:
        current_name = asteroid.get('name', '').strip().upper().replace('(', '').replace(')', '').replace('-', ' ')
        if normalized_name == current_name or normalized_name == asteroid.get('neo_reference_id') or normalized_name in current_name:
            return asteroid
    return None

def get_top_n_nearest_asteroids(asteroid_list: List[Dict[str, Any]], n: int = 5) -> List[Tuple[str, float]]:
    """Analyzes the full dataset to find the top N asteroids with the smallest distance."""
    asteroid_distances: Dict[str, float] = {}
    for asteroid in asteroid_list:
        name = asteroid.get('name', 'UNKNOWN')
        if not asteroid.get('close_approach_data'):
            continue
        min_distance_km = float('inf')
        for approach in asteroid['close_approach_data']:
            try:
                distance_km = float(approach['miss_distance']['kilometers'])
                if distance_km < min_distance_km:
                    min_distance_km = distance_km
            except (KeyError, ValueError):
                continue
        if min_distance_km != float('inf'):
            if name not in asteroid_distances or min_distance_km < asteroid_distances[name]:
                asteroid_distances[name] = min_distance_km
    sorted_asteroids = sorted(asteroid_distances.items(), key=lambda item: item[1])
    return sorted_asteroids[:n]

def calculate_destruction_radius(energy_megatons: float) -> Tuple[float, float]:
    """Estimates the radius and area of significant destruction based on impact energy."""
    if energy_megatons < 1e-6:
        return 0.0, 0.0
    radius_km = DAMAGE_SCALING_CONSTANT_KM * (energy_megatons ** (1/3))
    area_sq_km = math.pi * (radius_km ** 2)
    return radius_km, area_sq_km

def calculate_impact_energy(asteroid_data: Dict[str, Any], density: float) -> Dict[str, Any]:
    """Calculates the approximate impact energy (kinetic energy)."""
    if not asteroid_data or not asteroid_data.get('close_approach_data'):
        return {"error": "Missing critical close approach data."}
    try:
        estimated_diameter = asteroid_data['estimated_diameter']['meters']
        avg_diameter = (float(estimated_diameter['estimated_diameter_min']) + float(estimated_diameter['estimated_diameter_max'])) / 2
        avg_radius = avg_diameter / 2
        # Use the LAST close approach data entry as the basis for the current impact scenario
        close_approach = asteroid_data['close_approach_data'][-1]
        relative_velocity_km_s = float(close_approach['relative_velocity']['kilometers_per_second'])
        V_m_s = relative_velocity_km_s * 1000
        mass_kg = density * (4/3) * math.pi * (avg_radius ** 3)
        kinetic_energy_joules = 0.5 * mass_kg * (V_m_s ** 2)
        energy_megatons = kinetic_energy_joules / JOULES_PER_MEGATON
        radius_km, area_sq_km = calculate_destruction_radius(energy_megatons)
        density_label = next((k for k, v in DENSITIES.items() if v == density), "Custom")

        return {
            "name": asteroid_data.get('name', 'N/A'),
            "average_diameter_m": avg_diameter,
            "relative_velocity_km_s": relative_velocity_km_s,
            "assumed_density_kg_m3": density,
            "density_label": density_label,
            "calculated_mass_kg": mass_kg,
            "kinetic_energy_joules": kinetic_energy_joules,
            "impact_energy_megatons_tnt": energy_megatons,
            "is_potentially_hazardous": asteroid_data.get('is_potentially_hazardous_asteroid', False),
            "destruction_radius_km": radius_km,
            "destruction_area_sq_km": area_sq_km
        }
    except Exception as e:
        return {"error": f"An unexpected error occurred during calculation: {e}"}

def get_preparedness_guidance(energy_megatons: float) -> str:
    """Provides preparedness advice based on calculated energy."""
    if energy_megatons > 100:
        return "GLOBAL CATASTROPHE RISK. The impact is capable of massive ecological and climate-changing events. Immediate global coordination and evacuation planning are essential. (Follow PDCO/FEMA Level 5 Protocol)."
    elif energy_megatons > 10:
        return "MAJOR REGIONAL DISASTER. This impact could level a large state or small country. Evacuation orders must be followed immediately. Prepare deep shelter against seismic shock and atmospheric effects. (Follow PDCO/FEMA Level 4 Protocol)."
    elif energy_megatons > 1:
        return "TUNGUSKA SCALE.Capable of leveling a major metropolitan area or causing significant tsunamis. Follow official shelter-in-place or evacuation orders. **Prepare an Emergency Kit (Water, Food, Radio).**"
    elif energy_megatons > 0.01:
        return "LOCALIZED HAZARD. Significant airburst event (larger than Chelyabinsk). May cause widespread glass breakage and minor structural damage over a large area. Stay indoors away from windows."
    else:
        return "MINIMAL RISK. Likely a small airburst event causing localized damage (Chelyabinsk scale or smaller). Stay informed via official sources and follow basic disaster preparedness (FEMA guidance)."


# --- Land Cover Impact Models ---

def model_vegetation(E_J: float, E_MT: float) -> str:
    """Predicts the impact on forests, grasslands, or crops."""

    # --- Phase 1: Immediate Local Effects ---
    def calculate_local_destruction(E_MT):
        R_thermal_km = 0.5 * (E_MT)**(1/3.0)
        R_blast_km = 0.7 * (E_MT)**(1/3.0)
        destruction_radius_km = max(R_thermal_km, R_blast_km)
        destruction_area_sqkm = np.pi * destruction_radius_km**2
        return destruction_radius_km, destruction_area_sqkm

    # --- Phase 2: Long-Term Global Effects (Climate Change) ---
    def estimate_global_effects(E_J):
        dust_mass_kg = E_J * DUST_EJECTA_PER_J
        soot_mass_kg = E_J * SOOT_EJECTA_PER_J
        solar_dimming_factor = min(1.0, (dust_mass_kg + soot_mass_kg) / 1e12)
        global_temp_drop_C = 5.0 * solar_dimming_factor
        return solar_dimming_factor, global_temp_drop_C

    # --- Phase 3: Vegetation Impact Prediction ---
    def predict_vegetation_npp(solar_dimming, temp_drop, baseline_npp):
        npp_temp_reduction = temp_drop * NPP_REDUCTION_PER_DEGREE_C
        npp_dimming_reduction = NPP_REDUCTION_PER_DIMMING * (1 - np.exp(-5 * solar_dimming))
        total_npp_reduction = min(1.0, npp_temp_reduction + npp_dimming_reduction)
        final_npp = baseline_npp * (1.0 - total_npp_reduction)
        return final_npp, total_npp_reduction

    # Run the Simulation
    destruction_radius, destruction_area = calculate_local_destruction(E_MT)
    solar_dimming, temp_drop = estimate_global_effects(E_J)
    final_npp, total_reduction = predict_vegetation_npp(solar_dimming, temp_drop, BASELINE_NPP)

    return f"""
    <h2>Vegetation Impact Analysis </h2>

    <h3>Immediate Local Effects (Near Ground Zero)</h3>
    <p>
      Radius of Complete Destruction (Blast/Thermal):{destruction_radius:.2f} km<br>
      Area of Complete Destruction: {destruction_area:.2f} sq km<br>
      (Vegetation in this area is instantly vaporized, incinerated, or flattened)
    </p>

    <h3>Long-Term Global Effects (Due to atmospheric dust/soot)</h3>
    <p>
      Estimated Peak Global Cooling:{temp_drop:.2f} °C<br>
      Estimated Peak Solar Dimming:{solar_dimming*100:.2f} % reduction in sunlight<br>
      (These effects can last for months to years, causing an 'impact winter' and global crop failure.)
    </p>

    <h3>Global Vegetation Impact</h3>
    <p>
      Estimated Peak Global NPP Reduction (Ecosystem Collapse):{total_reduction*100:.2f} %<br>
      Predicted Final Normalized Global NPP:{final_npp:.2f}<br>
      (NPP drop indicates widespread agricultural failure and ecosystem collapse)
    </p>
    <div style="color: grey; font-size: 0.9em; margin-top: 15px;">
      Model Reference: Simplified model based on ecological impact studies of large scale events.
    </div>
    """


def model_artificial_surface(E_J: float, E_MT: float) -> str:
    """Predicts the impact on urban areas, buildings, and roads."""
    target_distance_km = 1.0
    def calculate_blast_overpressure(E_MT, distance_km):
        if distance_km == 0: return np.inf
        E_kt = E_MT * MT_TO_KT
        distance_m = distance_km * 1000
        Z = distance_m / (E_kt**(1/3))
        if Z < 100:
            P_op = 1000 * (1.3 / Z) + 50
        elif Z < 1000:
            P_op = P_ATM * (20 / Z**2 + 1 / Z)
        else:
            P_op = 10 * (E_kt**(1/3) / distance_km)
        return P_op


    def get_damage_description(P_op_kPa):
        if P_op_kPa >= DAMAGE_CRITERIA_KPA["Complete Destruction (CRATER)"]:
            return "Complete Destruction (CRATER)", "Reinforced concrete/steel structures demolished. Roads obliterated."
        elif P_op_kPa >= DAMAGE_CRITERIA_KPA["Severe Structural Damage"]:
            return "Severe Structural Damage", "Residential structures collapse. Roads buckle and crack severely."
        elif P_op_kPa >= DAMAGE_CRITERIA_KPA["Moderate Structural Damage"]:
            return "Moderate Structural Damage", "Walls severely damaged, doors/windows blown out."
        elif P_op_kPa >= DAMAGE_CRITERIA_KPA["Heavy Glass/Minor Structural"]:
            return "Heavy Glass/Minor Structural", "All window glass shatters (major hazard). Minor facade damage."
        else:
            return "Negligible Damage", "No significant damage."

    p_op = calculate_blast_overpressure(E_MT, target_distance_km)
    damage_level, damage_description = get_damage_description(p_op)

    glass_shatter_approx_km = 0.9 * (E_MT)**(1/3) * (P_ATM / 7.0) * 10

    return f"""
    <h2>Artificial Surface (Urban) Impact Analysis </h2>

    <h3>Immediate Structural Effects (Blast Wave)</h3>
    <p>
      Peak Blast Overpressure (at {target_distance_km:.1f} km):{p_op:.2f} kPa<br>
      Predicted Damage Level:{damage_level}<br>
      Predicted Effect on Buildings & Roads:{damage_description}
    </p>

    <h3>Hazard Radius Estimates (Approximate)</h3>
    <p>
      Maximum Radius for Severe Structural Damage (35 kPa):{0.5 * (E_MT)**(1/3) :.2f} km<br>
      Maximum Range for Glass Shatter (7 kPa):{glass_shatter_approx_km:.2f} km<br>
      (The blast wave is the primary destruction mechanism for built areas.)
    </p>
    <div style="color: grey; font-size: 0.9em; margin-top: 15px;">
      Model Reference: Simplified scaling laws based on nuclear/large explosion phenomenology.
    </div>
    """


def model_bare_ground(E_J: float, E_MT: float) -> str:
    """Predicts the impact on soil, sand, or rock (cratering and seismicity)."""
    target_density_kg_m3 = 2500

    def calculate_crater_dimensions(E_J, rho_t, g):
        if E_J <= 0: return 0.0, 0.0, 0.0
        D_meters = K1 * (rho_t * g / E_J)**(NU / (2 * NU - 3)) * (E_J / rho_t)**(1/3)
        D_crater_km = D_meters / 1000
        d_crater_km = D_crater_km / 6.0
        volume_m3 = (np.pi / 4) * D_meters**2 * d_crater_km * 1000
        return D_crater_km, d_crater_km, volume_m3

    def calculate_ejecta_thickness(D_crater_km, distance_km):
        if D_crater_km <= 0 or distance_km <= D_crater_km / 2: return 0.0
        R_crater_km = D_crater_km / 2.0
        normalized_distance = distance_km / R_crater_km
        h_ejecta_m = R_crater_km * 0.003 * (normalized_distance)**(-3.0)
        return h_ejecta_m

    def calculate_seismic_shaking(E_J, seismic_efficiency):
        E_seismic_J = E_J * seismic_efficiency
        if E_seismic_J <= 0: return 0.0
        Mw = 0.67 * np.log10(E_seismic_J) - 5.87
        return max(0.0, Mw)

    D_crater, d_crater, V_crater = calculate_crater_dimensions(E_J, target_density_kg_m3, G_EARTH)
    Mw_seismic = calculate_seismic_shaking(E_J, SEISMIC_EFFICIENCY)

    R_ejecta_near = D_crater * 1.5
    R_ejecta_far = D_crater * 2.5
    h_near = calculate_ejecta_thickness(D_crater, R_ejecta_near)
    h_far = calculate_ejecta_thickness(D_crater, R_ejecta_far)

    return f"""
    <h2>Bare Ground Impact Analysis </h2>

    <h3>Immediate Local Effects (Crater Formation)</h3>
    <p>
        Final Crater Diameter (D):{D_crater:.2f} km<br>
        Crater Depth (d, simple):{d_crater:.2f} km<br>
        Excavated Volume (approx):{V_crater:.2e} cubic meters<br>
      (All original ground material is vaporized, melted, or ejected)
    </p>

    <h3>Secondary Effects (Ejecta Deposition)</h3>
    <p>
        Ejecta Thickness at {R_ejecta_near:.2f} km:{h_near*100:.2f} cm<br>
        Ejecta Thickness at {R_ejecta_far:.2f} km:{h_far*100:.2f} cm<br>
      (Ejecta blanket covers the surrounding area, burying the original surface)
    </p>

    <h3>Long-Range Hazard (Seismic Shaking)</h3>
    <p>
      Estimated Seismic Moment Magnitude (Mw): {Mw_seismic:.2f}<br>
      (Seismic waves can cause ground disruption hundreds of kilometers away)
    </p>
    <div style="color: grey; font-size: 0.9em; margin-top: 15px;">
      Model Reference: Simplified Pi-scaling laws for impact cratering in the gravity regime (Holsapple & Schmidt).
    </div>
    """


def model_water_bodies(E_J: float, E_MT: float) -> str:
    """Predicts the impact on rivers, lakes, or oceans (wave and cavity formation)."""
    avg_water_depth_m = DEFAULT_WATER_DEPTH_M

    # --- Phase 1: Transient Cavity Dimensions ---
    def calculate_water_cavity(E_J, rho_w, g):
        if E_J <= 0: return 0.0, "", ""
        E_prime = E_J / (rho_w * g)
        R_tc_m = K_CAVITY * (E_prime)**(1/4.0)
        D_tc_m = R_tc_m / 2.0
        R_tc_km = R_tc_m / 1000.0
        if D_tc_m > avg_water_depth_m:
            impact_type = "Bottom-out (Sub-surface Crater)"
            depth_str = f"{D_tc_m:.1f} m (Impact reached seabed)"
        else:
            impact_type = "Deep Water (Fluid Dynamics Only)"
            depth_str = f"{D_tc_m:.1f} m"
        return R_tc_km, depth_str, impact_type

    def calculate_initial_wave_height(R_tc_km):
        if R_tc_km <= 0: return 0.0
        H_wave_m = K_WAVE * R_tc_km * 1000
        return H_wave_m

    def calculate_turbidity_radius(R_tc_km):
        if R_tc_km <= 0: return 0.0
        R_turb_km = R_tc_km * 4.0
        return R_turb_km

    R_cavity, D_cavity_str, impact_type = calculate_water_cavity(E_J, RHO_WATER, G_EARTH)
    H_wave = calculate_initial_wave_height(R_cavity)
    R_turb = calculate_turbidity_radius(R_cavity)

    return f"""
    <h2>Water Body Impact Analysis </h2>

    <h3>Immediate Local Effects (Cavity & Vaporization)</h3>
    <p>
      Assumed Water Depth:{avg_water_depth_m:.1f} m<br>
      Transient Cavity Radius:{R_cavity:.2f} km<br>
      Impact Type:{impact_type}<br>
      (A large volume of water is vaporized, creating a plume of steam and ejecta)
    </p>

    <h3>Wave Hazard (Tsunami/Impact Wave)</h3>
    <p>
      Estimated Initial Wave Height (near impact):{H_wave:.1f} m<br>
      Wave Effect:The wave will flood and devastate shorelines and wetlands.<br>
      (Wave height decreases rapidly with distance due to dispersion in shallow water)
    </p>

    <h3>Ecological and Sediment Disruption</h3>
    <p>
      Heavy Sediment Suspension Radius (approx): {R_turb:.2f} km<br>
      Water Quality Effect:Extreme turbidity and thermal shock to the entire water body.<br>
      (This massive disruption kills aquatic life and contaminates drinking sources)
    </p>
    <div style="color: grey; font-size: 0.9em; margin-top: 15px;">
      Model Reference: Simplified hydrodynamic models for wave and cavity generation.
    </div>
    """


def predict_impact_gradio(asteroid_name: str, density_label_with_value: str, land_type: str) -> Tuple[str, str, str, str, str]:
    """The main function called by Gradio to calculate and format the output."""

    try:
        # Parse density from the label
        density_str = density_label_with_value.split('(')[-1].replace(' kg/m³)', '').replace(',', '').strip()
        density = float(density_str)
    except:
        density = DEFAULT_DENSITY

    global ASTEROID_DATA_LIST
    asteroid_data = find_asteroid_data(asteroid_name, ASTEROID_DATA_LIST)

    if not asteroid_data:
        error_md = f"""## ERROR<p style='color:red;'>Asteroid '<strong>{asteroid_name}</strong>' not found. Please select from the list or check the spelling.</p>"""
        return error_md, "N/A", "N/A", "N/A", "N/A"


    results = calculate_impact_energy(asteroid_data, density)

    if "error" in results:
        error_md = f"## CALCULATION ERROR <p style='color:red;'>{results['error']}</p>"
        return error_md, "N/A", "N/A", "N/A", "N/A"

    energy_megatons = results['impact_energy_megatons_tnt']
    energy_joules = results['kinetic_energy_joules']
    radius_km = results['destruction_radius_km']

    density_type_label = results['density_label'].split('(')[0].strip()

    analysis_markdown = f"""
    ## Analysis for Asteroid: {results['name']}

    <div style="border: 3px dashed #007bff; padding: 15px; background-color: #007bff15; color: white;">

    <table style="width:100%; border-collapse: collapse; color: white;">
      <thead>
        <tr>
          <th colspan="2" style="text-align: left; padding-bottom: 5px;">*ASTEROID PARAMETERS*</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>POTENTIALLY HAZARDOUS:</td><td>{'True' if results['is_potentially_hazardous'] else 'False'}</td></tr>
        <tr><td>ASSUMED DENSITY (TYPE):</td><td>{results['assumed_density_kg_m3']:,} kg/m³ ({density_type_label})</td></tr>
        <tr><td>AVG DIAMETER (ESTIMATE):</td><td>{results['average_diameter_m']:.2f} meters</td></tr>
        <tr><td>RELATIVE VELOCITY:</td><td>{results['relative_velocity_km_s']:.3f} km/s</td></tr>
        <tr><td>CALCULATED MASS:</td><td>{results['calculated_mass_kg']:.2e} kg</td></tr>
        <tr><td>IMPACT LOCATION:</td><td>{land_type}</td></tr>
      </tbody>
    </table>

    <h3 style="margin-top: 15px;">Impact Energy Prediction</h3>

    <table style="width:100%; border-collapse: collapse; color: white;">
      <tbody>
        <tr><td>KINETIC ENERGY:</td><td>{results['kinetic_energy_joules']:.2e} Joules</td></tr>
        <tr><td>EQUIVALENT ENERGY:</td><td>{energy_megatons:.2e} Megatons of TNT (MT)</td></tr>
      </tbody>
    </table>

    </div>
    """

    # 2. & 3. Output Energy & Area Metrics (Cards 1 & 2)
    energy_markdown = f"""
    ### {energy_megatons:.2e} MT
    <p style='color:#a0dfff;'>Equivalent Energy of TNT</p>
    Velocity: {results['relative_velocity_km_s']:.2f} km/s
    Density: {results['assumed_density_kg_m3']:,} kg/m³
    """
    area_markdown = f"""
    ### {radius_km:.2f} km
    <p style='color:#a0dfff;'>Destruction Radius (General Model)</p>
    Affected Area: {results['destruction_area_sq_km']:,.0f} sq. km
    Hazardous:{'YES' if results['is_potentially_hazardous'] else 'No'}
    """

    guidance_markdown = get_preparedness_guidance(energy_megatons)

    if land_type == "Vegetation (Forests, Grasslands, Crops)":
        specific_output = model_vegetation(energy_joules, energy_megatons)
    elif land_type == "Artificial Surfaces (Buildings, Roads)":
        specific_output = model_artificial_surface(energy_joules, energy_megatons)
    elif land_type == "Bare Ground (Soil, Sand, Rock)":
        specific_output = model_bare_ground(energy_joules, energy_megatons)
    elif land_type == "Water Bodies (Oceans, Lakes, Rivers)":
        specific_output = model_water_bodies(energy_joules, energy_megatons)
    else:
        specific_output = "Select a Land Type above to run the specific impact model."


    return analysis_markdown, energy_markdown, area_markdown, guidance_markdown, specific_output


# --- 5. Gradio Interface Setup ---

# --- Data Initialization (Executed ONCE when the script starts) ---
FILE_PATHS_TO_LOAD = [f'{i}.json' for i in range(1, 421)]
ASTEROID_DATA_LIST = load_data(FILE_PATHS_TO_LOAD)
ACTUAL_TOP_5_NEAREST = get_top_n_nearest_asteroids(ASTEROID_DATA_LIST, 5)

# Prepare the Top 5 List Markdown
top_5_list_markdown = "## Top 5 Nearest Asteroids (NASA PDCO Data)\n\n"
if ACTUAL_TOP_5_NEAREST:
    top_5_list_markdown += "| Rank | Asteroid Name | Closest Approach (km) |\n| :--- | :--- | :--- |\n"
    for i, (name, distance) in enumerate(ACTUAL_TOP_5_NEAREST):
        display_name = name[:20] + '...' if len(name) > 23 else name
        top_5_list_markdown += f"| {i+1} | {display_name:<20} | {distance:,.0f} |\n"
else:
    top_5_list_markdown += "No close approach data found in your JSON files."

density_choices = [f"{k} ({v:,} kg/m³)" for k, v in DENSITIES.items()]
land_type_choices = [
    "Vegetation (Forests, Grasslands, Crops)",
    "Artificial Surfaces (Buildings, Roads)",
    "Bare Ground (Soil, Sand, Rock)",
    "Water Bodies (Oceans, Lakes, Rivers)",
]

# Custom CSS for the visual fix: sets the light blue background and ensures readable text.
CUSTOM_CSS = """
    body {
        /* This is the light blue background color from your desired image */
        background-color: #e6f7ff !important;
    }

    .gradio-container {
        /* Ensures the container itself doesn't hide the body background */
        background-color: transparent !important;
    }

    /* Target major text blocks (headers, titles, and general text) to force a dark color */
    div.gradio-container h1,
    div.gradio-container h2,
    div.gradio-container h3,
    div.gradio-container p,
    div.gradio-container .gr-markdown,
    div.gradio-container .gr-text-input,
    div.gradio-container .gr-dropdown
    {
        color: #262626 !important; /* Dark color for readability */
    }

    /* Force table text to be dark for readability against white table background */
    .gr-markdown table th,
    .gr-markdown table td {
        color: #262626 !important;
        background-color: transparent !important; /* keep table cells transparent */
    }

    /* Keep your specific analysis box styling, forcing white text inside the dark blue box */
    div[style*="border: 3px dashed #007bff"] * {
        color: white !important;
    }

    /* Keep the panel styling dark (for inputs and outputs) */
    .gradio-container .panel {
        background-color: #1f2937 !important; /* Dark slate background */
        color: white !important;
    }
    .gradio-container .panel * {
        color: white !important;
    }
"""

# Gradio Block Theme Setup
with gr.Blocks(
    # NOTE: We keep the Monochromatic theme, but use CSS to override its background and text colors.
    theme=gr.themes.Monochrome(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate"
    ),
    title="ImpactorScope: NASA Planetary Defense",
    css=CUSTOM_CSS
) as themed_app:

    # --- Title Section ---
    gr.Markdown(
        """
        # ☄️ ImpactorScope: Asteroid Impact Energy Prediction Model

        This tool integrates NASA NEO parameters with USGS-based physical models to predict
        the energy and tailored consequences of asteroid impacts.
        ---
        """
    )

    # --- Input & Top 5 List Row ---
    with gr.Row():

        with gr.Column(scale=1):
            gr.Markdown("### 1. Asteroid Selection")
            name_input = gr.Textbox(
                label="Enter Asteroid Name/ID",
                value=ACTUAL_TOP_5_NEAREST[0][0] if ACTUAL_TOP_5_NEAREST else '99942 Apophis',
                placeholder="E.g., 99942 Apophis or 2024 Near-Miss"
            )
            gr.Markdown("### 2. Physical & Environmental Inputs")
            density_input = gr.Dropdown(
                label="Asteroid Density (Affects Mass)",
                choices=density_choices,
                value=density_choices[3], # Default to Rocky
                interactive=True
            )
            land_type_input = gr.Dropdown(
                label="Impact Location Land Type (Crucial for Consequence)",
                choices=land_type_choices,
                value=land_type_choices[1], # Default to Artificial Surfaces
                interactive=True
            )
            calculate_btn = gr.Button("RUN IMPACT SIMULATION", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown(top_5_list_markdown)


    gr.Markdown("---")

    # --- Main Analysis Block ---
    output_analysis_block = gr.Markdown(
        "## **Analysis for Asteroid: (Awaiting Input)**",
        elem_id="analysis_output",
    )

    gr.Markdown("---")
    gr.Markdown("## **Consequence and Preparedness Analysis**")

    # --- Output/Storyboard Card Row ---
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### **Intensity Prediction (MT)**")
            output_energy = gr.Markdown("### **Awaiting Calculation...**")

        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### **General Destruction Radius (km)**")
            output_area = gr.Markdown("### **Awaiting Calculation...**")

        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### **Preparedness Guidance (PDCO/FEMA)**")
            output_guidance = gr.Markdown("**Follow Official Alerts.** Click 'Calculate' to get tailored advice.")

    # --- Land-Type Specific Results ---
    gr.Markdown("---")
    gr.Markdown("## **Detailed Impact Model: Result Specific to Land Type**")
    specific_output_block = gr.Markdown("## **Select Land Type and Calculate Above**")


    # --- Event Handler to link button to function ---
    calculate_btn.click(
        fn=predict_impact_gradio,
        inputs=[name_input, density_input, land_type_input],
        outputs=[output_analysis_block, output_energy, output_area, output_guidance, specific_output_block]
    )
