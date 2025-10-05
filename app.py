from flask import Flask, render_template, request, jsonify
import json
import math
from typing import Dict, List, Optional, Any, Tuple

app = Flask(__name__)

# --- Constants for Asteroid Density based on Type (kg/m^3) ---
DENSITIES = {
    "C-type (Carbonaceous)": 1300,
    "S-type (Stony)": 2700,
    "M-type (Metallic)": 5000,
    "Rocky": 3000
}
DEFAULT_DENSITY = DENSITIES["Rocky"]

# Conversion constant for Megatons of TNT
JOULES_PER_MEGATON = 4.184e15

def load_data(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Loads and flattens asteroid data from multiple JSON files."""
    all_asteroids = []
    for path in file_paths:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            neo_by_date = data.get('near_earth_objects', {})
            for date_data in neo_by_date.values():
                all_asteroids.extend(date_data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return all_asteroids

def find_asteroid_data(asteroid_name: str, asteroid_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Finds the asteroid by name and returns its complete data dictionary."""
    normalized_name = asteroid_name.strip().upper().replace('(', '').replace(')', '').replace('-', ' ')
    
    for asteroid in asteroid_list:
        current_name = asteroid.get('name', '').strip().upper().replace('(', '').replace(')', '').replace('-', ' ')
        if normalized_name == current_name or normalized_name == asteroid.get('neo_reference_id'):
            return asteroid
        if normalized_name in current_name:
            return asteroid
    return None

def get_top_n_nearest_asteroids(asteroid_list: List[Dict[str, Any]], n: int = 5) -> List[Tuple[str, float]]:
    """Returns top N asteroids with smallest minimum close approach distance."""
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

def calculate_impact_energy(asteroid_data: Dict[str, Any], density: float) -> Dict[str, Any]:
    """Calculates the approximate impact energy of the asteroid."""
    if not asteroid_data:
        return {"error": "Asteroid data not found."}
    
    try:
        # Extract and Calculate Average Diameter
        estimated_diameter = asteroid_data['estimated_diameter']['meters']
        min_d = float(estimated_diameter['estimated_diameter_min'])
        max_d = float(estimated_diameter['estimated_diameter_max'])
        avg_diameter = (min_d + max_d) / 2
        avg_radius = avg_diameter / 2
        
        # Extract Relative Velocity
        close_approach = asteroid_data['close_approach_data'][-1]
        relative_velocity_km_s = float(close_approach['relative_velocity']['kilometers_per_second'])
        V_m_s = relative_velocity_km_s * 1000
        
        # Calculate Mass: M = rho * (4/3) * pi * r^3
        mass_kg = density * (4/3) * math.pi * (avg_radius ** 3)
        
        # Calculate Kinetic Energy: E_k = 0.5 * M * V^2
        kinetic_energy_joules = 0.5 * mass_kg * (V_m_s ** 2)
        
        # Convert to Megatons of TNT
        energy_megatons = kinetic_energy_joules / JOULES_PER_MEGATON
        
        # Contextual Interpretation
        if energy_megatons > 100:
            interpretation = "This is a devastating impact event, comparable to large scale nuclear war scenarios. Mitigation is critical."
        elif energy_megatons > 10:
            interpretation = "This is a major regional impact event, capable of causing widespread destruction and global climate effects."
        elif energy_megatons > 1:
            interpretation = "This is a powerful impact, capable of leveling a major metropolitan area (like the Tunguska event)."
        else:
            interpretation = "This is a smaller impact, likely resulting in localized damage or an atmospheric explosion (airburst)."
        
        return {
            "name": asteroid_data.get('name', 'N/A'),
            "average_diameter_m": f"{avg_diameter:.2f} meters",
            "relative_velocity_km_s": f"{relative_velocity_km_s:.3f} km/s",
            "assumed_density_kg_m3": f"{density:,} kg/m³",
            "calculated_mass_kg": f"{mass_kg:.2e} kg",
            "kinetic_energy_joules": f"{kinetic_energy_joules:.2e}",
            "impact_energy_megatons_tnt": f"{energy_megatons:.2e} Megatons of TNT",
            "is_potentially_hazardous": asteroid_data.get('is_potentially_hazardous_asteroid', False),
            "interpretation": interpretation
        }
    
    except KeyError as e:
        return {"error": f"Missing required data field: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred during calculation: {e}"}

# Load asteroid data on startup
file_paths = [f'{i}.json' for i in range(1, 151)]
ASTEROID_LIST = load_data(file_paths)

@app.route('/')
def index():
    """Renders the main page with top 5 nearest asteroids."""
    top_asteroids = get_top_n_nearest_asteroids(ASTEROID_LIST, 5)
    return render_template('index.html', top_asteroids=top_asteroids, densities=DENSITIES)

@app.route('/calculate_impact', methods=['POST'])
def calculate_impact():
    """Handles the impact energy calculation request."""
    asteroid_name = request.form.get('asteroid_name', '').strip()
    density_value = float(request.form.get('density_value', DEFAULT_DENSITY))
    
    if not asteroid_name:
        return jsonify({"error": "Asteroid name is required."})
    
    asteroid_data = find_asteroid_data(asteroid_name, ASTEROID_LIST)
    
    if not asteroid_data:
        return jsonify({"error": f"Asteroid '{asteroid_name}' not found in the dataset."})
    
    results = calculate_impact_energy(asteroid_data, density_value)
    return jsonify(results)

if __name__ == '__main__':
    if not ASTEROID_LIST:
        print("WARNING: No asteroid data loaded. Place JSON files (1.json to 150.json) in the same directory.")
    app.run(debug=True)
