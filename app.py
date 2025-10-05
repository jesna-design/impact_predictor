import json
import math
from typing import Dict, List, Optional, Any, Tuple

# --- Constants for Asteroid Density based on Type (kg/m^3) ---
# These are standard astronomical assumptions.
DENSITIES = {
    "C-type (Carbonaceous)": 1300,  # Lower density, most common type
    "S-type (Stony)": 2700,      # Medium density, common
    "M-type (Metallic)": 5000,    # High density, metallic
    "Rocky": 3000                 # General assumption for rocky body (used if no specific type is chosen)
}
DEFAULT_DENSITY = DENSITIES["Rocky"]

# Conversion constant for Megatons of TNT (Joules per Megaton)
JOULES_PER_MEGATON = 4.184e15

def load_data(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Loads and flattens asteroid data from multiple JSON files."""
    all_asteroids = []
    for path in file_paths:
        try:
            # NOTE: Due to lack of access to your files, this will likely fail
            # unless the files are present in the execution environment.
            with open(path, 'r') as f:
                data = json.load(f)

            neo_by_date = data.get('near_earth_objects', {})

            for date_data in neo_by_date.values():
                all_asteroids.extend(date_data)

        except FileNotFoundError:
            # print(f"Error: File not found at {path}. Skipping.")
            pass # Suppress print for cleaner output when many files are missing
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {path}. Skipping.")

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
    """
    Analyzes the full dataset to find the top N asteroids with the smallest
    minimum close approach distance (in kilometers).
    Returns a list of (name, min_distance_km).
    """
    asteroid_distances: Dict[str, float] = {}

    for asteroid in asteroid_list:
        name = asteroid.get('name', 'UNKNOWN')
        # Only consider objects with close approach data
        if not asteroid['close_approach_data']:
            continue
            
        # Find the minimum distance across all close approaches for this asteroid
        min_distance_km = float('inf')
        for approach in asteroid['close_approach_data']:
            try:
                distance_km = float(approach['miss_distance']['kilometers'])
                if distance_km < min_distance_km:
                    min_distance_km = distance_km
            except (KeyError, ValueError):
                continue

        if min_distance_km != float('inf'):
            # Store the minimum distance for this unique asteroid name
            # If the same asteroid appears multiple times in the dataset (which it shouldn't
            # in a flattened list from NEO-WS), we take the absolute closest one recorded.
            if name not in asteroid_distances or min_distance_km < asteroid_distances[name]:
                 asteroid_distances[name] = min_distance_km


    # Convert to a list of (name, distance) tuples and sort by distance
    sorted_asteroids = sorted(asteroid_distances.items(), key=lambda item: item[1])
    
    # Return the top N
    return sorted_asteroids[:n]


def calculate_impact_energy(asteroid_data: Dict[str, Any], density: float) -> Dict[str, Any]:
    """
    Calculates the approximate impact energy (kinetic energy) of the asteroid
    at its closest approach to Earth, using a specified density.
    """
    if not asteroid_data:
        return {"error": "Asteroid data not found."}

    try:
        # 1. Extract and Calculate Average Diameter (in meters)
        estimated_diameter = asteroid_data['estimated_diameter']['meters']
        min_d = float(estimated_diameter['estimated_diameter_min'])
        max_d = float(estimated_diameter['estimated_diameter_max'])

        # Calculate the average radius (r) and average diameter (d)
        avg_diameter = (min_d + max_d) / 2
        avg_radius = avg_diameter / 2

        # 2. Extract Relative Velocity (in km/s and convert to m/s)
        # We assume the last listed close approach data for the velocity
        close_approach = asteroid_data['close_approach_data'][-1]

        # Velocity is given in km/s, so we must convert to m/s (multiply by 1000)
        relative_velocity_km_s = float(close_approach['relative_velocity']['kilometers_per_second'])
        V_m_s = relative_velocity_km_s * 1000

        # 3. Calculate Mass (M)
        # M = rho * (4/3) * pi * r^3
        # *CORRECTED: Use the exponentiation operator () for power*
        mass_kg = density * (4/3) * math.pi * (avg_radius ** 3)

        # 4. Calculate Kinetic Energy (E_k) in Joules
        # E_k = 0.5 * M * V^2
        # *CORRECTED: Use the exponentiation operator () for power*
        kinetic_energy_joules = 0.5 * mass_kg * (V_m_s ** 2)

        # 5. Convert to Megatons of TNT
        energy_megatons = kinetic_energy_joules / JOULES_PER_MEGATON
        return {
            "name": asteroid_data.get('name', 'N/A'),
            "average_diameter_m": avg_diameter,
            "relative_velocity_km_s": relative_velocity_km_s,
            "assumed_density_kg_m3": density,
            "calculated_mass_kg": mass_kg,
            "kinetic_energy_joules": kinetic_energy_joules,
            "impact_energy_megatons_tnt": energy_megatons,
            "is_potentially_hazardous": asteroid_data.get('is_potentially_hazardous_asteroid', False)
        }

    except KeyError as e:
        return {"error": f"Missing required data field: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred during calculation: {e}"}

def predict_impact_energy_model(asteroid_name: str, asteroid_density: float):
    """Main function to load data, find asteroid, and predict impact energy."""
    print("--- Asteroid Impact Energy Prediction Model ---")

    # Files provided by the user
    file_paths = [f'{i}.json' for i in range(1, 151)] # Generates '1.json' to '150.json'

    # 1. Load Data
    asteroid_list = load_data(file_paths)
    if not asteroid_list:
        print("Error: Could not load data from JSON files. Check file paths and content.")
        # Attempt to run top_n analysis even with partial data
        if not asteroid_list:
            return

    # 2. Find Asteroid
    asteroid_data = find_asteroid_data(asteroid_name, asteroid_list)

    if not asteroid_data:
        print(f"\nAsteroid '{asteroid_name}' not found in the loaded dataset.")
        return

    # 3. Calculate Energy
    # Pass the selected density to the calculation function
    results = calculate_impact_energy(asteroid_data, asteroid_density)

    # 4. Display Results
    if "error" in results:
        print(f"\nCalculation Error: {results['error']}")
        return
    
    print(f"\nAnalysis for Asteroid: {results['name']}")
    print("-" * 40)
    print(f"Potentially Hazardous: {results['is_potentially_hazardous']}")
    print(f"Assumed Density (Type): {results['assumed_density_kg_m3']:,} kg/m³")
    print(f"Avg Diameter (Estimate): {results['average_diameter_m']:.2f} meters")
    print(f"Relative Velocity: {results['relative_velocity_km_s']:.3f} km/s")
    print(f"Calculated Mass: {results['calculated_mass_kg']:.2e} kg")
    print("\n--- IMPACT ENERGY PREDICTION ---")
    print(f"Kinetic Energy: {results['kinetic_energy_joules']:.2e} Joules")
    print(f"Equivalent Energy: {results['impact_energy_megatons_tnt']:.2e} Megatons of TNT (MT)")
    print("-" * 40)

    # Contextual Interpretation
    if results['impact_energy_megatons_tnt'] > 100:
        print("Interpretation: This is a devastating impact event, comparable to large scale nuclear war scenarios. Mitigation is critical.")
    elif results['impact_energy_megatons_tnt'] > 10:
        print("Interpretation: This is a major regional impact event, capable of causing widespread destruction and global climate effects.")
    elif results['impact_energy_megatons_tnt'] > 1:
        print("Interpretation: This is a powerful impact, capable of leveling a major metropolitan area (like the Tunguska event).")
    else:
        print("Interpretation: This is a smaller impact, likely resulting in localized damage or an atmospheric explosion (airburst).")

# --- Interactive Execution ---
if _name_ == "_main_":
    
    # 1. Load Data
    file_paths = [f'{i}.json' for i in range(1, 151)]
    asteroid_list = load_data(file_paths)

    if not asteroid_list:
        print("\n*WARNING*: Could not load any asteroid data. Check if files '1.json' through '150.json' are present.")
        # Proceed with mock data/exit if no data is available
        exit()


    # 2. Analyze and Display Top 5 Nearest
    print("\n--- DATASET ANALYSIS: TOP 5 NEAREST ASTEROIDS ---")
    
    top_5_asteroids = get_top_n_nearest_asteroids(asteroid_list, 5)
    
    if top_5_asteroids:
        print("Rank | Asteroid Name | Closest Approach (km)")
        print("------------------------------------------")
        for i, (name, distance) in enumerate(top_5_asteroids):
            print(f"{i+1:<4} | {name:<13} | {distance:,.0f}")
        print("------------------------------------------")
    else:
        print("No asteroids with close approach data found in the dataset.")
        exit()

    # 3. Get User Input: Asteroid Name
    user_input_name = input("\nEnter the name of the asteroid you want to analyze from the list (or any other asteroid): ")

    # 4. Get User Input: Asteroid Type/Density
    print("\n--- ASTEROID DENSITY SELECTION ---")
    print("This selection determines the density (mass) used in the impact calculation.")
    
    density_options = list(DENSITIES.keys())
    print("Options:")
    for i, option in enumerate(density_options):
        print(f"  {i+1}: {option} ({DENSITIES[option]:,} kg/m³)")
    print(f"  (Default is 'Rocky' if not chosen)")

    density_choice = input(f"Enter the number for the asteroid type (1-{len(density_options)}): ")

    # Determine the density
    try:
        index = int(density_choice) - 1
        if 0 <= index < len(density_options):
            selected_type = density_options[index]
            selected_density = DENSITIES[selected_type]
            print(f"Selected type: {selected_type}. Density set to {selected_density:,} kg/m³.")
        else:
            selected_density = DEFAULT_DENSITY
            print(f"Invalid choice. Using default density: {DEFAULT_DENSITY:,} kg/m³ (Rocky).")
    except ValueError:
        selected_density = DEFAULT_DENSITY
        print(f"Invalid input. Using default density: {DEFAULT_DENSITY:,} kg/m³ (Rocky).")


    # 5. Run the prediction model
    predict_impact_energy_model(user_input_name, selected_density)
