import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# =====================================================================
# 1. THE FITTING ROUTINE 
# =====================================================================
def fit_discrimination_ellipse_from_df(df_chunk, center):
    """
    Fits an ellipse to a chunk of threshold datapoints (e.g., 20 points)
    by minimizing the sum of squares of the log radial distances.
    """
    cx, cy = center
    
    # Extract coordinates from the passed dataframe chunk (Using guaranteed lowercase keys)
    x_coords = df_chunk['x'].to_numpy()
    y_coords = df_chunk['y'].to_numpy()
    
    # Shift relative to the fixed center
    dx = x_coords - cx
    dy = y_coords - cy
    
    # Convert to polar coordinates (r, phi)
    r_data = np.sqrt(dx**2 + dy**2)
    phi_data = np.arctan2(dy, dx)
    
    # Theoretical ellipse radius calculation
    def ellipse_radius(phi, a, b, theta):
        cos_part = np.cos(phi - theta)
        sin_part = np.sin(phi - theta)
        return 1.0 / np.sqrt((cos_part / a)**2 + (sin_part / b)**2)
    
    # Objective: minimize sum of squares of log radial distances
    def objective(params):
        a, b, theta = params
        r_ellipse = ellipse_radius(phi_data, a, b, theta)
        # FIXED: Removed the stray '[cite: 46]' text that broke your run
        return np.sum((np.log(r_data) - np.log(r_ellipse))**2)
    
    initial_guess = [np.mean(r_data), np.mean(r_data), 0.0]
    bounds = [(1e-6, None), (1e-6, None), (-np.pi, np.pi)]
    
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    if not result.success:
        raise ValueError("Optimization failed to converge: " + result.message)
        
    return result.x

def generate_ellipse_xy(center, fit_a, fit_b, fit_angle):
    ellipse_angles = np.linspace(0, 2 * np.pi, 200)
    cos_e = np.cos(ellipse_angles - fit_angle)
    sin_e = np.sin(ellipse_angles - fit_angle)
    fitted_radii = 1.0 / np.sqrt((cos_e / fit_a)**2 + (sin_e / fit_b)**2)
    return center[0] + fitted_radii * np.cos(ellipse_angles), center[1] + fitted_radii * np.sin(ellipse_angles)


# =====================================================================
# 2. LOADING THE FILE
# =====================================================================
file_path = "/Users/cameronzhang/Downloads/1994_participant_data - cr_testing-2.csv" 

if not file_path:
    raise ValueError("Please provide a valid file pathway inside the blank file_path quotes.")

# Read the file and automatically convert ALL columns to lowercase to prevent matching crashes
df = pd.read_csv(file_path)
df = df.rename(columns=str.lower)

# =====================================================================
# 3. LOOP AND PROCESS DATA IN CHUNKS OF 20
# =====================================================================
plt.figure(figsize=(9, 8))

# Correct way to pull color maps across modern matplotlib environments
colors = plt.colormaps.get_cmap('tab10') 

# Calculate how many total chunks of 20 exist in the data
num_rows = len(df)
chunk_size = 20
num_ellipses = int(np.ceil(num_rows / chunk_size))

print(f"Detected {num_rows} total rows of data. Processing {num_ellipses} ellipse chunks...")

# Loop through each chunk sequentially
for i in range(num_ellipses):
    start_idx = i * chunk_size
    end_idx = min(start_idx + chunk_size, num_rows)
    
    # 1. Slice the current chunk of 20 points
    df_chunk = df.iloc[start_idx:end_idx]
    
    # 2. Extract the center coordinate from the current loop row index
    center_row_idx = min(i, len(df['x_center'].dropna()) - 1)
    bg_center = (df['x_center'].iloc[center_row_idx], df['y_center'].iloc[center_row_idx])
    
    # 3. Fit the current chunk
    fit_a, fit_b, fit_angle = fit_discrimination_ellipse_from_df(df_chunk, bg_center)
    
    print(f"\n--- Ellipse {i+1} Results (Rows {start_idx} to {end_idx}) ---")
    print(f"Center Coordinate:      {bg_center}")
    print(f"Semi-axis 'a' (Length): {fit_a:.5f}")
    print(f"Semi-axis 'b' (Width):  {fit_b:.5f}")
    print(f"Rotation Angle:         {np.degrees(fit_angle):.2f}°")
    
    # 4. Plot current chunk elements
    current_color = colors(i % 10) # modulo prevents crash if you have more than 10 ellipses
    
    # Standardized to lowercase 'x' and 'y'
    plt.scatter(df_chunk['x'], df_chunk['y'], color=current_color, alpha=0.5, label=f"Data Chunk {i+1}")
    
    # Plot the continuous line boundary
    ellipse_x, ellipse_y = generate_ellipse_xy(bg_center, fit_a, fit_b, fit_angle)
    plt.plot(ellipse_x, ellipse_y, color=current_color, linewidth=2, label=f"Fitted Ellipse {i+1}")
    
    # Plot the matching color center marker
    plt.scatter(bg_center[0], bg_center[1], color=current_color, marker='+', s=150, zorder=5)

# =====================================================================
# 4. PLOT FORMATTING
# =====================================================================
plt.xlabel("u' Coordinate")
plt.ylabel("v' Coordinate")
plt.title('Regan & Mollon Multi-Ellipse Fitting Loop')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.grid(True, linestyle='--', alpha=0.5)

# Set the axis limits as requested
plt.xlim(0, 0.45)
plt.ylim(0, 0.65)

# Enforces a 1:1 pixel scaling so circles/ellipses preserve true geometric shape
plt.axis('equal') 

plt.tight_layout()
plt.show()