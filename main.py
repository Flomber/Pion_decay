import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit


# ============================================================
# CONFIGURATION
# ============================================================
USE_RANDOMNESS = True  # True: Random smearing, False: Deterministic +1σ smearing
RANDOM_SEED = 42       # Seed for reproducibility (when USE_RANDOMNESS=True)


def load_photon_data(decay_process='B+'):
    """
    Load photon data from the CSV files.

    Parameters:
    -----------
    decay_process : str
        'B+' for B⁺ → π⁰π⁺ decay
        'D0' for D⁰ → K⁺K⁻π⁻π⁺π⁰ decay

    Returns:
    --------
    gamma_0 : numpy array (N, 4)
        4-vectors of the first photon [px, py, pz, E]
    gamma_1 : numpy array (N, 4)
        4-vectors of the second photon [px, py, pz, E]
    """
    # File selection based on the decay process
    if decay_process == 'B+':
        filename = 'photons_from_BpToPipPiz.csv'
        print(f"Loading data for B⁺ → π⁰π⁺ decay...")
    elif decay_process == 'D0':
        filename = 'photons_from_D0ToKmKpPimPipPiz.csv'
        print(f"Loading data for D⁰ → K⁺K⁻π⁻π⁺π⁰ decay...")
    else:
        raise ValueError("decay_process must be 'B+' or 'D0'")
    
    # Read data
    filepath = Path(__file__).parent / filename
    df = pd.read_csv(filepath)
    
    # Extract 4-vectors for both photons
    gamma_0 = df[['gamma_0_PX_TRUE', 'gamma_0_PY_TRUE', 
                   'gamma_0_PZ_TRUE', 'gamma_0_E_TRUE']].values
    gamma_1 = df[['gamma_1_PX_TRUE', 'gamma_1_PY_TRUE', 
                   'gamma_1_PZ_TRUE', 'gamma_1_E_TRUE']].values
    
    print(f"  Number of events: {len(df)}")
    print(f"  Gamma 0 shape: {gamma_0.shape}")
    print(f"  Gamma 1 shape: {gamma_1.shape}")
    
    return gamma_0, gamma_1


def calculate_opening_angle(gamma_0, gamma_1):
    """
    Compute the opening angle between two photons.

    Parameters:
    -----------
    gamma_0, gamma_1 : numpy array (N, 4) with [px, py, pz, E]

    Returns:
    --------
    theta : numpy array (N,) - opening angle in radians
    """
    # Momentum vectors (first 3 components)
    p0 = gamma_0[:, :3]
    p1 = gamma_1[:, :3]
    
    # Energies (4th component)
    E0 = gamma_0[:, 3]
    E1 = gamma_1[:, 3]
    
    # Dot product of the momenta
    dot_product = np.sum(p0 * p1, axis=1)
    
    # cos(theta) = (p0 · p1) / (|p0| * |p1|) = (p0 · p1) / (E0 * E1)
    cos_theta = dot_product / (E0 * E1)
    
    # Check for values outside [-1, 1] and warn
    out_of_bounds = (cos_theta < -1) | (cos_theta > 1)
    if np.any(out_of_bounds):
        indices = np.where(out_of_bounds)[0]
        print(f"\n⚠️  WARNING: Numerical rounding issues in {len(indices)} events!")
        print(f"   Applying clip function to the following events:")
        for idx in indices[:10]:  # Zeige max. 10 Beispiele
            print(f"   - Event {idx}: cos(theta) = {cos_theta[idx]:.15f}")
        if len(indices) > 10:
            print(f"   ... and {len(indices) - 10} additional events")
    
    # Numerical stability: clip to [-1, 1]
    cos_theta = np.clip(cos_theta, -1, 1)
    
    # Opening angle
    theta = np.arccos(cos_theta)
    
    return theta


def calculate_calorimeter_positions(gamma_0, gamma_1, z_calo=12.0):
    """
    Compute positions on the calorimeter and separations.

    Parameters:
    -----------
    gamma_0, gamma_1 : numpy array (N, 4) with [px, py, pz, E]
    z_calo : float - z-position of the calorimeter in meters

    Returns:
    --------
    positions_0 : numpy array (N, 2) - (x, y) positions of photon 0
    positions_1 : numpy array (N, 2) - (x, y) positions of photon 1
    distances : numpy array (N,) - separations in meters
    """
    # Extract momentum components
    px0, py0, pz0 = gamma_0[:, 0], gamma_0[:, 1], gamma_0[:, 2]
    px1, py1, pz1 = gamma_1[:, 0], gamma_1[:, 1], gamma_1[:, 2]
    
    # Compute positions on the calorimeter
    # x_calo = z_calo * px / pz
    x0 = z_calo * px0 / pz0
    y0 = z_calo * py0 / pz0
    
    x1 = z_calo * px1 / pz1
    y1 = z_calo * py1 / pz1
    
    # Combine to position vectors
    positions_0 = np.column_stack([x0, y0])
    positions_1 = np.column_stack([x1, y1])
    
    # Compute separations
    distances = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    
    return positions_0, positions_1, distances


def plot_histograms(opening_angles, distances, decay_process):
    """
    Create histograms for opening angles and separations.

    Parameters:
    -----------
    opening_angles : numpy array - opening angles in radians
    distances : numpy array - separations in meters
    decay_process : str - name of the decay process

    Returns:
    --------
    cell_size : float - maximum cell size (31.7% quantile) in cm
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram 1: Opening angle
    angles_deg = np.degrees(opening_angles)
    ax1.hist(angles_deg, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Opening angle [°]', fontsize=12)
    ax1.set_ylabel('Number of events', fontsize=12)
    ax1.set_title(f'Opening angle ({decay_process})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics on plot
    mean_angle = np.mean(angles_deg)
    median_angle = np.median(angles_deg)
    ax1.text(0.95, 0.95, f'Mean: {mean_angle:.2f}°\nMedian: {median_angle:.2f}°',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Histogram 2: Separation on calorimeter
    distances_cm = distances * 100  # Meter → Zentimeter
    ax2.hist(distances_cm, bins=100, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Separation on calorimeter [cm]', fontsize=12)
    ax2.set_ylabel('Number of events', fontsize=12)
    ax2.set_title(f'Separation on ECAL ({decay_process})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Compute and mark the 31.7% quantile (so 68.3% events are separated)
    cell_size = np.percentile(distances_cm, 31.7)
    ax2.axvline(cell_size, color='red', linestyle='--', linewidth=2,
                label=f'Max cell size (31.7% quantile): {cell_size:.2f} cm')
    ax2.legend(fontsize=10)
    
    # Add statistics on plot
    mean_dist = np.mean(distances_cm)
    median_dist = np.median(distances_cm)
    ax2.text(0.95, 0.95, f'Mean: {mean_dist:.2f} cm\nMedian: {median_dist:.2f} cm',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save histogram as PNG
    filename = f'histograms_opening_angle_distance_{decay_process}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved histograms as: {filename}")
    
    plt.show()
    
    return cell_size


def select_accepted_events(gamma_0, gamma_1, distances, cell_size_cm):
    """
    Select events where the photons hit separate cells.

    Parameters:
    -----------
    gamma_0, gamma_1 : numpy array (N, 4)
    distances : numpy array (N,) - separations in meters
    cell_size_cm : float - maximum cell size in cm

    Returns:
    --------
    gamma_0_accepted : numpy array (M, 4) - accepted photons 0
    gamma_1_accepted : numpy array (M, 4) - accepted photons 1
    acceptance_rate : float - fraction of accepted events
    """
    # Convert separations to cm
    distances_cm = distances * 100
    
    # Boolean mask: True if separation > cell size
    accepted_mask = distances_cm > cell_size_cm
    
    # Apply mask (select True entries only)
    gamma_0_accepted = gamma_0[accepted_mask]
    gamma_1_accepted = gamma_1[accepted_mask]
    
    # Compute acceptance rate
    n_total = len(gamma_0)
    n_accepted = len(gamma_0_accepted)
    acceptance_rate = n_accepted / n_total
    
    print(f"\nEvent selection:")
    print(f"  Cell size: {cell_size_cm:.2f} cm")
    print(f"  Total events: {n_total}")
    print(f"  Accepted events: {n_accepted}")
    print(f"  Rejected: {n_total - n_accepted}")
    print(f"  Acceptance rate: {acceptance_rate*100:.1f}%")
    
    return gamma_0_accepted, gamma_1_accepted, acceptance_rate


def apply_energy_smearing(energies, resolution_factor=0.042, seed=42, use_randomness=True):
    """
    Apply Gaussian energy smearing.

    Parameters:
    -----------
    energies : numpy array - true energies in GeV
    resolution_factor : float - resolution factor (4.2% = 0.042)
    seed : int or None - random seed for reproducibility
    use_randomness : bool - True: random smearing, False: deterministic +1σ

    Returns:
    --------
    energies_smeared : numpy array - "measured" energies in GeV
    """
    # Compute uncertainty for each energy
    # σ_E = resolution_factor × √E
    sigma_E = resolution_factor * np.sqrt(energies)
    
    if not use_randomness:
        # Deterministic mode: apply exactly +1σ error
        print("  ℹ️  Randomness disabled — using deterministic +1σ smearing")
        energies_smeared = energies + sigma_E
        return energies_smeared
    
    # Random mode: set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Draw random errors from a normal distribution
    # For each energy, draw an individual error from N(0, σ_E)
    random_errors = np.random.normal(0, sigma_E)
    
    # Add error to the true energy
    energies_smeared = energies + random_errors
    
    # Check for negative energies and warn
    negative_mask = energies_smeared < 0
    if np.any(negative_mask):
        indices = np.where(negative_mask)[0]
        print(f"\n⚠️  WARNING: {len(indices)} negative energies after smearing!")
        print(f"   Setting these to 1 MeV.")
        print(f"   Affected events (showing up to 10):")
        for idx in indices[:10]:
            print(f"   - Event {idx}: E_true = {energies[idx]:.4f} GeV, "
                  f"Fehler = {random_errors[idx]:+.4f} GeV, "
                  f"E_smeared = {energies_smeared[idx]:.4f} GeV")
        if len(indices) > 10:
            print(f"   ... and {len(indices) - 10} additional events")
    
    # Prevent negative energies (unphysical)
    energies_smeared = np.maximum(energies_smeared, 0.001)  # Minimum 1 MeV
    
    return energies_smeared


def calculate_invariant_mass(gamma_0, gamma_1, E0_measured, E1_measured):
    """
    Compute the invariant mass of two photons.

    Parameters:
    -----------
    gamma_0, gamma_1 : numpy array (N, 4) - 4-vectors [px, py, pz, E]
    E0_measured, E1_measured : numpy array (N,) - measured energies in GeV

    Returns:
    --------
    invariant_mass : numpy array (N,) - invariant mass in GeV/c²
    """
    # Extract true momentum vectors (only direction is used)
    p0_true = gamma_0[:, :3]  # [px, py, pz]
    p1_true = gamma_1[:, :3]
    
    # Compute momentum magnitudes (should equal E for photons)
    p0_mag = np.sqrt(np.sum(p0_true**2, axis=1))
    p1_mag = np.sqrt(np.sum(p1_true**2, axis=1))
    
    # Compute unit vectors (directions)
    p0_unit = p0_true / p0_mag[:, np.newaxis]
    p1_unit = p1_true / p1_mag[:, np.newaxis]
    
    # Reconstruct momenta with measured energies
    # For photons: |p| = E
    p0_reconstructed = E0_measured[:, np.newaxis] * p0_unit
    p1_reconstructed = E1_measured[:, np.newaxis] * p1_unit
    
    # Compute total energy and total momentum
    E_total = E0_measured + E1_measured
    p_total = p0_reconstructed + p1_reconstructed
    
    # Magnitude of total momentum
    p_total_mag = np.sqrt(np.sum(p_total**2, axis=1))
    
    # Invariant mass: m² = E² - p²
    mass_squared = E_total**2 - p_total_mag**2
    
    # Check for negative values and warn
    negative_mask = mass_squared < 0
    if np.any(negative_mask):
        indices = np.where(negative_mask)[0]
        print(f"\n⚠️  WARNING: {len(indices)} negative mass squares!")
        print(f"   Setting these to 0 (numerical rounding).")
        print(f"   Affected events (showing up to 10):")
        for idx in indices[:10]:
            print(f"   - Event {idx}: E_total² = {E_total[idx]**2:.6f} GeV², "
                  f"p_total² = {p_total_mag[idx]**2:.6f} GeV², "
                  f"m² = {mass_squared[idx]:.6e} GeV²")
        if len(indices) > 10:
            print(f"   ... and {len(indices) - 10} additional events")
    
    # Prevent negative values due to numerical errors
    mass_squared = np.maximum(mass_squared, 0)
    
    invariant_mass = np.sqrt(mass_squared)
    
    return invariant_mass


def gaussian(x, amplitude, mean, sigma):
    """Gaussian function for fitting."""
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma)**2)


def plot_mass_histogram(invariant_mass, decay_process):
    """
    Create a histogram of the invariant mass and fit a Gaussian curve.

    Parameters:
    -----------
    invariant_mass : numpy array - invariant masses in GeV/c²
    decay_process : str - name of the decay process

    Returns:
    --------
    fit_params : dict - fit parameters (mean, sigma, resolution)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    counts, bin_edges, patches = ax.hist(invariant_mass, bins=100, 
                                          alpha=0.7, color='blue', 
                                          edgecolor='black', label='Data')
    
    # Compute bin centers for fitting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Initial estimates for fit parameters
    max_count = np.max(counts)
    mean_estimate = np.mean(invariant_mass)
    sigma_estimate = np.std(invariant_mass)
    
    try:
        # Perform Gaussian fit
        popt, pcov = curve_fit(gaussian, bin_centers, counts,
                                p0=[max_count, mean_estimate, sigma_estimate])
        
        amplitude_fit, mean_fit, sigma_fit = popt
        
        # Plot fitted curve
        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)
        y_fit = gaussian(x_fit, amplitude_fit, mean_fit, sigma_fit)
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Gaussian fit')
        
        # Compute relative resolution
        resolution = (sigma_fit / mean_fit) * 100  # in percent
        
        # Text box with fit results
        textstr = f'Fit result:\n'
        textstr += f'μ = {mean_fit*1000:.2f} MeV/c²\n'
        textstr += f'σ = {sigma_fit*1000:.2f} MeV/c²\n'
        textstr += f'σ/μ = {resolution:.2f}%\n'
        textstr += f'\nTheory:\n'
        textstr += f'm(π⁰) = 135.0 MeV/c²'
        
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Mark theoretical value
        ax.axvline(0.135, color='green', linestyle='--', linewidth=2,
                   label='Theoretical value (135 MeV)')
        
        fit_success = True
        fit_params = {
            'mean': mean_fit,
            'sigma': sigma_fit,
            'resolution': resolution,
            'amplitude': amplitude_fit
        }
        
    except Exception as e:
        print(f"\n⚠️  WARNING: Gaussian fit failed: {e}")
        print(f"   Falling back to simple statistics instead of fit.")
        fit_success = False
        fit_params = {
            'mean': mean_estimate,
            'sigma': sigma_estimate,
            'resolution': (sigma_estimate / mean_estimate) * 100,
            'amplitude': None
        }
    
    # Axis labels and title
    ax.set_xlabel('Invariant mass [GeV/c²]', fontsize=12)
    ax.set_ylabel('Number of events', fontsize=12)
    ax.set_title(f'π⁰ mass peak ({decay_process})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save histogram as PNG
    filename = f'histogram_pion_mass_{decay_process}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved mass histogram as: {filename}")
    
    plt.show()
    
    return fit_params


def analyze_decay(decay_process):
    """
    Run the complete analysis for a decay process.

    Parameters:
    -----------
    decay_process : str - 'B+' or 'D0'
    """
    print("\n" + "="*70)
    print(f"  HOMEWORK 4 - Calorimeter analysis for {decay_process} decay")
    print("="*70)
    
    # ========== TASK A) ==========
    print("\n" + "#"*70)
    print("#  TASK A) - Determine maximum cell size")
    print("#"*70)
    
    # Load photon data
    gamma_0, gamma_1 = load_photon_data(decay_process)
    
    print(f"\nFirst 3 events (photon 0):")
    print(gamma_0[:3])
    print(f"\nFirst 3 events (photon 1):")
    print(gamma_1[:3])
    
    # Compute opening angles
    print(f"\n{'='*60}")
    print("Compute opening angle between photons...")
    print('='*60)
    opening_angles = calculate_opening_angle(gamma_0, gamma_1)
    
    print(f"\nOpening angle statistics:")
    print(f"  Mean:   {np.mean(opening_angles):.4f} rad = {np.degrees(np.mean(opening_angles)):.2f}°")
    print(f"  Median: {np.median(opening_angles):.4f} rad = {np.degrees(np.median(opening_angles)):.2f}°")
    print(f"  Min:    {np.min(opening_angles):.4f} rad = {np.degrees(np.min(opening_angles)):.2f}°")
    print(f"  Max:    {np.max(opening_angles):.4f} rad = {np.degrees(np.max(opening_angles)):.2f}°")
    
    # Compute positions on the calorimeter and separations
    print(f"\n{'='*60}")
    print("Compute positions on the calorimeter (z = 12 m)...")
    print('='*60)
    positions_0, positions_1, distances = calculate_calorimeter_positions(gamma_0, gamma_1, z_calo=12.0)
    
    print(f"\nCalorimeter separation statistics:")
    print(f"  Mean:   {np.mean(distances):.4f} m = {np.mean(distances)*100:.2f} cm")
    print(f"  Median: {np.median(distances):.4f} m = {np.median(distances)*100:.2f} cm")
    print(f"  Min:    {np.min(distances):.4f} m = {np.min(distances)*100:.2f} cm")
    print(f"  Max:    {np.max(distances):.4f} m = {np.max(distances)*100:.2f} cm")
    print(f"\nFirst 5 positions:")
    for i in range(min(5, len(positions_0))):
        print(f"  Event {i}: γ₀=({positions_0[i,0]:6.3f}, {positions_0[i,1]:6.3f}) m, "
              f"γ₁=({positions_1[i,0]:6.3f}, {positions_1[i,1]:6.3f}) m, "
              f"Separation={distances[i]:.4f} m")
    
    # Create histograms
    print(f"\n{'='*60}")
    print("Create histograms...")
    print('='*60)
    quantile_68_cm = plot_histograms(opening_angles, distances, decay_process)
    
    print(f"\n{'='*60}")
    print("RESULT TASK A): Maximum cell size")
    print('='*60)
    print(f"31.7% quantile of separations: {quantile_68_cm:.2f} cm")
    print(f"\nMaximum square cell size: {quantile_68_cm:.2f} cm × {quantile_68_cm:.2f} cm")
    print(f"At this cell size, in 68.3% of cases the two photons")
    print(f"from the π⁰ decay hit separate cells (separation > cell size).")
    
    # ========== TASK B) ==========
    print("\n" + "#"*70)
    print("#  TASK B) - π⁰ mass resolution")
    print("#"*70)
    
    # Select accepted events
    print(f"\n{'='*60}")
    print("Select accepted events...")
    print('='*60)
    gamma_0_accepted, gamma_1_accepted, acceptance_rate = select_accepted_events(
        gamma_0, gamma_1, distances, quantile_68_cm
    )
    
    # Apply energy smearing
    print(f"\n{'='*60}")
    print("Apply energy smearing (simulate detector resolution)...")
    print(f"Resolution: σ_E/E = 4.2%/√E[GeV]")
    print('='*60)
    
    # Extract true energies
    E0_true = gamma_0_accepted[:, 3]
    E1_true = gamma_1_accepted[:, 3]
    
    print(f"\nTrue energy statistics (accepted events):")
    print(f"  Photon 0 - Mean: {np.mean(E0_true):.3f} GeV, Range: [{np.min(E0_true):.3f}, {np.max(E0_true):.3f}] GeV")
    print(f"  Photon 1 - Mean: {np.mean(E1_true):.3f} GeV, Range: [{np.min(E1_true):.3f}, {np.max(E1_true):.3f}] GeV")
    
    # Apply smearing (with fixed seed for reproducibility)
    E0_measured = apply_energy_smearing(E0_true, resolution_factor=0.042, 
                                         seed=RANDOM_SEED, use_randomness=USE_RANDOMNESS)
    E1_measured = apply_energy_smearing(E1_true, resolution_factor=0.042, 
                                         seed=RANDOM_SEED+1, use_randomness=USE_RANDOMNESS)
    
    if USE_RANDOMNESS:
        print(f"\n  ✓ Applied random energy smearing (seed={RANDOM_SEED})")
    else:
        print(f"\n  ✓ Applied deterministic +1σ smearing")
    
    print(f"\nMeasured energy statistics (after smearing):")
    print(f"  Photon 0 - Mean: {np.mean(E0_measured):.3f} GeV, Range: [{np.min(E0_measured):.3f}, {np.max(E0_measured):.3f}] GeV")
    print(f"  Photon 1 - Mean: {np.mean(E1_measured):.3f} GeV, Range: [{np.min(E1_measured):.3f}, {np.max(E1_measured):.3f}] GeV")
    
    # Compute invariant mass
    print(f"\n{'='*60}")
    print("Compute invariant mass of π⁰...")
    print(f"Theoretical π⁰ mass: 0.135 GeV/c²")
    print('='*60)
    
    invariant_mass = calculate_invariant_mass(gamma_0_accepted, gamma_1_accepted, 
                                               E0_measured, E1_measured)
    
    print(f"\nInvariant mass statistics:")
    print(f"  Mean:   {np.mean(invariant_mass):.4f} GeV/c²")
    print(f"  Median: {np.median(invariant_mass):.4f} GeV/c²")
    print(f"  Std. dev.:  {np.std(invariant_mass):.4f} GeV/c²")
    print(f"  Range:    [{np.min(invariant_mass):.4f}, {np.max(invariant_mass):.4f}] GeV/c²")
    
    # Create histogram and perform Gaussian fit
    print(f"\n{'='*60}")
    print("Create mass histogram and determine resolution...")
    print('='*60)
    
    fit_params = plot_mass_histogram(invariant_mass, decay_process)
    
    print(f"\n{'='*60}")
    print("RESULT TASK B): Mass resolution")
    print('='*60)
    print(f"Reconstructed π⁰ mass:       {fit_params['mean']*1000:.2f} MeV/c²")
    print(f"Theoretical π⁰ mass:         135.0 MeV/c²")
    print(f"Deviation:                   {(fit_params['mean']*1000 - 135.0):.2f} MeV/c²")
    print(f"\nMass resolution (σ):         {fit_params['sigma']*1000:.2f} MeV/c²")
    print(f"Relative resolution (σ/μ):   {fit_params['resolution']:.2f}%")


def main():
    """
    Main function: analyze both decay processes.
    """
    # List of decay processes to analyze
    processes = ['B+', 'D0']
    
    for i, process in enumerate(processes, 1):
        print("\n" + "#"*70)
        print(f"### ANALYZE DECAY PROCESS {i}/2: {process}")
        print("#"*70)
        
        analyze_decay(process)
        
        # Separator between processes (except after the last)
        if i < len(processes):
            print("\n" + "="*70)
            print("="*70)
            print("\n")


if __name__ == "__main__":
    main()
