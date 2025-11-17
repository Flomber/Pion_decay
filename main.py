import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit


def load_photon_data(decay_process='B+'):
    """
    Lädt Photonendaten aus den CSV-Dateien.
    
    Parameters:
    -----------
    decay_process : str
        'B+' für B⁺ → π⁰π⁺ Zerfall
        'D0' für D⁰ → K⁺K⁻π⁻π⁺π⁰ Zerfall
    
    Returns:
    --------
    gamma_0 : numpy array (N, 4)
        4-Vektoren des ersten Photons [px, py, pz, E]
    gamma_1 : numpy array (N, 4)
        4-Vektoren des zweiten Photons [px, py, pz, E]
    """
    # Dateipfade basierend auf dem Zerfallsprozess
    if decay_process == 'B+':
        filename = 'photons_from_BpToPipPiz.csv'
        print(f"Lade Daten für B⁺ → π⁰π⁺ Zerfall...")
    elif decay_process == 'D0':
        filename = 'photons_from_D0ToKmKpPimPipPiz.csv'
        print(f"Lade Daten für D⁰ → K⁺K⁻π⁻π⁺π⁰ Zerfall...")
    else:
        raise ValueError("decay_process muss 'B+' oder 'D0' sein")
    
    # Daten einlesen
    filepath = Path(__file__).parent / filename
    df = pd.read_csv(filepath)
    
    # Extrahiere 4-Vektoren für beide Photonen
    gamma_0 = df[['gamma_0_PX_TRUE', 'gamma_0_PY_TRUE', 
                   'gamma_0_PZ_TRUE', 'gamma_0_E_TRUE']].values
    gamma_1 = df[['gamma_1_PX_TRUE', 'gamma_1_PY_TRUE', 
                   'gamma_1_PZ_TRUE', 'gamma_1_E_TRUE']].values
    
    print(f"  Anzahl Events: {len(df)}")
    print(f"  Gamma 0 shape: {gamma_0.shape}")
    print(f"  Gamma 1 shape: {gamma_1.shape}")
    
    return gamma_0, gamma_1


def calculate_opening_angle(gamma_0, gamma_1):
    """
    Berechnet den Öffnungswinkel zwischen zwei Photonen.
    
    Parameters:
    -----------
    gamma_0, gamma_1 : numpy array (N, 4) mit [px, py, pz, E]
    
    Returns:
    --------
    theta : numpy array (N,) - Öffnungswinkel in Radiant
    """
    # Impulsvektoren (erste 3 Komponenten)
    p0 = gamma_0[:, :3]
    p1 = gamma_1[:, :3]
    
    # Energien (4. Komponente)
    E0 = gamma_0[:, 3]
    E1 = gamma_1[:, 3]
    
    # Skalarprodukt der Impulse
    dot_product = np.sum(p0 * p1, axis=1)
    
    # cos(theta) = (p0 · p1) / (|p0| * |p1|) = (p0 · p1) / (E0 * E1)
    cos_theta = dot_product / (E0 * E1)
    
    # Prüfe auf Werte außerhalb [-1, 1] und gib Warnung aus
    out_of_bounds = (cos_theta < -1) | (cos_theta > 1)
    if np.any(out_of_bounds):
        indices = np.where(out_of_bounds)[0]
        print(f"\n⚠️  WARNUNG: Numerische Rundungsfehler bei {len(indices)} Events!")
        print(f"   Clip-Funktion wird angewendet auf folgende Events:")
        for idx in indices[:10]:  # Zeige max. 10 Beispiele
            print(f"   - Event {idx}: cos(theta) = {cos_theta[idx]:.15f}")
        if len(indices) > 10:
            print(f"   ... und {len(indices) - 10} weitere Events")
    
    # Numerische Stabilität: clip auf [-1, 1]
    cos_theta = np.clip(cos_theta, -1, 1)
    
    # Öffnungswinkel
    theta = np.arccos(cos_theta)
    
    return theta


def calculate_calorimeter_positions(gamma_0, gamma_1, z_calo=12.0):
    """
    Berechnet Positionen auf dem Kalorimeter und Abstände.
    
    Parameters:
    -----------
    gamma_0, gamma_1 : numpy array (N, 4) mit [px, py, pz, E]
    z_calo : float - z-Position des Kalorimeters in Metern
    
    Returns:
    --------
    positions_0 : numpy array (N, 2) - (x, y) Positionen von Photon 0
    positions_1 : numpy array (N, 2) - (x, y) Positionen von Photon 1
    distances : numpy array (N,) - Abstände in Metern
    """
    # Extrahiere Impulskomponenten
    px0, py0, pz0 = gamma_0[:, 0], gamma_0[:, 1], gamma_0[:, 2]
    px1, py1, pz1 = gamma_1[:, 0], gamma_1[:, 1], gamma_1[:, 2]
    
    # Berechne Positionen auf dem Kalorimeter
    # x_calo = z_calo * px / pz
    x0 = z_calo * px0 / pz0
    y0 = z_calo * py0 / pz0
    
    x1 = z_calo * px1 / pz1
    y1 = z_calo * py1 / pz1
    
    # Kombiniere zu Positionsvektoren
    positions_0 = np.column_stack([x0, y0])
    positions_1 = np.column_stack([x1, y1])
    
    # Berechne Abstände
    distances = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    
    return positions_0, positions_1, distances


def plot_histograms(opening_angles, distances, decay_process):
    """
    Erstellt Histogramme für Öffnungswinkel und Abstände.
    
    Parameters:
    -----------
    opening_angles : numpy array - Öffnungswinkel in Radiant
    distances : numpy array - Abstände in Metern
    decay_process : str - Name des Zerfallsprozesses
    
    Returns:
    --------
    quantile_68 : float - 68.3%-Quantil der Abstände in cm
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogramm 1: Öffnungswinkel
    angles_deg = np.degrees(opening_angles)
    ax1.hist(angles_deg, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Öffnungswinkel [°]', fontsize=12)
    ax1.set_ylabel('Anzahl Events', fontsize=12)
    ax1.set_title(f'Öffnungswinkel ({decay_process})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Statistik in Plot einfügen
    mean_angle = np.mean(angles_deg)
    median_angle = np.median(angles_deg)
    ax1.text(0.95, 0.95, f'Mittelwert: {mean_angle:.2f}°\nMedian: {median_angle:.2f}°',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Histogramm 2: Abstände auf Kalorimeter
    distances_cm = distances * 100  # Meter → Zentimeter
    ax2.hist(distances_cm, bins=100, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Abstand auf Kalorimeter [cm]', fontsize=12)
    ax2.set_ylabel('Anzahl Events', fontsize=12)
    ax2.set_title(f'Abstand auf ECAL ({decay_process})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # 68.3% Quantil berechnen und markieren
    quantile_68 = np.percentile(distances_cm, 68.3)
    ax2.axvline(quantile_68, color='red', linestyle='--', linewidth=2,
                label=f'68.3% Quantil: {quantile_68:.2f} cm')
    ax2.legend(fontsize=10)
    
    # Statistik in Plot einfügen
    mean_dist = np.mean(distances_cm)
    median_dist = np.median(distances_cm)
    ax2.text(0.95, 0.95, f'Mittelwert: {mean_dist:.2f} cm\nMedian: {median_dist:.2f} cm',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Speichere Histogramm als PNG
    filename = f'histograms_opening_angle_distance_{decay_process}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Histogramme gespeichert als: {filename}")
    
    plt.show()
    
    return quantile_68


def select_accepted_events(gamma_0, gamma_1, distances, cell_size_cm):
    """
    Selektiert Events, bei denen die Photonen in separate Zellen treffen.
    
    Parameters:
    -----------
    gamma_0, gamma_1 : numpy array (N, 4)
    distances : numpy array (N,) - Abstände in Metern
    cell_size_cm : float - Maximale Zellgröße in cm
    
    Returns:
    --------
    gamma_0_accepted : numpy array (M, 4) - Akzeptierte Photonen 0
    gamma_1_accepted : numpy array (M, 4) - Akzeptierte Photonen 1
    acceptance_rate : float - Anteil der akzeptierten Events
    """
    # Konvertiere Abstände zu cm
    distances_cm = distances * 100
    
    # Boolean Mask: True wenn Abstand > Zellgröße
    accepted_mask = distances_cm > cell_size_cm
    
    # Wende Mask an (selektiert nur True-Einträge)
    gamma_0_accepted = gamma_0[accepted_mask]
    gamma_1_accepted = gamma_1[accepted_mask]
    
    # Berechne Akzeptanzrate
    n_total = len(gamma_0)
    n_accepted = len(gamma_0_accepted)
    acceptance_rate = n_accepted / n_total
    
    print(f"\nEvent-Selektion:")
    print(f"  Zellgröße: {cell_size_cm:.2f} cm")
    print(f"  Total Events: {n_total}")
    print(f"  Akzeptierte Events: {n_accepted}")
    print(f"  Verworfen: {n_total - n_accepted}")
    print(f"  Akzeptanzrate: {acceptance_rate*100:.1f}%")
    
    return gamma_0_accepted, gamma_1_accepted, acceptance_rate


def apply_energy_smearing(energies, resolution_factor=0.042, seed=42):
    """
    Wendet Gauß'sches Energie-Smearing an.
    
    Parameters:
    -----------
    energies : numpy array - Wahre Energien in GeV
    resolution_factor : float - Auflösungsfaktor (4.2% = 0.042)
    seed : int or None - Random seed für Reproduzierbarkeit
    
    Returns:
    --------
    energies_smeared : numpy array - "Gemessene" Energien in GeV
    """
    # Setze Seed für Reproduzierbarkeit
    if seed is not None:
        np.random.seed(seed)
    
    # Berechne Unsicherheit für jede Energie
    # σ_E = resolution_factor × √E
    sigma_E = resolution_factor * np.sqrt(energies)
    
    # Ziehe Zufallsfehler aus Normalverteilung
    # Für jede Energie wird ein individueller Fehler aus N(0, σ_E) gezogen
    random_errors = np.random.normal(0, sigma_E)
    
    # Addiere Fehler zur wahren Energie
    energies_smeared = energies + random_errors
    
    # Prüfe auf negative Energien und gib Warnung aus
    negative_mask = energies_smeared < 0
    if np.any(negative_mask):
        indices = np.where(negative_mask)[0]
        print(f"\n⚠️  WARNUNG: {len(indices)} negative Energien nach Smearing!")
        print(f"   Diese werden auf 1 MeV gesetzt.")
        print(f"   Betroffene Events (max. 10 angezeigt):")
        for idx in indices[:10]:
            print(f"   - Event {idx}: E_true = {energies[idx]:.4f} GeV, "
                  f"Fehler = {random_errors[idx]:+.4f} GeV, "
                  f"E_smeared = {energies_smeared[idx]:.4f} GeV")
        if len(indices) > 10:
            print(f"   ... und {len(indices) - 10} weitere Events")
    
    # Verhindere negative Energien (physikalisch unmöglich)
    energies_smeared = np.maximum(energies_smeared, 0.001)  # Minimum 1 MeV
    
    return energies_smeared


def calculate_invariant_mass(gamma_0, gamma_1, E0_measured, E1_measured):
    """
    Berechnet die invariante Masse zweier Photonen.
    
    Parameters:
    -----------
    gamma_0, gamma_1 : numpy array (N, 4) - 4-Vektoren [px, py, pz, E]
    E0_measured, E1_measured : numpy array (N,) - Gemessene Energien in GeV
    
    Returns:
    --------
    invariant_mass : numpy array (N,) - Invariante Masse in GeV/c²
    """
    # Extrahiere wahre Impulsvektoren (nur Richtung wird verwendet)
    p0_true = gamma_0[:, :3]  # [px, py, pz]
    p1_true = gamma_1[:, :3]
    
    # Berechne Impulsbeträge (sollten = E für Photonen sein)
    p0_mag = np.sqrt(np.sum(p0_true**2, axis=1))
    p1_mag = np.sqrt(np.sum(p1_true**2, axis=1))
    
    # Berechne Einheitsvektoren (Richtungen)
    p0_unit = p0_true / p0_mag[:, np.newaxis]
    p1_unit = p1_true / p1_mag[:, np.newaxis]
    
    # Rekonstruiere Impulse mit gemessenen Energien
    # Für Photonen: |p| = E
    p0_reconstructed = E0_measured[:, np.newaxis] * p0_unit
    p1_reconstructed = E1_measured[:, np.newaxis] * p1_unit
    
    # Berechne Gesamtenergie und Gesamtimpuls
    E_total = E0_measured + E1_measured
    p_total = p0_reconstructed + p1_reconstructed
    
    # Betrag des Gesamtimpulses
    p_total_mag = np.sqrt(np.sum(p_total**2, axis=1))
    
    # Invariante Masse: m² = E² - p²
    mass_squared = E_total**2 - p_total_mag**2
    
    # Prüfe auf negative Werte und gib Warnung aus
    negative_mask = mass_squared < 0
    if np.any(negative_mask):
        indices = np.where(negative_mask)[0]
        print(f"\n⚠️  WARNUNG: {len(indices)} negative Massen-Quadrate!")
        print(f"   Diese werden auf 0 gesetzt (numerische Rundungsfehler).")
        print(f"   Betroffene Events (max. 10 angezeigt):")
        for idx in indices[:10]:
            print(f"   - Event {idx}: E_total² = {E_total[idx]**2:.6f} GeV², "
                  f"p_total² = {p_total_mag[idx]**2:.6f} GeV², "
                  f"m² = {mass_squared[idx]:.6e} GeV²")
        if len(indices) > 10:
            print(f"   ... und {len(indices) - 10} weitere Events")
    
    # Verhindere negative Werte durch numerische Fehler
    mass_squared = np.maximum(mass_squared, 0)
    
    invariant_mass = np.sqrt(mass_squared)
    
    return invariant_mass


def gaussian(x, amplitude, mean, sigma):
    """Gauß-Funktion für Fit."""
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma)**2)


def plot_mass_histogram(invariant_mass, decay_process):
    """
    Erstellt Histogramm der invarianten Masse und fittet Gauß-Kurve.
    
    Parameters:
    -----------
    invariant_mass : numpy array - Invariante Massen in GeV/c²
    decay_process : str - Name des Zerfallsprozesses
    
    Returns:
    --------
    fit_params : dict - Fit-Parameter (mean, sigma, resolution)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogramm erstellen
    counts, bin_edges, patches = ax.hist(invariant_mass, bins=100, 
                                          alpha=0.7, color='blue', 
                                          edgecolor='black', label='Daten')
    
    # Bin-Zentren für Fit berechnen
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Initialer Schätzwert für Fit-Parameter
    max_count = np.max(counts)
    mean_estimate = np.mean(invariant_mass)
    sigma_estimate = np.std(invariant_mass)
    
    try:
        # Gauß-Fit durchführen
        popt, pcov = curve_fit(gaussian, bin_centers, counts,
                                p0=[max_count, mean_estimate, sigma_estimate])
        
        amplitude_fit, mean_fit, sigma_fit = popt
        
        # Fit-Kurve plotten
        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)
        y_fit = gaussian(x_fit, amplitude_fit, mean_fit, sigma_fit)
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Gauß-Fit')
        
        # Berechne relative Auflösung
        resolution = (sigma_fit / mean_fit) * 100  # in Prozent
        
        # Textbox mit Fit-Ergebnissen
        textstr = f'Fit-Ergebnis:\n'
        textstr += f'μ = {mean_fit*1000:.2f} MeV/c²\n'
        textstr += f'σ = {sigma_fit*1000:.2f} MeV/c²\n'
        textstr += f'σ/μ = {resolution:.2f}%\n'
        textstr += f'\nTheoretisch:\n'
        textstr += f'm(π⁰) = 135.0 MeV/c²'
        
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Markiere theoretischen Wert
        ax.axvline(0.135, color='green', linestyle='--', linewidth=2,
                   label='Theoretischer Wert (135 MeV)')
        
        fit_success = True
        fit_params = {
            'mean': mean_fit,
            'sigma': sigma_fit,
            'resolution': resolution,
            'amplitude': amplitude_fit
        }
        
    except Exception as e:
        print(f"\n⚠️  WARNUNG: Gauß-Fit fehlgeschlagen: {e}")
        print(f"   Verwende direkte Statistik statt Fit.")
        fit_success = False
        fit_params = {
            'mean': mean_estimate,
            'sigma': sigma_estimate,
            'resolution': (sigma_estimate / mean_estimate) * 100,
            'amplitude': None
        }
    
    # Achsenbeschriftung und Titel
    ax.set_xlabel('Invariante Masse [GeV/c²]', fontsize=12)
    ax.set_ylabel('Anzahl Events', fontsize=12)
    ax.set_title(f'π⁰-Massenpeak ({decay_process})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Speichere Histogramm als PNG
    filename = f'histogram_pion_mass_{decay_process}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Massenhistogramm gespeichert als: {filename}")
    
    plt.show()
    
    return fit_params


def main():
    # Wähle den Zerfallsprozess
    # 'B+' für B⁺ → π⁰π⁺
    # 'D0' für D⁰ → K⁺K⁻π⁻π⁺π⁰
    decay_process = 'B+'
    
    print("\n" + "="*70)
    print(f"  HOMEWORK 4 - Kalorimeter-Analyse für {decay_process}-Zerfall")
    print("="*70)
    
    # ========== AUFGABE A) ==========
    print("\n" + "#"*70)
    print("#  AUFGABE A) - Maximale Zellgröße bestimmen")
    print("#"*70)
    
    # Lade Photonendaten
    gamma_0, gamma_1 = load_photon_data(decay_process)
    
    print(f"\nErste 3 Events (Photon 0):")
    print(gamma_0[:3])
    print(f"\nErste 3 Events (Photon 1):")
    print(gamma_1[:3])
    
    # Berechne Öffnungswinkel
    print(f"\n{'='*60}")
    print("Berechne Öffnungswinkel zwischen Photonen...")
    print('='*60)
    opening_angles = calculate_opening_angle(gamma_0, gamma_1)
    
    print(f"\nStatistik der Öffnungswinkel:")
    print(f"  Mittelwert: {np.mean(opening_angles):.4f} rad = {np.degrees(np.mean(opening_angles)):.2f}°")
    print(f"  Median:     {np.median(opening_angles):.4f} rad = {np.degrees(np.median(opening_angles)):.2f}°")
    print(f"  Minimum:    {np.min(opening_angles):.4f} rad = {np.degrees(np.min(opening_angles)):.2f}°")
    print(f"  Maximum:    {np.max(opening_angles):.4f} rad = {np.degrees(np.max(opening_angles)):.2f}°")
    
    # Berechne Positionen auf dem Kalorimeter und Abstände
    print(f"\n{'='*60}")
    print("Berechne Positionen auf dem Kalorimeter (z = 12 m)...")
    print('='*60)
    positions_0, positions_1, distances = calculate_calorimeter_positions(gamma_0, gamma_1, z_calo=12.0)
    
    print(f"\nStatistik der Abstände auf dem Kalorimeter:")
    print(f"  Mittelwert: {np.mean(distances):.4f} m = {np.mean(distances)*100:.2f} cm")
    print(f"  Median:     {np.median(distances):.4f} m = {np.median(distances)*100:.2f} cm")
    print(f"  Minimum:    {np.min(distances):.4f} m = {np.min(distances)*100:.2f} cm")
    print(f"  Maximum:    {np.max(distances):.4f} m = {np.max(distances)*100:.2f} cm")
    print(f"\nErste 5 Positionen:")
    for i in range(min(5, len(positions_0))):
        print(f"  Event {i}: γ₀=({positions_0[i,0]:6.3f}, {positions_0[i,1]:6.3f}) m, "
              f"γ₁=({positions_1[i,0]:6.3f}, {positions_1[i,1]:6.3f}) m, "
              f"Abstand={distances[i]:.4f} m")
    
    # Erstelle Histogramme
    print(f"\n{'='*60}")
    print("Erstelle Histogramme...")
    print('='*60)
    quantile_68_cm = plot_histograms(opening_angles, distances, decay_process)
    
    print(f"\n{'='*60}")
    print("ERGEBNIS AUFGABE A): Maximale Zellgröße")
    print('='*60)
    print(f"68.3%-Quantil der Abstände: {quantile_68_cm:.2f} cm")
    print(f"\nMaximale quadratische Zellgröße: {quantile_68_cm:.2f} cm × {quantile_68_cm:.2f} cm")
    print(f"Damit treffen in mindestens 68.3% der Fälle die beiden Photonen")
    print(f"aus dem π⁰-Zerfall in separate Zellen.")
    
    # ========== AUFGABE B) ==========
    print("\n" + "#"*70)
    print("#  AUFGABE B) - Massenauflösung des π⁰")
    print("#"*70)
    
    # Selektiere akzeptierte Events
    print(f"\n{'='*60}")
    print("Selektiere akzeptierte Events...")
    print('='*60)
    gamma_0_accepted, gamma_1_accepted, acceptance_rate = select_accepted_events(
        gamma_0, gamma_1, distances, quantile_68_cm
    )
    
    # Wende Energie-Smearing an
    print(f"\n{'='*60}")
    print("Wende Energie-Smearing an (Detektorauflösung simulieren)...")
    print(f"Auflösung: σ_E/E = 4.2%/√E[GeV]")
    print('='*60)
    
    # Extrahiere wahre Energien
    E0_true = gamma_0_accepted[:, 3]
    E1_true = gamma_1_accepted[:, 3]
    
    print(f"\nStatistik wahre Energien (akzeptierte Events):")
    print(f"  Photon 0 - Mittelwert: {np.mean(E0_true):.3f} GeV, Bereich: [{np.min(E0_true):.3f}, {np.max(E0_true):.3f}] GeV")
    print(f"  Photon 1 - Mittelwert: {np.mean(E1_true):.3f} GeV, Bereich: [{np.min(E1_true):.3f}, {np.max(E1_true):.3f}] GeV")
    
    # Wende Smearing an (mit festem Seed für Reproduzierbarkeit)
    E0_measured = apply_energy_smearing(E0_true, resolution_factor=0.042, seed=42)
    E1_measured = apply_energy_smearing(E1_true, resolution_factor=0.042, seed=43)
    
    print(f"\nStatistik gemessene Energien (nach Smearing):")
    print(f"  Photon 0 - Mittelwert: {np.mean(E0_measured):.3f} GeV, Bereich: [{np.min(E0_measured):.3f}, {np.max(E0_measured):.3f}] GeV")
    print(f"  Photon 1 - Mittelwert: {np.mean(E1_measured):.3f} GeV, Bereich: [{np.min(E1_measured):.3f}, {np.max(E1_measured):.3f}] GeV")
    
    # Berechne invariante Masse
    print(f"\n{'='*60}")
    print("Berechne invariante Masse des π⁰...")
    print(f"Theoretische π⁰-Masse: 0.135 GeV/c²")
    print('='*60)
    
    invariant_mass = calculate_invariant_mass(gamma_0_accepted, gamma_1_accepted, 
                                               E0_measured, E1_measured)
    
    print(f"\nStatistik invariante Masse:")
    print(f"  Mittelwert: {np.mean(invariant_mass):.4f} GeV/c²")
    print(f"  Median:     {np.median(invariant_mass):.4f} GeV/c²")
    print(f"  Std.-Abw.:  {np.std(invariant_mass):.4f} GeV/c²")
    print(f"  Bereich:    [{np.min(invariant_mass):.4f}, {np.max(invariant_mass):.4f}] GeV/c²")
    
    # Erstelle Histogramm und führe Gauß-Fit durch
    print(f"\n{'='*60}")
    print("Erstelle Massenhistogramm und bestimme Auflösung...")
    print('='*60)
    
    fit_params = plot_mass_histogram(invariant_mass, decay_process)
    
    print(f"\n{'='*60}")
    print("ERGEBNIS AUFGABE B): Massenauflösung")
    print('='*60)
    print(f"Rekonstruierte π⁰-Masse:     {fit_params['mean']*1000:.2f} MeV/c²")
    print(f"Theoretische π⁰-Masse:       135.0 MeV/c²")
    print(f"Abweichung:                  {(fit_params['mean']*1000 - 135.0):.2f} MeV/c²")
    print(f"\nMassenauflösung (σ):         {fit_params['sigma']*1000:.2f} MeV/c²")
    print(f"Relative Auflösung (σ/μ):    {fit_params['resolution']:.2f}%")


if __name__ == "__main__":
    main()
