# Homework 4

## Calorimeter properties

You are asked to instrument an electromagnetic calorimeter 12 m downstream of an interaction point at the LHC in the pseudo-rapidity range of 2.5 to 3. A colleague simulated the processes  
**B⁺ → π⁰π⁺** and **D⁰ → K⁺K⁻π⁻π⁺π⁰**, with **π⁰ → γγ** for you and gave you access to csv files containing 4-vectors of the photons (**⃗p, E in GeV**). Assume that all photons originate from (0,0,0):

- File 1 download link (click on Cancel, then download in the middle of the page)  
- File 2 download link

**a)** Which is the maximum (quadratic) cell size, such that in at least 68.3% of the cases, the photons from the π⁰ decay end up in separate cells? Check both processes and make histograms for the opening angle between photons and their distance on the calorimeter surface.

**b)** What is the mass resolution of the π⁰ peaks given the reconstructed results of a) and an ECAL resolution of σ_E/E = 4.2%/√E[GeV].  
Hint: in order to emulate the reconstruction step, apply a Gaussian smearing to the simulated energy of an accepted candidate. Prepare a histogram that shows the π⁰ mass peak.

**c)** Comment on the results. Do you think it would be possible to build both calorimeters? Think about, and name at least 4 typical constraints, and technical challenges that you would face when trying to build the calorimeter. Remember to cite your sources.

---

## Code Implementation and Methodology

### Overview

The analysis is performed using Python with the following key libraries:

- **NumPy**: Numerical computations and array operations
- **Pandas**: CSV data loading and manipulation
- **Matplotlib**: Visualization and histogram generation
- **SciPy**: Gaussian fitting for mass peak analysis

### Configuration

The code includes configurable parameters at the top:

```python
USE_RANDOMNESS = True  # Toggle between random and deterministic energy smearing
RANDOM_SEED = 42       # Ensures reproducibility of results
```

### Analysis Workflow

#### 1. Data Loading

The CSV files contain 4-vectors (px, py, pz, E) for both photons from each π⁰ decay. For each decay process:

- **B⁺ → π⁰π⁺**: 13,876 simulated events
- **D⁰ → K⁺K⁻π⁻π⁺π⁰**: 13,381 simulated events

#### 2. Opening Angle Calculation

The opening angle θ between two photons is calculated using the scalar product of their momentum vectors:

$$\cos(\theta) = \frac{\vec{p}_0 \cdot \vec{p}_1}{E_0 \cdot E_1}$$

For photons (massless particles), the momentum magnitude equals the energy: |**p⃗**| = E.

#### 3. Calorimeter Position Projection

Photons originating from (0,0,0) are projected onto the calorimeter surface at z = 12 m using ray tracing:

$$x_{\text{calo}} = \frac{12 \cdot p_x}{p_z}, \quad y_{\text{calo}} = \frac{12 \cdot p_y}{p_z}$$

The distance between two photons on the calorimeter is:

$$d = \sqrt{(x_1 - x_0)^2 + (y_1 - y_0)^2}$$

#### 4. Cell Size Determination

The maximum cell size is determined by the 31.7% quantile of the distance distribution. This ensures that in at least 68.3% of cases, the two photons hit separate cells (distance > cell size). The 31.7% quantile means that 31.7% of events have distances smaller than the cell size (photons in the same cell), while 68.3% have larger distances (photons in separate cells).

#### 5. Event Selection

Events are accepted if the distance between photons exceeds the cell size (photons in separate cells). This selection is crucial for proper mass reconstruction.

#### 6. Energy Smearing

To simulate realistic detector resolution, Gaussian energy smearing is applied:

$$\sigma_E = \frac{4.2\%}{\sqrt{E[\text{GeV}]}} \cdot E = 0.042 \cdot \sqrt{E}$$

For each photon energy, a random error is drawn from N(0, σ_E) and added to the true energy. The fixed random seed ensures reproducibility.

#### 7. Invariant Mass Reconstruction

The invariant mass is calculated using the relativistic formula:

$$m_{\text{inv}} = \sqrt{(E_1 + E_2)^2 - |\vec{p}_1 + \vec{p}_2|^2}$$

The momentum vectors are reconstructed using the measured energies while preserving the original directions.

#### 8. Mass Resolution Analysis

A Gaussian fit is applied to the mass distribution histogram to extract:

- **μ**: Peak position (reconstructed π⁰ mass)
- **σ**: Peak width (absolute mass resolution)
- **σ/μ**: Relative mass resolution

---

## Answer to Task a): Maximum Cell Size

### B⁺ → π⁰π⁺ Decay

**Results:**

- **Maximum cell size**: 11.77 cm × 11.77 cm (31.7% quantile)
- **Acceptance rate**: 68.3% of events (9,477 out of 13,876)

**Opening Angle Statistics:**

- Mean: 1.04°
- Median: 0.78°
- Range: [0.05°, 15.97°]

**Distance on Calorimeter Statistics:**

- Mean: 22.13 cm
- Median: 16.52 cm
- Range: [0.98 cm, 336.60 cm]

See histogram: `histograms_opening_angle_distance_B+.png`

### D⁰ → K⁺K⁻π⁻π⁺π⁰ Decay

**Results:**

- **Maximum cell size**: 85.60 cm × 85.60 cm (31.7% quantile)
- **Acceptance rate**: 68.3% of events (9,138 out of 13,379)

**Opening Angle Statistics:**

- Mean: 7.00°
- Median: 6.08°
- Range: [0.17°, 18.70°]

**Distance on Calorimeter Statistics:**

- Mean: 147.86 cm
- Median: 128.35 cm
- Range: [3.65 cm, 395.16 cm]

See histogram: `histograms_opening_angle_distance_D0.png`

### Interpretation

The cell size is chosen as the 31.7% quantile of the distance distribution, which ensures that 68.3% of photon pairs have separations larger than the cell size and thus hit separate cells. This threshold corresponds roughly to one standard deviation (±1σ) in a Gaussian distribution.

The calorimeter design accepts that approximately 31.7% of events will have photons landing in the same cell (distance < cell size), making them unresolvable. These events must be rejected for mass reconstruction. The 68.3% threshold balances detector granularity (cost, complexity) with event acceptance rate.

**Comparison of decay processes:**

- **B⁺ decay**: Small opening angles (median 0.78°) result in small separations (median 16.52 cm), requiring fine granularity (11.77 cm cells)
- **D⁰ decay**: Larger opening angles (median 6.08°) result in larger separations (median 128.35 cm), allowing coarser granularity (85.60 cm cells)

The D⁰ decay has approximately 7.3 times larger cell size than B⁺, making it significantly easier and cheaper to instrument.

---

## Answer to Task b): Mass Resolution

### B⁺ → π⁰π⁺ Decay

**Mass Reconstruction Results:**

- **Reconstructed π⁰ mass (μ)**: 135.00 MeV/c²
- **Theoretical π⁰ mass**: 135.0 MeV/c²
- **Deviation**: -0.00 MeV/c²

**Mass Resolution:**

- **Absolute resolution (σ)**: 1.69 MeV/c²
- **Relative resolution (σ/μ)**: 1.25%

See histogram: `histogram_pion_mass_B+.png`

### D⁰ → K⁺K⁻π⁻π⁺π⁰ Decay

**Mass Reconstruction Results:**

- **Reconstructed π⁰ mass (μ)**: 135.02 MeV/c²
- **Theoretical π⁰ mass**: 135.0 MeV/c²
- **Deviation**: +0.02 MeV/c²

**Mass Resolution:**

- **Absolute resolution (σ)**: 4.48 MeV/c²
- **Relative resolution (σ/μ)**: 3.32%

See histogram: `histogram_pion_mass_D0.png`

### Interpretation

The mass resolution is determined by:

1. **Detector energy resolution**: σ_E/E = 4.2%/√E[GeV], which dominates for high-energy photons
2. **Geometric effects**: Uncertainty in photon direction affects momentum reconstruction
3. **Event selection**: Only events with well-separated photons (distance > cell size) are used

The Gaussian fit to the invariant mass distribution provides a precise measurement of both the peak position and width. Both decay processes reconstruct the π⁰ mass extremely accurately, with deviations < 0.02 MeV/c² from the theoretical value of 135.0 MeV/c².

**Comparison of mass resolution:**

- **B⁺ decay**: σ = 1.69 MeV/c², σ/μ = 1.25% (better resolution due to higher photon energies, mean ~10 GeV)
- **D⁰ decay**: σ = 4.48 MeV/c², σ/μ = 3.32% (worse resolution due to lower photon energies, mean ~1.3 GeV)

The relative resolution follows the expected 1/√E dependence: higher energy photons in B⁺ decay lead to approximately 2.7× better resolution than in D⁰ decay. Both values are within the typical range of 1-5% for electromagnetic calorimeters in this energy regime.

---

## Answer to Task c): Feasibility and Technical Challenges

[TO BE FILLED]

---

## Use of Generative AI

[TO BE FILLED]

---

## Authors

[TO BE FILLED]

---

## License

[TO BE FILLED]

---
