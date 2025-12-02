# Directory Structure

`/EngineLab/프로젝트2020/2025 Aramco America/1124Showing_Filter`

- `100bar/`
  - Time-slice folders (`250us`–`5000us`) and `Shifted_J5_summary.xlsx`.
  - Each time-slice folder contains four `.lvm` variants: raw, `BCswap_Czero`, `BCswap_Czero_fftSmooth`, and `BCswap_Czero_fftSmooth_shifted`.
- `150bar/`
  - Single set of `Data_020*.lvm` files covering the same processing stages.
- `200bar/`
  - Time-slice folders mirroring the `100bar` layout plus `Shifted_J5_summary.xlsx` and a temporary Excel lock file (`~$Shifted_J5_summary.xlsx`).
- `250bar/` / `350bar/`
  - Each holds one quartet of `.lvm` files (`Data_001*`) for the four processing stages.
- `300bar/`
  - Time-slice folders (`250us`–`5000us`) identical in structure to `100bar`, plus `Shifted_J5_summary.xlsx`.
- `300bar - 복사본/`
  - Subset of time-slice folders (`300us`, `600us`, `900us`, `1500us`, `2000us`) and `Shifted_J5_summary.xlsx`.
- Root-level Excel files:
  - `300to2000.xlsx`, `Cd.xlsx`, and their temporary lock files (`~$300to2000.xlsx`, `~$Cd.xlsx`).

---

## Processing Pipeline

Each `.lvm` time-slice typically tracks four processing stages (raw → `BCswap_Czero` → `BCswap_Czero_fftSmooth` → `…_shifted`), consistent across pressure folders.

These `.lvm` files contain experimental data for **ROI (Rate of Injection) measurement using the Bosch method**.

---

## 1. Raw Data (`Data_001.lvm`)

The original unprocessed data with **4 columns**:

| Column | Description | Unit/Notes |
|--------|-------------|------------|
| 1 | Time | [s] |
| 2 | Pressure sensor Voltage | [0-10V] (can be converted to [0-50 bar] range for reference) |
| 3 | TTL input voltage | ~0V (low) / ~5V (high) |
| 4 | Injector input Current | [A] |

**Note:** Column 4 (Current data) may need processing to align the baseline to 0A when there's no signal, to match the provided injector sheet.

---

## 2. BCswap_Czero (`Data_001_BCswap_Czero.lvm`)

Processing steps:
1. **Column swap:** Swap columns 2 and 3 (Pressure and TTL)
2. **Zero offset correction:** Subtract the first-row value from all values in the new column 3 (pressure data) so that the first row becomes 0

Resulting **4-column structure**:

| Column | Description | Unit/Notes |
|--------|-------------|------------|
| 1 | Time | [s] |
| 2 | TTL input voltage | ~0V (low) / ~5V (high) |
| 3 | Pressure sensor Voltage (zero-corrected) | [0-10V] (pressure unit conversion not determined) |
| 4 | Injector input Current | [A] |

---

## 3. BCswap_Czero_fftSmooth (`Data_001_BCswap_Czero_fftSmooth.lvm`)

Adds FFT-based Low Pass Filtering to the pressure data (column 3). Contains **7 columns**:

| Column | Description | Unit/Notes |
|--------|-------------|------------|
| 1-4 | Same as `BCswap_Czero` | |
| 5 | Weak Filter (10kHz) | FFT filtering of column 3 |
| 6 | Strong Filter (3kHz) | FFT filtering of column 3 |
| 7 | Filter Combined | Combined filter result |
| 8 | Empty | |

### Filter Combination Logic:

The combined filter (column 7) uses a **two-region approach** based on the first peak of each filtered signal (where peaks are determined from integrated pressure data):

- **Transient region (initial phase):** Uses **Weak Filter (10kHz)** values up to the first peak of the weak filter (this typically occurs before the strong filter's first peak)
- **Steady-state region:** Uses **Strong Filter (3kHz)** values after the first peak of the strong filter
- **Transition region:** Between the two peaks, an interpolation is applied, following weak filter values initially and transitioning to strong filter values later

---

## 4. BCswap_Czero_fftSmooth_shifted (`Data_001_BCswap_Czero_fftSmooth_shifted.lvm`)

Time-shifted and analysis data added. Contains **11 columns**:

| Column | Description | Unit/Notes |
|--------|-------------|------------|
| 1-8 | Time-shifted data (corresponding to columns 1-8 of `BCswap_Czero_fftSmooth`) | |

**Time shifting process:**
- The TTL signal (column 2) trigger time is set as time = 0
- All data before the trigger is truncated to keep only up to -0.005 s (or -0.0005 s in some cases)
- All column data is shifted accordingly
- Example: If TTL signal occurred at 1.0 s, subtract 1.0 s from all time data, then keep only data from -0.005 s onward

**Analysis columns:**

| Column | Description | Formula/Notes |
|--------|-------------|---------------|
| 9 | Parameter label | One of 5 labels (see below) |
| 10 | Parameter value | Numerical value |
| 11 | Mass flow rate | [mg/ms] = $\frac{A}{c} \cdot p'$ (uses column 7 pressure) |

### Parameter labels (column 9) and values (column 10):

1. **Start 1:** Pressure wave start time ($t_0$)
2. **Start 2:** Returned pressure wave time ($t_e$)
3. **S.Speed:** Sound speed [m/s] = $c = \frac{2L}{t_e - t_0}$ (Round-trip/two-way travel-time speed-of-sound relation)
4. **A/C:** Area divided by sound speed = $\frac{A}{c}$
5. **∫Kdt(mg):** Total injected mass [mg] = $\frac{A}{c} \int^{t_e}_{t_0} p'(t) \, dt = m$
   - where $p'$ is the pressure fluctuation (column 7)

---

