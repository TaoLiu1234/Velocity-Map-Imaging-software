# Measurement principle and hardware

This software was built for a **PEPICO** experiment (photoelectron–photoion
coincidence): single photoionization events from a VUV photon source are
detected in coincidence — the photoelectron on a velocity-map imaging (VMI)
spectrometer, and the photoion on a time-resolved ion detector. Everything
the software computes follows from that coincidence principle.

## The spectrometer

- **Photon source.** A VUV lamp (helium lamp, as reflected in the data-file
  names, e.g. `...lampon_Ar_6.1bar_helium lamp pressure_8.3E-1_Electric
  field_100v_cm...`) ionizes the molecular beam. The same analysis applies to
  any pulsed or continuous single-photon source.
- **Electron side — VMI.** The extraction stack is tuned through the software's
  voltage panel (**V_Repeller, V_Extractor, V_E.Grid, V_I.Grid, V_Detector**,
  plus the extraction field **F (V/cm)**, **Lens L** and an optional
  **V_offset**). The VMI lens projects the electron Newton sphere onto a
  position-sensitive detector: the impact position `(x, y)` encodes the
  electron's speed (radius) and emission angle (azimuth). Because the lens is
  voltage-tuned, the speed scale depends on the settings — hence the
  software's F/Lens/V_offset calibration inputs and the kinetic-energy axis
  conversion on the radial profile.
- **Ion side — TOF.** The photoion's time of flight (ns to µs scale) is
  measured against the same trigger, giving the mass-to-charge ratio through
  the square law `m/q = a·t²` (calibrated in the Ion Histogram tab). The ion
  detector records arrival position `(x, y)` as well, which the software uses
  for spatial background rejection (the rectangular filter and the Ion X/Y-TOF
  map), rotation alignment, and TOF-centering.

## Coincidence and the data files

The acquisition (TDC) writes three headerless CSV files per measurement
(the file names carry the sample, pressure, and field metadata):

| file | columns | meaning |
|---|---|---|
| `*_DAn.dat` (trigger) | `event_no, ion_tof, ion_index, electron_index` | the per-cycle trigger table linking electron and ion events |
| `*.lmf_elec_DAn.dat` | `x, y, t` | electron detector hits (VMI position + time) |
| `*.lmf_ion_DAn.dat` | `x, y, t` | ion detector hits (position + TOF) |

`NaN` marks a missing channel (e.g. an electron without its ion partner).
One physical event may appear across all three files; the software's
"trigger modes" (1e+1i, 1e+2i, 1e+3i, all-valid) convert the trigger table
into electron–ion pairs, which is the statistical heart of PEPICO: true
pairs survive the coincidence selection, while false coincidences (ions
from other cycles) are removed by the TOF background model and spatial
filters.

## How the workflow maps onto the hardware

| Software step | Hardware quantity |
|---|---|
| Load + trigger pairing | the TDC coincidence table |
| Ion histogram ROI + m/q calibration | ion TOF -> mass-to-charge |
| TOF background model / ion rect filter | false-coincidence + spatial rejection |
| Electron scatter + center estimation | VMI image Newton sphere centering |
| Apply ring selection + denoised binning | velocity-space projection |
| rBasex / other Abel inversion | recovered speed distribution I(r) and anisotropy beta(r) |
| Voltage panel (F, Lens, V_*) | the VMI lens calibration used for the energy axis |

See `docs/science.md` for the algorithms and their assumptions, and
`docs/user-guide.md` for the operating procedure.
