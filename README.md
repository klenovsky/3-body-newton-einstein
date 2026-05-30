# Interactive N-body problem: Newton gravity vs. Einstein GTR 1PN approximation

This is a Streamlit web application for an educational 3D N-body gravitational model.

The application compares:

- **Newton gravity** in the left panel,
- **Einstein GTR 1PN approximation** in the right panel.

The right panel uses a pairwise two-body first post-Newtonian correction. It is useful for illustrating weak relativistic corrections, but it is **not** a full Einstein-Infeld-Hoffmann N-body ephemeris and not numerical relativity.

## Features

- English / Czech interface.
- Presets:
  - figure-eight three-body orbit,
  - Lagrange equilateral triple,
  - binary plus intruder,
  - toy N-body disk.
- Adjustable number of bodies, up to 8.
- Individual sliders/inputs for masses, initial positions, and initial velocities.
- Adjustable `G`, softening length, speed of light `c`, and 1PN multiplier.
- Live playback with Start / Pause / Reset time.
- Optional Plotly Play button.
- Fixed 3D axis box controlled by the view-box slider.
- Mathematical and physical explanation inside the app, including references.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. Open Streamlit Community Cloud.
4. Click `Create app`.
5. Select the repository, branch, and `app.py` as the entrypoint.
6. Deploy.

## Notes

The model uses arbitrary units:

```text
length = L0
time   = T0
mass   = M0
```

The speed of light is expressed in `L0/T0`. The default value `c = 100 L0/T0` is not meant to represent a particular physical system; it is chosen so that the 1PN correction can be made visible with the 1PN multiplier.

For quantitatively realistic post-Newtonian celestial mechanics one must use the full EIH equations and consistent initial conditions. This app intentionally keeps the model simpler for interactive education.
