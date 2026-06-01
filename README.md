# Interactive N-body problem: Newton vs. 1PN

Streamlit web application for an educational three-body / N-body gravity playground.

The app compares:

- **Newton gravity**
- **Einstein GTR 1PN approximation** using a pairwise two-body first post-Newtonian correction

It is designed for teaching and exploration. It is not a precision ephemeris and not a full Einstein-Infeld-Hoffmann numerical-relativity solver.

## Files

```text
app.py
requirements.txt
README.md
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create or open a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. In Streamlit Community Cloud create or reboot the app.
4. Set the main file path to `app.py`.

## Performance notes

This version includes two speed improvements:

1. The Newtonian acceleration is vectorized with NumPy broadcasting instead of a Python double loop.
2. Visual-only controls are outside the expensive **Apply and recompute** form. Changing marker sizes, view-box scaling, animation-frame count, or trajectory curve density does not trigger a new numerical integration. Only physical/numerical changes such as masses, initial positions, velocities, `G`, `dt`, simulation time, `c`, or the 1PN multiplier require **Apply and recompute**.

The app also supports:

- progressive trajectory drawing,
- animated Plotly playback,
- GIF export,
- TXT simulation protocol export,
- English/Czech interface,
- fixed/full/dynamic 3D view-box scaling.
