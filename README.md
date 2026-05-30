# Fast browser-animated N-body Streamlit app

This Streamlit app visualizes a toy 3D N-body gravitational problem.

The left panel integrates Newtonian gravity. The right panel integrates Newtonian gravity plus a pairwise two-body first post-Newtonian (1PN) correction inspired by General Relativity.

This version is optimized for Streamlit Community Cloud:

- it does **not** use `streamlit-autorefresh`;
- all expensive controls are inside a Streamlit form;
- moving sliders does not immediately recompute the simulation;
- click **Apply and recompute** to apply changed parameters;
- trajectories are computed once and cached;
- trajectory curves are drawn as static downsampled 3D lines;
- Plotly animation updates only the body marker positions in the browser;
- visible **Play / Pause / Reset** buttons are placed above the Plotly figure;
- the 3D view box can be fixed, fitted to the full computed trajectory, or dynamically auto-fitted during playback.

## Axis scaling modes

The sidebar control **View-box scaling mode** has three options:

- **Fixed by slider**: the 3D axes use the selected fixed half-width and do not change during playback. This is best for stable manual zoom, pan, and rotation.
- **Fit full computed trajectory**: the app chooses one constant box large enough for the whole computed trajectory.
- **Dynamic auto-fit during playback**: the box is recomputed from the current body positions during the Plotly animation, so the displayed region can expand or shrink as the system evolves.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload these files to a GitHub repository and deploy `app.py` on Streamlit Community Cloud:

```text
app.py
requirements.txt
README.md
```

## Notes

The model is educational. It is not a full Einstein--Infeld--Hoffmann N-body ephemeris and not a numerical-relativity calculation.
