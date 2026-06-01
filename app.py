#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit web application: educational N-body dynamics.

The app compares Newtonian N-body gravity with a pairwise two-body 1PN
post-Newtonian correction inspired by general relativity.  It is intended for
interactive teaching and exploration, not for precision celestial mechanics or
full numerical relativity.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =============================================================================
# Defaults, translations, presets
# =============================================================================

MAX_BODIES = 8
BODY_COLORS = [
    "royalblue", "crimson", "seagreen", "orange", "purple", "sienna", "cyan", "magenta"
]

DEFAULTS = {
    "language": "English",
    "preset": "Figure-eight 3-body orbit",
    "n_bodies": 3,
    "g_value": 1.0,
    "softening": 1.0e-3,
    "total_time": 12.7,
    "dt": 0.005,
    "frame_stride": 2,
    "trail_frames": 400,
    "axis_half_range": 1.5,
    "axis_scaling": "fixed",
    "marker_base": 7.0,
    "marker_mass_gamma": 0.35,
    "log10_c": 2.0,
    "pn_log10": 0.0,
    "max_animation_frames": 120,
    "animation_frame_duration": 35,
    "orbit_curve_points": 900,
    "export_gif_frames": 80,
}

# Default body values are overwritten by the default preset during initialization.
for i in range(MAX_BODIES):
    DEFAULTS[f"m_{i}"] = 1.0 if i < 3 else 0.2
    DEFAULTS[f"x_{i}"] = 0.0
    DEFAULTS[f"y_{i}"] = 0.0
    DEFAULTS[f"z_{i}"] = 0.0
    DEFAULTS[f"vx_{i}"] = 0.0
    DEFAULTS[f"vy_{i}"] = 0.0
    DEFAULTS[f"vz_{i}"] = 0.0


@dataclass(frozen=True)
class Preset:
    n: int
    masses: tuple[float, ...]
    positions: tuple[tuple[float, float, float], ...]
    velocities: tuple[tuple[float, float, float], ...]
    g: float
    c: float
    total_time: float
    dt: float
    frame_stride: int
    trail_frames: int
    axis_half_range: float
    note_en: str
    note_cs: str


def lagrange_preset() -> Preset:
    """Three equal bodies on the Lagrange equilateral solution."""
    r = 1.0
    m = 1.0
    g = 1.0
    omega = math.sqrt(g * m / math.sqrt(3.0))
    positions = []
    velocities = []
    for k in range(3):
        theta = 2.0 * math.pi * k / 3.0
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        positions.append((x, y, 0.0))
        velocities.append((-omega * y, omega * x, 0.0))
    return Preset(
        n=3,
        masses=(1.0, 1.0, 1.0),
        positions=tuple(positions),
        velocities=tuple(velocities),
        g=g,
        c=100.0,
        total_time=16.0,
        dt=0.01,
        frame_stride=2,
        trail_frames=500,
        axis_half_range=1.6,
        note_en="Three equal masses form a rotating equilateral triangle around their common center of mass.",
        note_cs="Tři stejné hmotnosti tvoří rotující rovnostranný trojúhelník kolem společného těžiště.",
    )


PRESETS: dict[str, Preset] = {
    "Figure-eight 3-body orbit": Preset(
        n=3,
        masses=(1.0, 1.0, 1.0),
        positions=(
            (-0.97000436, 0.24308753, 0.0),
            (0.97000436, -0.24308753, 0.0),
            (0.0, 0.0, 0.0),
        ),
        velocities=(
            (0.4662036850, 0.4323657300, 0.0),
            (0.4662036850, 0.4323657300, 0.0),
            (-0.9324073700, -0.8647314600, 0.0),
        ),
        g=1.0,
        c=100.0,
        total_time=12.7,
        dt=0.005,
        frame_stride=2,
        trail_frames=400,
        axis_half_range=1.5,
        note_en="Classic equal-mass figure-eight orbit in Newtonian gravity.  The 1PN panel shows how relativistic corrections spoil the exact Newtonian periodicity when they are made visible.",
        note_cs="Klasická newtonovská osmičková dráha tří stejných hmotností. Panel 1PN ukazuje, jak relativistické korekce narušují přesnou newtonovskou periodicitu, pokud jsou dostatečně viditelné.",
    ),
    "Lagrange equilateral triple": lagrange_preset(),
    "Binary plus intruder": Preset(
        n=3,
        masses=(1.0, 1.0, 0.2),
        positions=((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 3.0, 0.15)),
        velocities=((0.0, -0.70710678, 0.0), (0.0, 0.70710678, 0.0), (0.75, -0.55, 0.02)),
        g=1.0,
        c=80.0,
        total_time=18.0,
        dt=0.006,
        frame_stride=3,
        trail_frames=350,
        axis_half_range=3.5,
        note_en="A near-circular binary is perturbed by a lighter third body.  This preset is useful for seeing scattering and chaotic sensitivity.",
        note_cs="Téměř kruhová dvojhvězda je porušena lehčím třetím tělesem. Preset je vhodný pro pozorování rozptylu a chaotické citlivosti.",
    ),
    "N-body disk, N = 6": Preset(
        n=6,
        masses=(1.0, 0.08, 0.06, 0.04, 0.03, 0.02),
        positions=(
            (0.0, 0.0, 0.0),
            (0.8, 0.0, 0.0),
            (0.0, 1.1, 0.05),
            (-1.35, 0.0, -0.05),
            (0.0, -1.75, 0.02),
            (2.2, 0.0, 0.08),
        ),
        velocities=(
            (0.0, 0.0, 0.0),
            (0.0, 1.10, 0.02),
            (-0.95, 0.0, -0.01),
            (0.0, -0.82, 0.01),
            (0.68, 0.0, 0.00),
            (0.0, 0.62, -0.02),
        ),
        g=1.0,
        c=100.0,
        total_time=28.0,
        dt=0.01,
        frame_stride=4,
        trail_frames=450,
        axis_half_range=2.8,
        note_en="A central mass with five lighter bodies in a disk-like configuration.  This is a toy N-body model, not a stable planetary system.",
        note_cs="Centrální hmotnost s pěti lehčími tělesy v diskové konfiguraci. Jde o hračkový N-tělesový model, ne o stabilní planetární soustavu.",
    ),
}

TR = {
    "English": {
        "language": "Language / Jazyk",
        "reset_initial": "Reset to initial values",
        "title": "Interactive N-body problem: Newton gravity vs. Einstein GTR 1PN approximation",
        "what": "What this app computes",
        "global_controls": "Global controls",
        "preset": "Initial-condition preset",
        "load_preset": "Load selected preset",
        "preset_note": "Preset note",
        "n_bodies": "Number of bodies N",
        "g_value": "G in model units",
        "softening": "Softening epsilon [L0]",
        "time_controls": "Time integration",
        "total_time": "Simulated time [T0]",
        "dt": "RK4 time step Δt [T0]",
        "frame_stride": "RK4 steps per displayed frame",
        "trail_frames": "Trail length [displayed frames]",
        "relativity": "1PN parameters",
        "log10_c": "log10(c [L0/T0])",
        "pn_log10": "log10(1PN multiplier)",
        "display": "Display",
        "visual_controls_no_recompute": "Visual controls (no recompute)",
        "axis_half_range": "Fixed view-box half-width [L0]",
        "axis_scaling": "View-box scaling mode",
        "axis_fixed": "Fixed by slider",
        "axis_full": "Fit full computed trajectory",
        "axis_dynamic": "Dynamic auto-fit during playback",
        "marker_base": "Base marker diameter [px]",
        "marker_mass_gamma": "Marker mass compression gamma",
        "body_parameters": "Initial masses, positions and velocities",
        "mass": "Mass m [M0]",
        "pos": "Initial position [L0]",
        "vel": "Initial velocity [L0/T0]",
        "x": "x", "y": "y", "z": "z", "vx": "vx", "vy": "vy", "vz": "vz",
        "playback": "Live playback",
        "live_interval_ms": "Live playback refresh [ms]",
        "frames_per_refresh": "Frames advanced per refresh",
        "loop_playback": "Loop live playback",
        "plotly_animation": "Also create Plotly Play button",
        "max_animation_frames": "Max Plotly animation frames",
        "animation_frame_duration": "Animation frame duration [ms]",
        "orbit_curve_points": "Max points per trajectory curve",
        "export": "Export and downloads",
        "export_gif_frames": "Animated GIF frames",
        "generate_gif": "Generate downloadable GIF video",
        "download_gif": "Download GIF video",
        "gif_generating": "Rendering animated GIF. This can take a while on Streamlit Cloud...",
        "gif_ready": "Animated GIF is ready for download.",
        "gif_note": "The GIF is rendered only after pressing the button. It uses a reduced number of frames so that export remains practical on the web server.",
        "download_protocol": "Download simulation protocol TXT",
        "protocol_note": "The protocol stores the current numerical parameters, initial masses, initial positions and velocities, and the final Newton/1PN positions at the simulated final time.",
        "browser_animation_note": "Use the Play / Pause / Reset buttons above the Plotly graph. Playback runs in the browser; Streamlit does not rerun for every animation frame. Trajectory curves are progressive: only the already-travelled path is drawn.",
        "start": "▶ Start",
        "pause": "⏸ Pause",
        "reset_time": "↺ Reset time",
        "status_running": "running",
        "status_paused": "paused",
        "frame_slider": "Displayed time frame",
        "newton_title": "Newton gravity",
        "pn_title": "Einstein GTR 1PN approximation",
        "diagnostics": "Approximation diagnostics",
        "current_params": "Current body parameters",
        "displayed_time": "Displayed time",
        "body": "body",
        "body_i": "Body",
        "model_mass": "model mass [M0]",
        "warning_steps": "The selected time span and time step would require too many RK4 steps. Increase Δt, shorten the simulation, or increase the displayed-frame stride.",
        "pn_warning": "The chosen parameters push the system outside the comfortable weak-field / slow-motion 1PN regime. The visualization may still be interesting, but it should not be interpreted as a quantitatively valid relativistic model.",
        "no_autorefresh": "Live playback requires streamlit-autorefresh. Use the Plotly Play button or install the package.",
        "fixed_axes_note": "In fixed mode the 3D axes are locked by the view-box slider; in dynamic mode the box auto-fits the moving bodies during playback. Manual Plotly zoom/pan/rotate is most stable in fixed mode.",
        "caption": "Marker diameters are visually compressed. They are not drawn on the same linear scale as the coordinates.",
        "sources": "References and sources",
        "units_caption": "The model uses arbitrary dimensionless units: length L0, time T0, mass M0. Velocities are in L0/T0 and G is set by the slider.",
    },
    "Čeština": {
        "language": "Language / Jazyk",
        "reset_initial": "Obnovit výchozí hodnoty",
        "title": "Interaktivní problém N těles: Newtonova gravitace vs. Einsteinova OTR 1PN aproximace",
        "what": "Co aplikace počítá",
        "global_controls": "Globální ovládání",
        "preset": "Preset počátečních podmínek",
        "load_preset": "Načíst zvolený preset",
        "preset_note": "Poznámka k presetu",
        "n_bodies": "Počet těles N",
        "g_value": "G v modelových jednotkách",
        "softening": "Softening epsilon [L0]",
        "time_controls": "Časová integrace",
        "total_time": "Délka simulace [T0]",
        "dt": "Krok RK4 Δt [T0]",
        "frame_stride": "RK4 kroků na zobrazený snímek",
        "trail_frames": "Délka stopy [zobrazené snímky]",
        "relativity": "1PN parametry",
        "log10_c": "log10(c [L0/T0])",
        "pn_log10": "log10(násobku 1PN)",
        "display": "Zobrazení",
        "visual_controls_no_recompute": "Vizuální ovládání (bez přepočtu)",
        "axis_half_range": "Pevná polovina šířky boxu [L0]",
        "axis_scaling": "Režim škálování boxu",
        "axis_fixed": "Fixní podle posuvníku",
        "axis_full": "Přizpůsobit celé spočtené trajektorii",
        "axis_dynamic": "Dynamické přizpůsobování během přehrávání",
        "marker_base": "Základní průměr značky [px]",
        "marker_mass_gamma": "Komprese velikosti podle hmotnosti gamma",
        "body_parameters": "Počáteční hmotnosti, polohy a rychlosti",
        "mass": "Hmotnost m [M0]",
        "pos": "Počáteční poloha [L0]",
        "vel": "Počáteční rychlost [L0/T0]",
        "x": "x", "y": "y", "z": "z", "vx": "vx", "vy": "vy", "vz": "vz",
        "playback": "Živé přehrávání",
        "live_interval_ms": "Obnova přehrávání [ms]",
        "frames_per_refresh": "Počet snímků na obnovu",
        "loop_playback": "Přehrávat ve smyčce",
        "plotly_animation": "Vytvořit také Plotly Play tlačítko",
        "max_animation_frames": "Maximální počet Plotly animačních snímků",
        "animation_frame_duration": "Délka animačního snímku [ms]",
        "orbit_curve_points": "Maximální počet bodů na křivku trajektorie",
        "export": "Export a stažení",
        "export_gif_frames": "Počet snímků animovaného GIFu",
        "generate_gif": "Vygenerovat stažitelné GIF video",
        "download_gif": "Stáhnout GIF video",
        "gif_generating": "Renderuji animovaný GIF. Na Streamlit Cloud to může chvíli trvat...",
        "gif_ready": "Animovaný GIF je připraven ke stažení.",
        "gif_note": "GIF se renderuje až po stisku tlačítka. Používá omezený počet snímků, aby byl export na webovém serveru prakticky použitelný.",
        "download_protocol": "Stáhnout protokol simulace TXT",
        "protocol_note": "Protokol ukládá aktuální numerické parametry, počáteční hmotnosti, počáteční polohy a rychlosti a finální Newton/1PN polohy v konečném čase simulace.",
        "browser_animation_note": "Použij tlačítka Play / Pauza / Reset nad Plotly grafem. Přehrávání běží v prohlížeči; Streamlit se nespouští znovu pro každý animační snímek. Křivky trajektorií jsou progresivní: kreslí se pouze již proletěná dráha.",
        "start": "▶ Start",
        "pause": "⏸ Pauza",
        "reset_time": "↺ Reset času",
        "status_running": "běží",
        "status_paused": "pozastaveno",
        "frame_slider": "Zobrazený časový snímek",
        "newton_title": "Newtonova gravitace",
        "pn_title": "Einsteinova OTR 1PN aproximace",
        "diagnostics": "Diagnostika platnosti aproximace",
        "current_params": "Aktuální parametry těles",
        "displayed_time": "Zobrazený čas",
        "body": "těleso",
        "body_i": "Těleso",
        "model_mass": "modelová hmotnost [M0]",
        "warning_steps": "Zvolená délka simulace a krok by vyžadovaly příliš mnoho RK4 kroků. Zvětši Δt, zkrať simulaci nebo zvětši stride zobrazených snímků.",
        "pn_warning": "Zvolené parametry posouvají systém mimo pohodlný slabopolní / pomalý 1PN režim. Vizualizace může být zajímavá, ale nemá být interpretována jako kvantitativně platný relativistický model.",
        "no_autorefresh": "Živé přehrávání vyžaduje streamlit-autorefresh. Použij Plotly Play tlačítko nebo balíček nainstaluj.",
        "fixed_axes_note": "Ve fixním režimu jsou 3D osy zamčené posuvníkem velikosti boxu; v dynamickém režimu se box během přehrávání přizpůsobuje pohybujícím se tělesům. Ruční zoom/posun/rotace v Plotly je nejstabilnější ve fixním režimu.",
        "caption": "Průměry značek jsou vizuálně komprimované. Nejsou kreslené ve stejném lineárním měřítku jako souřadnice.",
        "sources": "Reference a zdroje",
        "units_caption": "Model používá libovolné bezrozměrné jednotky: délku L0, čas T0 a hmotnost M0. Rychlosti jsou v L0/T0 a G nastavuje posuvník.",
    },
}


def t(key: str) -> str:
    language = st.session_state.get("language", "English")
    return TR.get(language, TR["English"]).get(key, TR["English"].get(key, key))


def initialize_session_defaults() -> None:
    """Initialize session state before any widgets are created."""
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    # Load the figure-eight preset into per-body values on first run.
    if "_preset_initialized" not in st.session_state:
        load_preset_to_state("Figure-eight 3-body orbit", reset_playback=True)
        st.session_state["_preset_initialized"] = True


def load_preset_to_state(preset_name: str, reset_playback: bool = True) -> None:
    p = PRESETS[preset_name]
    # Do not assign st.session_state["preset"] here.  When the preset selectbox
    # already exists in the current Streamlit run, changing its key would raise
    # StreamlitAPIException.  The selected value is already held by the widget.
    st.session_state["n_bodies"] = p.n
    st.session_state["g_value"] = p.g
    st.session_state["log10_c"] = math.log10(p.c)
    st.session_state["total_time"] = p.total_time
    st.session_state["dt"] = p.dt
    st.session_state["frame_stride"] = p.frame_stride
    st.session_state["trail_frames"] = p.trail_frames
    st.session_state["axis_half_range"] = p.axis_half_range
    st.session_state["pn_log10"] = 0.0
    for i in range(MAX_BODIES):
        if i < p.n:
            st.session_state[f"m_{i}"] = float(p.masses[i])
            st.session_state[f"x_{i}"] = float(p.positions[i][0])
            st.session_state[f"y_{i}"] = float(p.positions[i][1])
            st.session_state[f"z_{i}"] = float(p.positions[i][2])
            st.session_state[f"vx_{i}"] = float(p.velocities[i][0])
            st.session_state[f"vy_{i}"] = float(p.velocities[i][1])
            st.session_state[f"vz_{i}"] = float(p.velocities[i][2])
        else:
            # Deterministic unused-body defaults so that increasing N gives a sensible start.
            angle = 2.0 * math.pi * (i - p.n + 1) / max(MAX_BODIES - p.n + 1, 1)
            radius = 1.4 + 0.25 * i
            st.session_state[f"m_{i}"] = 0.05
            st.session_state[f"x_{i}"] = radius * math.cos(angle)
            st.session_state[f"y_{i}"] = radius * math.sin(angle)
            st.session_state[f"z_{i}"] = 0.02 * ((-1) ** i)
            st.session_state[f"vx_{i}"] = -0.35 * math.sin(angle)
            st.session_state[f"vy_{i}"] = 0.35 * math.cos(angle)
            st.session_state[f"vz_{i}"] = 0.0
    if reset_playback:
        st.session_state["live_frame"] = 0
        st.session_state["running"] = False


def reset_to_initial_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("_"):
            continue
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    load_preset_to_state("Figure-eight 3-body orbit", reset_playback=True)
    st.session_state["language"] = "English"
    st.session_state["running"] = False
    st.session_state["live_frame"] = 0


# =============================================================================
# Physics and numerical integration
# =============================================================================

def barycentric_transform(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = float(np.sum(masses))
    if total <= 0.0:
        return pos, vel
    r_cm = np.sum(pos * masses[:, None], axis=0) / total
    v_cm = np.sum(vel * masses[:, None], axis=0) / total
    return pos - r_cm[None, :], vel - v_cm[None, :]


def collect_initial_conditions(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masses = np.zeros(n, dtype=float)
    pos = np.zeros((n, 3), dtype=float)
    vel = np.zeros((n, 3), dtype=float)
    for i in range(n):
        masses[i] = max(float(st.session_state[f"m_{i}"]), 0.0)
        pos[i] = (float(st.session_state[f"x_{i}"]), float(st.session_state[f"y_{i}"]), float(st.session_state[f"z_{i}"]))
        vel[i] = (float(st.session_state[f"vx_{i}"]), float(st.session_state[f"vy_{i}"]), float(st.session_state[f"vz_{i}"]))
    pos, vel = barycentric_transform(pos, vel, masses)
    return pos, vel, masses


def acceleration_newton(pos: np.ndarray, masses: np.ndarray, g_value: float, softening: float) -> np.ndarray:
    """Vectorized Newtonian N-body acceleration with Plummer-like softening.

    This replaces the older double Python loop by NumPy broadcasting.  For the
    small interactive systems used here the physics is unchanged, but the RK4
    integration is noticeably faster because the acceleration is evaluated four
    times per RK4 step and once for each Newton/1PN model evaluation.
    """
    pos = np.asarray(pos, dtype=float)
    masses = np.asarray(masses, dtype=float)
    n = len(masses)
    if n == 0:
        return np.zeros_like(pos)

    # dr[i,j] = r_i - r_j.  Body j contributes
    #   -G m_j (r_i-r_j) / (|r_i-r_j|^2+eps^2)^(3/2)
    # to body i.  The diagonal i=j is explicitly masked out, which is important
    # also when softening is set to zero.
    dr = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(dr * dr, axis=2) + float(softening) ** 2
    mask = ~np.eye(n, dtype=bool)
    mass_mask = masses[None, :] > 0.0
    valid = mask & mass_mask

    inv_r3 = np.zeros_like(r2, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_r3[valid] = 1.0 / (r2[valid] * np.sqrt(r2[valid]))

    acc = -float(g_value) * np.sum(masses[None, :, None] * dr * inv_r3[:, :, None], axis=1)
    return acc


def acceleration_pairwise_1pn(
    pos: np.ndarray,
    vel: np.ndarray,
    masses: np.ndarray,
    g_value: float,
    softening: float,
    c_value: float,
    pn_multiplier: float,
) -> np.ndarray:
    """Newtonian acceleration plus pairwise two-body 1PN corrections.

    The two-body relative 1PN correction is written in harmonic-coordinate form
    and applied pair by pair.  Genuine EIH three-body terms are intentionally not
    included, so the result is an educational pairwise approximation, not a full
    relativistic N-body ephemeris.
    """
    acc = acceleration_newton(pos, masses, g_value, softening)
    if pn_multiplier == 0.0:
        return acc
    c2 = float(c_value) ** 2
    if c2 <= 0.0:
        return acc
    eps2 = float(softening) ** 2
    n = len(masses)
    for i in range(n):
        for j in range(i + 1, n):
            mi = masses[i]
            mj = masses[j]
            mtot = mi + mj
            if mtot <= 0.0:
                continue
            dr = pos[i] - pos[j]
            r2 = float(np.dot(dr, dr)) + eps2
            r = math.sqrt(r2)
            nvec = dr / r
            vrel = vel[i] - vel[j]
            v2 = float(np.dot(vrel, vrel))
            rdot = float(np.dot(nvec, vrel))
            eta = (mi * mj) / (mtot * mtot)

            bracket = (
                nvec * ((4.0 + 2.0 * eta) * g_value * mtot / r - (1.0 + 3.0 * eta) * v2 + 1.5 * eta * rdot * rdot)
                + (4.0 - 2.0 * eta) * rdot * vrel
            )
            a_rel_corr = (g_value * mtot / (c2 * r2)) * bracket * pn_multiplier

            # Split the relative correction a_i - a_j while preserving pair momentum.
            acc[i] += (mj / mtot) * a_rel_corr
            acc[j] += -(mi / mtot) * a_rel_corr
    return acc


def rhs(
    state: np.ndarray,
    masses: np.ndarray,
    model: str,
    g_value: float,
    softening: float,
    c_value: float,
    pn_multiplier: float,
) -> np.ndarray:
    n = len(masses)
    pos = state[: 3 * n].reshape((n, 3))
    vel = state[3 * n :].reshape((n, 3))
    if model == "newton":
        acc = acceleration_newton(pos, masses, g_value, softening)
    elif model == "1pn":
        acc = acceleration_pairwise_1pn(pos, vel, masses, g_value, softening, c_value, pn_multiplier)
    else:
        raise ValueError(model)
    return np.concatenate((vel.reshape(-1), acc.reshape(-1)))


def rk4_step(
    state: np.ndarray,
    dt: float,
    masses: np.ndarray,
    model: str,
    g_value: float,
    softening: float,
    c_value: float,
    pn_multiplier: float,
) -> np.ndarray:
    k1 = rhs(state, masses, model, g_value, softening, c_value, pn_multiplier)
    k2 = rhs(state + 0.5 * dt * k1, masses, model, g_value, softening, c_value, pn_multiplier)
    k3 = rhs(state + 0.5 * dt * k2, masses, model, g_value, softening, c_value, pn_multiplier)
    k4 = rhs(state + dt * k3, masses, model, g_value, softening, c_value, pn_multiplier)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def newtonian_energy(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray, g_value: float, softening: float) -> float:
    kinetic = 0.5 * float(np.sum(masses[:, None] * vel * vel))
    potential = 0.0
    eps2 = float(softening) ** 2
    n = len(masses)
    for i in range(n):
        for j in range(i + 1, n):
            r = math.sqrt(float(np.dot(pos[i] - pos[j], pos[i] - pos[j])) + eps2)
            potential += -g_value * masses[i] * masses[j] / r
    return kinetic + potential


def diagnostics(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray, g_value: float, softening: float, c_value: float) -> dict[str, float]:
    speeds = np.linalg.norm(vel, axis=1)
    max_v_over_c = float(np.max(speeds) / max(c_value, 1.0e-30)) if len(speeds) else 0.0
    eps2 = float(softening) ** 2
    min_sep = float("inf")
    max_compactness = 0.0
    n = len(masses)
    for i in range(n):
        for j in range(i + 1, n):
            r = math.sqrt(float(np.dot(pos[i] - pos[j], pos[i] - pos[j])) + eps2)
            min_sep = min(min_sep, r)
            if c_value > 0.0:
                max_compactness = max(max_compactness, g_value * masses[i] / (r * c_value * c_value), g_value * masses[j] / (r * c_value * c_value))
    if min_sep == float("inf"):
        min_sep = 0.0
    return {"max_v_over_c": max_v_over_c, "max_GM_over_rc2": max_compactness, "min_separation": min_sep}


@st.cache_data(show_spinner=False)
def simulate_cached(
    n_bodies: int,
    masses_tuple: tuple[float, ...],
    pos_tuple: tuple[tuple[float, float, float], ...],
    vel_tuple: tuple[tuple[float, float, float], ...],
    g_value: float,
    softening: float,
    total_time: float,
    dt: float,
    frame_stride: int,
    c_value: float,
    pn_log10: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float], dict[str, float], float, float]:
    masses = np.asarray(masses_tuple, dtype=float)
    pos0 = np.asarray(pos_tuple, dtype=float)
    vel0 = np.asarray(vel_tuple, dtype=float)
    pos0, vel0 = barycentric_transform(pos0, vel0, masses)

    state_n = np.concatenate((pos0.reshape(-1), vel0.reshape(-1)))
    state_p = state_n.copy()
    n = n_bodies
    n_steps = int(math.ceil(total_time / dt))
    frame_stride = max(int(frame_stride), 1)
    pn_multiplier = 10.0 ** float(pn_log10)

    times: list[float] = []
    frames_n: list[np.ndarray] = []
    frames_p: list[np.ndarray] = []

    def store(step: int) -> None:
        times.append(step * dt)
        frames_n.append(state_n[: 3 * n].reshape((n, 3)).copy())
        frames_p.append(state_p[: 3 * n].reshape((n, 3)).copy())

    e0 = newtonian_energy(pos0, vel0, masses, g_value, softening)
    store(0)
    for step in range(1, n_steps + 1):
        state_n = rk4_step(state_n, dt, masses, "newton", g_value, softening, c_value, pn_multiplier)
        state_p = rk4_step(state_p, dt, masses, "1pn", g_value, softening, c_value, pn_multiplier)
        if step % frame_stride == 0 or step == n_steps:
            store(step)

    pos_n_end = state_n[: 3 * n].reshape((n, 3))
    vel_n_end = state_n[3 * n :].reshape((n, 3))
    pos_p_end = state_p[: 3 * n].reshape((n, 3))
    vel_p_end = state_p[3 * n :].reshape((n, 3))
    diag_n = diagnostics(pos_n_end, vel_n_end, masses, g_value, softening, c_value)
    diag_p = diagnostics(pos_p_end, vel_p_end, masses, g_value, softening, c_value)
    e_n_end = newtonian_energy(pos_n_end, vel_n_end, masses, g_value, softening)
    rel_energy_drift = abs((e_n_end - e0) / e0) if abs(e0) > 1.0e-30 else float("nan")

    return np.asarray(times), np.asarray(frames_n), np.asarray(frames_p), masses, diag_n, diag_p, e0, rel_energy_drift


# =============================================================================
# Plotting
# =============================================================================

def body_labels(n: int) -> list[str]:
    if st.session_state.get("language", "English") == "Čeština":
        return [f"Těleso {i + 1}" for i in range(n)]
    return [f"Body {i + 1}" for i in range(n)]


def marker_sizes(masses: np.ndarray, base_size: float, gamma: float) -> list[float]:
    max_m = max(float(np.max(masses)), 1.0e-12)
    sizes = []
    for m in masses:
        norm = max(float(m) / max_m, 1.0e-12)
        sizes.append(float(base_size * (0.75 + 1.8 * norm ** gamma)))
    return sizes


def trail_slice(frame: int, trail_frames: int) -> slice:
    start = max(0, frame - max(int(trail_frames), 1) + 1)
    return slice(start, frame + 1)


def progressive_path_indices(frame: int, trail_frames: int, max_points: int) -> np.ndarray:
    """Indices for the already-travelled part of a trajectory.

    If trail_frames <= 0, the full history from the beginning up to ``frame`` is
    shown.  Otherwise only the last ``trail_frames`` displayed frames are shown.
    The returned indices are downsampled to at most ``max_points`` so that the
    browser and GIF export remain usable.
    """
    fidx = max(int(frame), 0)
    if int(trail_frames) > 0:
        start = max(0, fidx - int(trail_frames) + 1)
    else:
        start = 0
    max_points = max(int(max_points), 2)
    count = fidx - start + 1
    if count <= 1:
        return np.array([fidx], dtype=int)
    if count <= max_points:
        return np.arange(start, fidx + 1, dtype=int)
    return np.unique(np.linspace(start, fidx, max_points).astype(int))


def axis_half_range_for_mode(
    frames_n: np.ndarray,
    frames_p: np.ndarray,
    mode: str,
    slider_half_range: float,
    frame_index: int,
) -> float:
    """Return a symmetric half-range for the two 3D panels.

    Modes:
    - fixed: use only the user slider; this keeps the visual box constant.
    - full: choose one constant box large enough for the whole computed trajectory.
    - dynamic: recompute the box from the current Newton/1PN body positions.

    The same half-range is used for the left and right panels so that the two
    models remain visually comparable.
    """
    slider_half_range = max(float(slider_half_range), 0.1)
    if mode == "full":
        max_abs = float(max(np.max(np.abs(frames_n)), np.max(np.abs(frames_p))))
        return max(slider_half_range, 1.12 * max_abs, 0.1)
    if mode == "dynamic":
        fidx = int(np.clip(frame_index, 0, len(frames_n) - 1))
        max_abs = float(max(np.max(np.abs(frames_n[fidx])), np.max(np.abs(frames_p[fidx]))))
        return max(slider_half_range, 1.20 * max_abs, 0.1)
    return slider_half_range


def axis_template_from_half_range(half_range: float, dynamic: bool = False) -> dict:
    lim = max(float(half_range), 0.1)
    template = dict(
        xaxis=dict(title="x [L0]", range=[-lim, lim], autorange=False),
        yaxis=dict(title="y [L0]", range=[-lim, lim], autorange=False),
        zaxis=dict(title="z [L0]", range=[-lim, lim], autorange=False),
        aspectmode="cube",
    )
    if not dynamic:
        template["uirevision"] = "fixed-camera"
    return template


def make_figure(
    times: np.ndarray,
    frames_n: np.ndarray,
    frames_p: np.ndarray,
    masses: np.ndarray,
    frame_index: int,
    trail_frames: int,
    axis_half_range: float,
    axis_scaling_mode: str,
    base_marker: float,
    mass_gamma: float,
    animate: bool,
    max_animation_frames: int,
    orbit_curve_points: int,
    animation_frame_duration: int,
) -> go.Figure:
    """Build a Plotly 3D figure with progressive trajectory trails.

    The numerical trajectory is still precomputed and cached after pressing
    Apply and recompute. The display, however, shows only the part of each
    trajectory that has already been travelled up to the current animation
    frame. Future trajectory segments are not drawn before the bodies move.
    """
    n = len(masses)
    labels = body_labels(n)
    colors = BODY_COLORS[:n]
    sizes = marker_sizes(masses, base_marker, mass_gamma)
    frame_index = int(np.clip(frame_index, 0, len(times) - 1))

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(t("newton_title"), t("pn_title")),
        horizontal_spacing=0.02,
    )

    n_total = len(times)
    max_curve_points = max(int(orbit_curve_points), 10)

    def add_model_progressive_paths_and_markers(frames: np.ndarray, col: int, prefix: str) -> tuple[list[int], int]:
        line_trace_indices: list[int] = []
        path_idx = progressive_path_indices(frame_index, trail_frames, max_curve_points)
        for i in range(n):
            xyz = frames[path_idx, i, :]
            line_trace_indices.append(len(fig.data))
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                    mode="lines",
                    line=dict(width=2, color=colors[i]),
                    name=f"{prefix} {labels[i]} trajectory",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=col,
            )
        pts = frames[frame_index, :, :]
        marker_trace_index = len(fig.data)
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers+text",
                marker=dict(size=sizes, color=colors, opacity=0.96, sizemode="diameter"),
                text=labels,
                textposition="top center",
                name=f"{prefix} bodies",
                showlegend=False,
                hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
            ),
            row=1, col=col,
        )
        return line_trace_indices, marker_trace_index

    newton_line_traces, newton_marker_trace = add_model_progressive_paths_and_markers(frames_n, 1, "Newton")
    pn_line_traces, pn_marker_trace = add_model_progressive_paths_and_markers(frames_p, 2, "1PN")
    all_trace_indices = newton_line_traces + [newton_marker_trace] + pn_line_traces + [pn_marker_trace]

    axis_mode = str(axis_scaling_mode)
    initial_half_range = axis_half_range_for_mode(frames_n, frames_p, axis_mode, axis_half_range, frame_index)
    dynamic_axes = axis_mode == "dynamic"
    axis_template = axis_template_from_half_range(initial_half_range, dynamic=dynamic_axes)
    layout_kwargs = dict(
        scene=axis_template,
        scene2=axis_template,
        height=760,
        margin=dict(l=5, r=5, t=70, b=5),
        title=f"N-body model: t = {times[frame_index]:.3f} T0",
    )
    if not dynamic_axes:
        layout_kwargs["uirevision"] = "fixed-camera"
    fig.update_layout(**layout_kwargs)

    if animate:
        if n_total <= max_animation_frames:
            selected = list(range(n_total))
        else:
            selected = sorted(set(np.linspace(0, n_total - 1, int(max_animation_frames)).astype(int).tolist()))

        frames_out = []
        for fidx in selected:
            path_idx = progressive_path_indices(fidx, trail_frames, max_curve_points)
            pts_n = frames_n[fidx, :, :]
            pts_p = frames_p[fidx, :, :]
            frame_data = []
            for i in range(n):
                xyz = frames_n[path_idx, i, :]
                frame_data.append(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2]))
            frame_data.append(go.Scatter3d(x=pts_n[:, 0], y=pts_n[:, 1], z=pts_n[:, 2], text=labels))
            for i in range(n):
                xyz = frames_p[path_idx, i, :]
                frame_data.append(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2]))
            frame_data.append(go.Scatter3d(x=pts_p[:, 0], y=pts_p[:, 1], z=pts_p[:, 2], text=labels))

            frame_kwargs = dict(data=frame_data, traces=all_trace_indices, name=str(fidx))
            if dynamic_axes:
                dyn_half_range = axis_half_range_for_mode(frames_n, frames_p, "dynamic", axis_half_range, fidx)
                dyn_template = axis_template_from_half_range(dyn_half_range, dynamic=True)
                frame_kwargs["layout"] = go.Layout(
                    scene=dyn_template,
                    scene2=dyn_template,
                    title=f"N-body model: t = {times[fidx]:.3f} T0",
                )
            frames_out.append(go.Frame(**frame_kwargs))
        fig.frames = frames_out
        duration = max(int(animation_frame_duration), 1)
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.02,
                    y=1.12,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[None, {"frame": {"duration": duration, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                        ),
                        dict(
                            label="↺ Reset",
                            method="animate",
                            args=[[str(selected[0])], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    x=0.1,
                    y=0.01,
                    len=0.8,
                    currentvalue={"prefix": "t = ", "suffix": " T0"},
                    steps=[
                        dict(
                            method="animate",
                            label=f"{times[fidx]:.2f}",
                            args=[[str(fidx)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        )
                        for fidx in selected
                    ],
                )
            ],
        )

    return fig


# =============================================================================
# Export helpers
# =============================================================================

def selected_frame_indices(n_total: int, max_frames: int) -> np.ndarray:
    """Return evenly spaced frame indices for browser/GIF export."""
    n_total = max(int(n_total), 1)
    max_frames = max(int(max_frames), 1)
    if n_total <= max_frames:
        return np.arange(n_total, dtype=int)
    return np.unique(np.linspace(0, n_total - 1, max_frames).astype(int))


def build_protocol_text(
    language: str,
    preset_name: str,
    n_bodies: int,
    masses: np.ndarray,
    initial_pos: np.ndarray,
    initial_vel: np.ndarray,
    times: np.ndarray,
    frames_n: np.ndarray,
    frames_p: np.ndarray,
    g_value: float,
    softening: float,
    dt: float,
    frame_stride: int,
    c_value: float,
    pn_log10: float,
    axis_scaling_mode: str,
    axis_half_range: float,
    diag_n: dict[str, float],
    diag_p: dict[str, float],
    energy_drift: float,
) -> str:
    """Create a plain-text report/protocol for the current simulation."""
    labels = body_labels(n_bodies)
    final_time = float(times[-1]) if len(times) else 0.0
    pn_multiplier = 10.0 ** float(pn_log10)

    if language == "Čeština":
        lines = [
            "Protokol simulace N teles",
            "============================",
            "",
            f"Preset: {preset_name}",
            f"Pocet teles N: {n_bodies}",
            f"Konecny cas simulace: {final_time:.10g} T0",
            "",
            "Numericke a fyzikalni parametry:",
            f"  G = {g_value:.10g}",
            f"  softening epsilon = {softening:.10g} L0",
            f"  RK4 krok dt = {dt:.10g} T0",
            f"  RK4 kroku na ulozeny frame = {frame_stride}",
            f"  c = {c_value:.10g} L0/T0",
            f"  1PN multiplier = {pn_multiplier:.10g}",
            f"  rezim boxu = {axis_scaling_mode}",
            f"  fixni polovina boxu = {axis_half_range:.10g} L0",
            "",
            "Poznamka: pocatecni polohy a rychlosti nize jsou hodnoty pouzite integratorem po posunu do barycentricke soustavy.",
            "",
            "Pocatecni stav:",
        ]
        for i in range(n_bodies):
            lines.append(
                f"  {labels[i]}: m={masses[i]:.12g} M0; "
                f"r0=({initial_pos[i,0]:.12g}, {initial_pos[i,1]:.12g}, {initial_pos[i,2]:.12g}) L0; "
                f"v0=({initial_vel[i,0]:.12g}, {initial_vel[i,1]:.12g}, {initial_vel[i,2]:.12g}) L0/T0"
            )
        lines += ["", f"Finalni polohy v case t={final_time:.10g} T0:", "  Newton gravity:"]
        for i in range(n_bodies):
            r = frames_n[-1, i, :]
            lines.append(f"    {labels[i]}: r=({r[0]:.12g}, {r[1]:.12g}, {r[2]:.12g}) L0")
        lines.append("  Einstein GTR 1PN approximation:")
        for i in range(n_bodies):
            r = frames_p[-1, i, :]
            lines.append(f"    {labels[i]}: r=({r[0]:.12g}, {r[1]:.12g}, {r[2]:.12g}) L0")
        lines += [
            "",
            "Diagnostika:",
            f"  Newton max v/c = {diag_n['max_v_over_c']:.12e}",
            f"  Newton max Gm/(rc^2) = {diag_n['max_GM_over_rc2']:.12e}",
            f"  1PN max v/c = {diag_p['max_v_over_c']:.12e}",
            f"  1PN max Gm/(rc^2) = {diag_p['max_GM_over_rc2']:.12e}",
            f"  minimalni separace = {min(diag_n['min_separation'], diag_p['min_separation']):.12e} L0",
            f"  relativni drift Newtonovske energie = {energy_drift:.12e}",
            "",
            "Model: Newtonovska N-telesova gravitace vlevo a parova dvoutelesova 1PN korekce vpravo. Nejde o plnou EIH vice-telesovou efemeridu.",
        ]
    else:
        lines = [
            "N-body simulation protocol",
            "==========================",
            "",
            f"Preset: {preset_name}",
            f"Number of bodies N: {n_bodies}",
            f"Final simulated time: {final_time:.10g} T0",
            "",
            "Numerical and physical parameters:",
            f"  G = {g_value:.10g}",
            f"  softening epsilon = {softening:.10g} L0",
            f"  RK4 time step dt = {dt:.10g} T0",
            f"  RK4 steps per stored frame = {frame_stride}",
            f"  c = {c_value:.10g} L0/T0",
            f"  1PN multiplier = {pn_multiplier:.10g}",
            f"  view-box mode = {axis_scaling_mode}",
            f"  fixed view-box half-width = {axis_half_range:.10g} L0",
            "",
            "Note: the initial positions and velocities below are the values used by the integrator after transformation to the barycentric frame.",
            "",
            "Initial state:",
        ]
        for i in range(n_bodies):
            lines.append(
                f"  {labels[i]}: m={masses[i]:.12g} M0; "
                f"r0=({initial_pos[i,0]:.12g}, {initial_pos[i,1]:.12g}, {initial_pos[i,2]:.12g}) L0; "
                f"v0=({initial_vel[i,0]:.12g}, {initial_vel[i,1]:.12g}, {initial_vel[i,2]:.12g}) L0/T0"
            )
        lines += ["", f"Final positions at t={final_time:.10g} T0:", "  Newton gravity:"]
        for i in range(n_bodies):
            r = frames_n[-1, i, :]
            lines.append(f"    {labels[i]}: r=({r[0]:.12g}, {r[1]:.12g}, {r[2]:.12g}) L0")
        lines.append("  Einstein GTR 1PN approximation:")
        for i in range(n_bodies):
            r = frames_p[-1, i, :]
            lines.append(f"    {labels[i]}: r=({r[0]:.12g}, {r[1]:.12g}, {r[2]:.12g}) L0")
        lines += [
            "",
            "Diagnostics:",
            f"  Newton max v/c = {diag_n['max_v_over_c']:.12e}",
            f"  Newton max Gm/(rc^2) = {diag_n['max_GM_over_rc2']:.12e}",
            f"  1PN max v/c = {diag_p['max_v_over_c']:.12e}",
            f"  1PN max Gm/(rc^2) = {diag_p['max_GM_over_rc2']:.12e}",
            f"  minimum separation = {min(diag_n['min_separation'], diag_p['min_separation']):.12e} L0",
            f"  relative Newtonian energy drift = {energy_drift:.12e}",
            "",
            "Model: Newtonian N-body gravity on the left and a pairwise two-body 1PN correction on the right. This is not a full EIH many-body ephemeris.",
        ]
    return "\n".join(lines) + "\n"


def render_animation_gif(
    times: np.ndarray,
    frames_n: np.ndarray,
    frames_p: np.ndarray,
    masses: np.ndarray,
    axis_half_range: float,
    axis_scaling_mode: str,
    base_marker: float,
    mass_gamma: float,
    max_gif_frames: int,
    frame_duration_ms: int,
    orbit_curve_points: int,
) -> bytes:
    """Render the current comparison as a downloadable animated GIF.

    The GIF follows the same progressive-trail logic as the browser animation:
    at each frame it draws only the already-travelled part of the trajectory.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    n = len(masses)
    labels = body_labels(n)
    colors = BODY_COLORS[:n]
    sizes = marker_sizes(masses, base_marker, mass_gamma)
    selected = selected_frame_indices(len(times), max_gif_frames)
    max_curve_points = max(int(orbit_curve_points), 10)

    dynamic_axes = str(axis_scaling_mode) == "dynamic"
    initial_half_range = axis_half_range_for_mode(frames_n, frames_p, str(axis_scaling_mode), axis_half_range, int(selected[0]))

    fig = plt.figure(figsize=(12, 5.7))
    ax_n = fig.add_subplot(1, 2, 1, projection="3d")
    ax_p = fig.add_subplot(1, 2, 2, projection="3d")
    axes = (ax_n, ax_p)

    def set_axes(ax, half_range):
        lim = max(float(half_range), 0.1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel("x [L0]")
        ax.set_ylabel("y [L0]")
        ax.set_zlabel("z [L0]")
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    ax_n.set_title(t("newton_title"))
    ax_p.set_title(t("pn_title"))
    for ax in axes:
        set_axes(ax, initial_half_range)

    first_idx = int(selected[0])
    first_path = progressive_path_indices(first_idx, 0, max_curve_points)
    line_artists_n = []
    line_artists_p = []
    for i in range(n):
        xyz_n = frames_n[first_path, i, :]
        xyz_p = frames_p[first_path, i, :]
        (line_n,) = ax_n.plot(xyz_n[:, 0], xyz_n[:, 1], xyz_n[:, 2], color=colors[i], linewidth=1.2, alpha=0.85)
        (line_p,) = ax_p.plot(xyz_p[:, 0], xyz_p[:, 1], xyz_p[:, 2], color=colors[i], linewidth=1.2, alpha=0.85)
        line_artists_n.append(line_n)
        line_artists_p.append(line_p)

    pts_n0 = frames_n[first_idx, :, :]
    pts_p0 = frames_p[first_idx, :, :]
    scat_n = ax_n.scatter(pts_n0[:, 0], pts_n0[:, 1], pts_n0[:, 2], s=[s * s for s in sizes], c=colors, depthshade=True)
    scat_p = ax_p.scatter(pts_p0[:, 0], pts_p0[:, 1], pts_p0[:, 2], s=[s * s for s in sizes], c=colors, depthshade=True)
    text_artists = []

    def update_texts(ax, pts):
        artists = []
        for i in range(n):
            artists.append(ax.text(pts[i, 0], pts[i, 1], pts[i, 2], labels[i], fontsize=7))
        return artists

    text_artists.extend(update_texts(ax_n, pts_n0))
    text_artists.extend(update_texts(ax_p, pts_p0))

    def set_line_3d(line, xyz):
        line.set_data(xyz[:, 0], xyz[:, 1])
        line.set_3d_properties(xyz[:, 2])

    def update(k):
        nonlocal text_artists
        fidx = int(selected[k])
        path_idx = progressive_path_indices(fidx, 0, max_curve_points)
        pts_n = frames_n[fidx, :, :]
        pts_p = frames_p[fidx, :, :]
        for i in range(n):
            set_line_3d(line_artists_n[i], frames_n[path_idx, i, :])
            set_line_3d(line_artists_p[i], frames_p[path_idx, i, :])
        scat_n._offsets3d = (pts_n[:, 0], pts_n[:, 1], pts_n[:, 2])
        scat_p._offsets3d = (pts_p[:, 0], pts_p[:, 1], pts_p[:, 2])
        for artist in text_artists:
            artist.remove()
        text_artists = []
        text_artists.extend(update_texts(ax_n, pts_n))
        text_artists.extend(update_texts(ax_p, pts_p))
        if dynamic_axes:
            half_range = axis_half_range_for_mode(frames_n, frames_p, "dynamic", axis_half_range, fidx)
            for ax in axes:
                set_axes(ax, half_range)
        fig.suptitle(f"N-body model: t = {times[fidx]:.3f} T0")
        return [*line_artists_n, *line_artists_p, scat_n, scat_p, *text_artists]

    fps = max(2, min(30, int(round(1000.0 / max(int(frame_duration_ms), 1)))))
    ani = FuncAnimation(fig, update, frames=len(selected), interval=max(int(frame_duration_ms), 1), blit=False)
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp_name = tmp.name
    tmp.close()
    try:
        ani.save(tmp_name, writer=PillowWriter(fps=fps), dpi=110)
        with open(tmp_name, "rb") as fh:
            data = fh.read()
    finally:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        plt.close(fig)
    return data


# =============================================================================
# UI
# =============================================================================

st.set_page_config(page_title="N-body Newton vs 1PN", layout="wide")
initialize_session_defaults()

# Deferred state-changing actions.  They are handled before any widgets are
# created, which avoids StreamlitAPIException for already-instantiated widgets.
if st.session_state.pop("_reset_requested", False):
    reset_to_initial_state()
    st.rerun()

_pending_preset = st.session_state.pop("_load_preset_requested", None)
if _pending_preset is not None:
    load_preset_to_state(str(_pending_preset), reset_playback=True)
    st.rerun()

# Language is the only control outside the form so that the interface language
# can be changed immediately.  It does not trigger recomputation because the
# trajectory calculation is cached and depends only on the physical parameters.
language = st.sidebar.selectbox(t("language"), ("English", "Čeština"), key="language")
st.title(t("title"))
st.caption("Build: N-body v2 fast tuning (vectorized Newton + visual-only controls)")

if st.sidebar.button(t("reset_initial"), use_container_width=True):
    st.session_state["_reset_requested"] = True
    st.rerun()

with st.expander(t("what"), expanded=False):
    if language == "Čeština":
        st.markdown(
            r"""
Tato aplikace řeší hračkový gravitační problém více těles ve 3D.  Levý panel
počítá Newtonovu gravitaci.  Pravý panel počítá Newtonovu gravitaci plus
párovou dvoutělesovou 1PN korekci inspirovanou obecnou relativitou.  Model
používá bezrozměrné jednotky: délku $L_0$, čas $T_0$ a hmotnost $M_0$.

Pro každé těleso lze měnit hmotnost, počáteční polohu a počáteční rychlost.
Parametr $G$ nastavuje sílu Newtonovy gravitace v modelových jednotkách.
Parametr $c$ má jednotku $L_0/T_0$ a určuje velikost relativistické korekce.
Čím větší $c$, tím blíže je pravý panel Newtonovu limitu.  Posuvník
"násobek 1PN" je pouze edukační lupa na relativistické členy.
            """
        )
        st.markdown(
            r"""
**Výchozí scénáře.**  Rozbalovací menu *Preset počátečních podmínek* nabízí
několik připravených konfigurací.  *Figure-eight 3-body orbit* je známé
periodické řešení tří stejných hmotností, které v Newtonově teorii kreslí
osmičku.  *Lagrange equilateral triple* ukazuje tři stejná tělesa v rotujícím
rovnostranném trojúhelníku.  *Binary plus intruder* představuje téměř kruhovou
dvojici narušenou lehčím třetím tělesem.  *N-body disk, N = 6* je hračkový disk
s jedním centrálním tělesem a několika lehčími tělesy.  Tlačítko *Načíst
zvolený preset* přepíše počáteční hmotnosti, polohy, rychlosti i doporučené
parametry daného scénáře.

**Ovládání aplikace.**  Nejprve lze zvolit jazyk a případně obnovit výchozí
nastavení.  V levém panelu se vybírá preset, počet těles $N$, gravitační
konstanta $G$, softening $\epsilon$, délka simulace, integrační krok $\Delta t$,
parametry 1PN modelu a způsob zobrazení 3D boxu.  V části *Počáteční hmotnosti,
polohy a rychlosti* lze každému tělesu samostatně nastavit $m_i$, $x_i,y_i,z_i$
a $v_{x,i},v_{y,i},v_{z,i}$.  Protože výpočet trajektorie je nejdražší část,
změny fyzikálních sliderů se do simulace promítnou až po tlačítku *Použít a přepočítat*.
Čistě vizuální volby, například velikost značek, rozsah 3D boxu nebo počet
animačních snímků, jsou mimo tento formulář a nevyvolávají novou numerickou
integraci. Samotné přehrávání probíhá v prohlížeči pomocí tlačítek *Play*,
*Pause* a *Reset* nad grafem.  Graf lze ručně otáčet, přibližovat a posouvat nástroji
Plotly.  Trajektorie se po stisku *Použít a přepočítat* nejprve spočítají a uloží,
ale při animaci se kreslí pouze již proletěná část dráhy.  Budoucí část dráhy
se tedy nezobrazuje před pohybem těles.
            """
        )
        st.latex(r"\ddot{\mathbf r}_i=-\sum_{j\ne i}Gm_j\frac{\mathbf r_i-\mathbf r_j}{\left(|\mathbf r_i-\mathbf r_j|^2+\epsilon^2\right)^{3/2}}")
        st.markdown(
            r"""
Softening $\epsilon$ je numerická regularizace singularity při velmi blízkém
průletu.  Není to fyzikální relativistický efekt; pouze brání nekonečné síle
při $r_{ij}\to0$.
            """
        )
        st.latex(r"\mathbf a_i=\mathbf a_i^{\rm Newton}+\lambda_{\rm 1PN}\sum_{j\ne i}\mathbf a_{ij}^{\rm pairwise\;1PN}")
        st.markdown(
            r"""
Párový 1PN člen používá standardní dvoutělesovou relativní 1PN akceleraci v
harmonických souřadnicích,
            """
        )
        st.latex(r"\mathbf a_{ij}^{\rm rel,1PN}=\frac{GM}{c^2r^2}\left[\mathbf n\left((4+2\eta)\frac{GM}{r}-(1+3\eta)v^2+\frac32\eta\dot r^2\right)+(4-2\eta)\dot r\,\mathbf v\right]")
        st.markdown(
            r"""
kde $M=m_i+m_j$, $\eta=m_im_j/M^2$, $\mathbf n=(\mathbf r_i-\mathbf r_j)/r$,
$\mathbf v=\mathbf v_i-\mathbf v_j$ a $\dot r=\mathbf n\cdot\mathbf v$.
Tato aplikace **neobsahuje úplné Einsteinovy-Infeldovy-Hoffmannovy vícetělesové
1PN členy**, proto pravý panel není přesná relativistická efemerida.  Je to
názorná párová aproximace.  Integrace probíhá klasickou metodou Runge--Kutta 4.
řádu:
            """
        )
        st.latex(r"\mathbf y_{n+1}=\mathbf y_n+\frac{\Delta t}{6}(\mathbf k_1+2\mathbf k_2+2\mathbf k_3+\mathbf k_4)")
        st.markdown(
            r"""
Diagnostiky $\max(v/c)$ a $\max(Gm/(rc^2))$ ukazují, zda je slabopolní a
pomalá 1PN aproximace ještě rozumná.  Pokud jsou tyto hodnoty velké, simulace
může být vizuálně zajímavá, ale nemá kvantitativní relativistický význam.
            """
        )
    else:
        st.markdown(
            r"""
This app solves a toy 3D gravitational N-body problem.  The left panel computes
Newtonian gravity.  The right panel computes Newtonian gravity plus a pairwise
two-body 1PN correction inspired by general relativity.  The model uses
arbitrary dimensionless units: length $L_0$, time $T_0$, and mass $M_0$.

For each body the mass, initial position, and initial velocity can be changed.
The parameter $G$ controls the Newtonian gravitational strength in model units.
The parameter $c$ has units $L_0/T_0$ and controls the size of the relativistic
correction.  Larger $c$ brings the right panel closer to the Newtonian limit.
The 1PN multiplier is only an educational magnifier for the relativistic terms.
            """
        )
        st.markdown(
            r"""
**Default scenarios.**  The *Initial-condition preset* menu provides several
ready-made configurations.  *Figure-eight 3-body orbit* is the well-known
periodic equal-mass Newtonian solution in which the three bodies follow a common
figure-eight curve.  *Lagrange equilateral triple* shows three equal masses in a
rotating equilateral triangle.  *Binary plus intruder* starts from a nearly
circular binary perturbed by a lighter third body.  *N-body disk, N = 6* is a toy
disk-like system with one central mass and several lighter bodies.  Pressing
*Load selected preset* replaces the initial masses, positions, velocities, and
recommended numerical settings for the selected scenario.

**How to control the app.**  The sidebar contains the language selector and the
reset button.  Inside the control form one can select the preset, the number of
bodies $N$, the gravitational constant $G$, the softening $\epsilon$, the
integration time, the RK4 step $\Delta t$, the 1PN parameters, and the 3D
view-box behavior.  In *Initial masses, positions and velocities* each body can
be edited separately through $m_i$, $x_i,y_i,z_i$ and
$v_{x,i},v_{y,i},v_{z,i}$.  Since recomputing the trajectory is the expensive
step, physical slider changes are applied only after pressing *Apply and recompute*.
Purely visual controls, such as marker sizes, 3D view-box scaling, and the
number of animation frames, are outside this form and do not trigger a new
numerical integration. Playback itself runs in the browser through the *Play*,
*Pause*, and *Reset* buttons above the graph.  The Plotly view can be rotated, zoomed, and panned
manually.  After *Apply and recompute* the trajectories are precomputed and
stored, but the animation draws only the already-travelled part of each path.
Future trajectory segments are not shown before the bodies move.
            """
        )
        st.latex(r"\ddot{\mathbf r}_i=-\sum_{j\ne i}Gm_j\frac{\mathbf r_i-\mathbf r_j}{\left(|\mathbf r_i-\mathbf r_j|^2+\epsilon^2\right)^{3/2}}")
        st.markdown(
            r"""
The softening $\epsilon$ is a numerical regularization of close encounters.  It
is not a relativistic effect; it only prevents an infinite force when
$r_{ij}\to0$.
            """
        )
        st.latex(r"\mathbf a_i=\mathbf a_i^{\rm Newton}+\lambda_{\rm 1PN}\sum_{j\ne i}\mathbf a_{ij}^{\rm pairwise\;1PN}")
        st.markdown(
            r"""
The pairwise 1PN term uses the standard two-body relative 1PN acceleration in
harmonic-coordinate form,
            """
        )
        st.latex(r"\mathbf a_{ij}^{\rm rel,1PN}=\frac{GM}{c^2r^2}\left[\mathbf n\left((4+2\eta)\frac{GM}{r}-(1+3\eta)v^2+\frac32\eta\dot r^2\right)+(4-2\eta)\dot r\,\mathbf v\right]")
        st.markdown(
            r"""
where $M=m_i+m_j$, $\eta=m_im_j/M^2$, $\mathbf n=(\mathbf r_i-\mathbf r_j)/r$,
$\mathbf v=\mathbf v_i-\mathbf v_j$, and $\dot r=\mathbf n\cdot\mathbf v$.
The app **does not include the full Einstein--Infeld--Hoffmann many-body 1PN
terms**, so the right panel is not a precision relativistic ephemeris.  It is a
visual pairwise approximation.  Time integration uses the classical fourth-order
Runge--Kutta method:
            """
        )
        st.latex(r"\mathbf y_{n+1}=\mathbf y_n+\frac{\Delta t}{6}(\mathbf k_1+2\mathbf k_2+2\mathbf k_3+\mathbf k_4)")
        st.markdown(
            r"""
The diagnostics $\max(v/c)$ and $\max(Gm/(rc^2))$ indicate whether the weak-field,
slow-motion 1PN approximation is still reasonable.  If they become large, the
simulation can remain visually interesting but should not be interpreted as a
quantitatively valid relativistic model.
            """
        )

    st.markdown(f"**{t('sources')}**")
    if language == "Čeština":
        st.markdown(
            """
- I. Newton, *Philosophiae Naturalis Principia Mathematica* (1687), klasická formulace gravitačního zákona.
- A. Einstein, "Die Feldgleichungen der Gravitation", *Sitzungsberichte der Königlich Preußischen Akademie der Wissenschaften* (1915), Einsteinovy rovnice pole.
- A. Einstein, L. Infeld, B. Hoffmann, "The Gravitational Equations and the Problem of Motion", *Annals of Mathematics* **39**, 65--100 (1938), DOI: 10.2307/1968714.
- L. Blanchet, "Gravitational Radiation from Post-Newtonian Sources and Inspiralling Compact Binaries", *Living Reviews in Relativity* **17**, 2 (2014), DOI: 10.12942/lrr-2014-2.
- A. Chenciner and R. Montgomery, "A remarkable periodic solution of the three-body problem in the case of equal masses", *Annals of Mathematics* **152**, 881--901 (2000), DOI: 10.2307/2661357.
- J. C. Butcher, "A history of Runge-Kutta methods", *Applied Numerical Mathematics* **20**, 247--260 (1996), DOI: 10.1016/0168-9274(95)00108-5.
- H. C. Plummer, *An Introductory Treatise on Dynamical Astronomy* (1918), klasický zdroj pro gravitační změkčení typu Plummerova potenciálu.
            """
        )
    else:
        st.markdown(
            """
- I. Newton, *Philosophiae Naturalis Principia Mathematica* (1687), classical inverse-square gravity.
- A. Einstein, "Die Feldgleichungen der Gravitation", *Sitzungsberichte der Königlich Preußischen Akademie der Wissenschaften* (1915), Einstein field equations.
- A. Einstein, L. Infeld, B. Hoffmann, "The Gravitational Equations and the Problem of Motion", *Annals of Mathematics* **39**, 65--100 (1938), DOI: 10.2307/1968714.
- L. Blanchet, "Gravitational Radiation from Post-Newtonian Sources and Inspiralling Compact Binaries", *Living Reviews in Relativity* **17**, 2 (2014), DOI: 10.12942/lrr-2014-2.
- A. Chenciner and R. Montgomery, "A remarkable periodic solution of the three-body problem in the case of equal masses", *Annals of Mathematics* **152**, 881--901 (2000), DOI: 10.2307/2661357.
- J. C. Butcher, "A history of Runge-Kutta methods", *Applied Numerical Mathematics* **20**, 247--260 (1996), DOI: 10.1016/0168-9274(95)00108-5.
- H. C. Plummer, *An Introductory Treatise on Dynamical Astronomy* (1918), classical reference for Plummer-type gravitational softening.
            """
        )

# All expensive physical controls are inside a form.  Moving sliders no longer
# reruns the whole app; values are submitted together by Apply and recompute.
with st.sidebar.form("nbody_controls_form"):
    st.header(t("global_controls"))
    preset_name = st.selectbox(t("preset"), tuple(PRESETS.keys()), key="preset")
    load_preset_clicked = st.form_submit_button(t("load_preset"), use_container_width=True)
    note = PRESETS[preset_name].note_cs if language == "Čeština" else PRESETS[preset_name].note_en
    st.caption(f"**{t('preset_note')}:** {note}")

    st.slider(t("n_bodies"), 2, MAX_BODIES, key="n_bodies")
    st.slider(t("g_value"), 0.05, 5.0, step=0.05, key="g_value")
    st.slider(t("softening"), 0.0, 0.10, step=0.001, key="softening")

    st.header(t("time_controls"))
    st.slider(t("total_time"), 0.5, 80.0, step=0.5, key="total_time")
    st.slider(t("dt"), 0.001, 0.10, step=0.001, key="dt")
    st.slider(t("frame_stride"), 1, 50, step=1, key="frame_stride")

    st.header(t("relativity"))
    st.slider(t("log10_c"), 0.5, 5.0, step=0.05, key="log10_c")
    st.caption(f"c = {10.0 ** float(st.session_state['log10_c']):.4g} L0/T0")
    st.slider(t("pn_log10"), -4.0, 6.0, step=0.1, key="pn_log10")
    st.caption(f"1PN multiplier = {10.0 ** float(st.session_state['pn_log10']):.4g}")

    n_for_widgets = int(st.session_state.get("n_bodies", DEFAULTS["n_bodies"]))
    with st.expander(t("body_parameters"), expanded=False):
        for i in range(n_for_widgets):
            st.markdown(f"**{t('body_i')} {i + 1}**")
            st.slider(t("mass"), 0.0, 10.0, step=0.01, key=f"m_{i}")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.number_input(t("x"), min_value=-10.0, max_value=10.0, step=0.01, format="%.6f", key=f"x_{i}")
            with c2:
                st.number_input(t("y"), min_value=-10.0, max_value=10.0, step=0.01, format="%.6f", key=f"y_{i}")
            with c3:
                st.number_input(t("z"), min_value=-10.0, max_value=10.0, step=0.01, format="%.6f", key=f"z_{i}")
            v1, v2, v3 = st.columns(3)
            with v1:
                st.number_input(t("vx"), min_value=-10.0, max_value=10.0, step=0.01, format="%.6f", key=f"vx_{i}")
            with v2:
                st.number_input(t("vy"), min_value=-10.0, max_value=10.0, step=0.01, format="%.6f", key=f"vy_{i}")
            with v3:
                st.number_input(t("vz"), min_value=-10.0, max_value=10.0, step=0.01, format="%.6f", key=f"vz_{i}")

    apply_clicked = st.form_submit_button("Apply and recompute" if language == "English" else "Použít a přepočítat", use_container_width=True)

if load_preset_clicked:
    st.session_state["_load_preset_requested"] = preset_name
    st.rerun()
if apply_clicked:
    st.session_state["manual_frame"] = 0

# Visual-only controls are intentionally outside the expensive Apply form.
# They may rerun the Streamlit script, but they do not change the cached
# integration inputs, so the numerical trajectories are reused.
st.sidebar.header(t("visual_controls_no_recompute"))
st.sidebar.slider(t("axis_half_range"), 0.2, 10.0, step=0.1, key="axis_half_range")
st.sidebar.selectbox(
    t("axis_scaling"),
    ("fixed", "full", "dynamic"),
    format_func=lambda value: {
        "fixed": t("axis_fixed"),
        "full": t("axis_full"),
        "dynamic": t("axis_dynamic"),
    }.get(value, str(value)),
    key="axis_scaling",
)
st.sidebar.slider(t("marker_base"), 2.0, 18.0, step=0.5, key="marker_base")
st.sidebar.slider(t("marker_mass_gamma"), 0.05, 1.0, step=0.05, key="marker_mass_gamma")
st.sidebar.slider(t("max_animation_frames"), 20, 300, step=10, key="max_animation_frames")
st.sidebar.slider(t("animation_frame_duration"), 10, 200, step=10, key="animation_frame_duration")
st.sidebar.slider(t("orbit_curve_points"), 100, 2500, step=100, key="orbit_curve_points")

st.info(t("units_caption"))
st.info(t("browser_animation_note"))

# Use the submitted/applied session-state values.
n_bodies = int(st.session_state["n_bodies"])
g_value = float(st.session_state["g_value"])
softening = float(st.session_state["softening"])
total_time = float(st.session_state["total_time"])
dt = float(st.session_state["dt"])
frame_stride = int(st.session_state["frame_stride"])
log10_c = float(st.session_state["log10_c"])
c_value = 10.0 ** log10_c
pn_log10 = float(st.session_state["pn_log10"])
axis_half_range = float(st.session_state["axis_half_range"])
axis_scaling_mode = str(st.session_state["axis_scaling"])
marker_base = float(st.session_state["marker_base"])
marker_mass_gamma = float(st.session_state["marker_mass_gamma"])
max_animation_frames = int(st.session_state["max_animation_frames"])
animation_frame_duration = int(st.session_state["animation_frame_duration"])
orbit_curve_points = int(st.session_state["orbit_curve_points"])

pos0, vel0, masses0 = collect_initial_conditions(n_bodies)
pos_tuple = tuple(tuple(float(v) for v in row) for row in pos0)
vel_tuple = tuple(tuple(float(v) for v in row) for row in vel0)
masses_tuple = tuple(float(v) for v in masses0)

n_step_estimate = int(math.ceil(total_time / dt))
frame_estimate = int(math.ceil(n_step_estimate / max(frame_stride, 1))) + 1
st.sidebar.caption(f"RK4 steps: {n_step_estimate:,}; displayed frames: about {frame_estimate:,}")
if n_step_estimate > 60_000:
    st.error(t("warning_steps"))
    st.stop()

with st.spinner("Integrating trajectories..." if language == "English" else "Integruji trajektorie..."):
    times, frames_n, frames_p, masses, diag_n, diag_p, e0, energy_drift = simulate_cached(
        n_bodies=n_bodies,
        masses_tuple=masses_tuple,
        pos_tuple=pos_tuple,
        vel_tuple=vel_tuple,
        g_value=g_value,
        softening=softening,
        total_time=total_time,
        dt=dt,
        frame_stride=frame_stride,
        c_value=c_value,
        pn_log10=pn_log10,
    )

st.subheader(t("playback"))
fig = make_figure(
    times=times,
    frames_n=frames_n,
    frames_p=frames_p,
    masses=masses,
    frame_index=0,
    trail_frames=0,
    axis_half_range=axis_half_range,
    axis_scaling_mode=axis_scaling_mode,
    base_marker=marker_base,
    mass_gamma=marker_mass_gamma,
    animate=True,
    max_animation_frames=max_animation_frames,
    orbit_curve_points=orbit_curve_points,
    animation_frame_duration=animation_frame_duration,
)
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
st.caption(t("fixed_axes_note"))
st.caption(t("caption"))

st.subheader(t("export"))
st.caption(t("protocol_note"))
protocol_text = build_protocol_text(
    language=language,
    preset_name=str(st.session_state["preset"]),
    n_bodies=n_bodies,
    masses=masses,
    initial_pos=pos0,
    initial_vel=vel0,
    times=times,
    frames_n=frames_n,
    frames_p=frames_p,
    g_value=g_value,
    softening=softening,
    dt=dt,
    frame_stride=frame_stride,
    c_value=c_value,
    pn_log10=pn_log10,
    axis_scaling_mode=axis_scaling_mode,
    axis_half_range=axis_half_range,
    diag_n=diag_n,
    diag_p=diag_p,
    energy_drift=energy_drift,
)
st.download_button(
    t("download_protocol"),
    data=protocol_text.encode("utf-8"),
    file_name="nbody_simulation_protocol.txt",
    mime="text/plain",
    use_container_width=True,
)

st.caption(t("gif_note"))
gif_frames = st.slider(t("export_gif_frames"), 20, 180, int(st.session_state.get("export_gif_frames", DEFAULTS["export_gif_frames"])), step=10, key="export_gif_frames")
export_signature = repr((
    language, n_bodies, masses_tuple, pos_tuple, vel_tuple, g_value, softening,
    total_time, dt, frame_stride, c_value, pn_log10, axis_half_range,
    axis_scaling_mode, marker_base, marker_mass_gamma, gif_frames,
    animation_frame_duration, orbit_curve_points,
))
if st.button(t("generate_gif"), use_container_width=True):
    with st.spinner(t("gif_generating")):
        st.session_state["export_gif_bytes"] = render_animation_gif(
            times=times,
            frames_n=frames_n,
            frames_p=frames_p,
            masses=masses,
            axis_half_range=axis_half_range,
            axis_scaling_mode=axis_scaling_mode,
            base_marker=marker_base,
            mass_gamma=marker_mass_gamma,
            max_gif_frames=int(gif_frames),
            frame_duration_ms=animation_frame_duration,
            orbit_curve_points=orbit_curve_points,
        )
        st.session_state["export_gif_signature"] = export_signature
if st.session_state.get("export_gif_signature") == export_signature and st.session_state.get("export_gif_bytes"):
    st.success(t("gif_ready"))
    st.download_button(
        t("download_gif"),
        data=st.session_state["export_gif_bytes"],
        file_name="nbody_animation.gif",
        mime="image/gif",
        use_container_width=True,
    )

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(t("displayed_time"), f"0.000 T0")
with c2:
    st.metric("N", f"{n_bodies}")
with c3:
    st.metric("1PN multiplier", f"{10.0 ** pn_log10:.3g}×")
with c4:
    st.metric("Newton energy drift", f"{energy_drift:.3e}")

st.subheader(t("diagnostics"))
d1, d2, d3, d4 = st.columns(4)
d1.metric("Newton max v/c", f"{diag_n['max_v_over_c']:.3e}")
d2.metric("Newton max Gm/(rc²)", f"{diag_n['max_GM_over_rc2']:.3e}")
d3.metric("1PN max v/c", f"{diag_p['max_v_over_c']:.3e}")
d4.metric("Min separation", f"{min(diag_n['min_separation'], diag_p['min_separation']):.3e} L0")

if max(diag_n["max_v_over_c"], diag_p["max_v_over_c"]) > 0.3 or max(diag_n["max_GM_over_rc2"], diag_p["max_GM_over_rc2"]) > 0.1:
    st.warning(t("pn_warning"))

st.subheader(t("current_params"))
rows = []
labels = body_labels(n_bodies)
for i in range(n_bodies):
    rows.append(
        {
            t("body"): labels[i],
            t("model_mass"): masses[i],
            "x0 [L0]": pos0[i, 0],
            "y0 [L0]": pos0[i, 1],
            "z0 [L0]": pos0[i, 2],
            "vx0 [L0/T0]": vel0[i, 0],
            "vy0 [L0/T0]": vel0[i, 1],
            "vz0 [L0/T0]": vel0[i, 2],
        }
    )
st.dataframe(rows, hide_index=True, use_container_width=True)
