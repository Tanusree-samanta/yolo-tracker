"""
streamlit.py
----------------
Streamlit web application for Real-Time Object Detection & Tracking.
Redesigned UI: Military HUD / Cyberpunk 3D aesthetic.

Features:
  • Upload a video file → process frame-by-frame with live preview
  • Live webcam stream via streamlit-webrtc (WebRTC)
  • Full sidebar controls: model, confidence, classes, tracker
  • Per-class object count panel
  • Download processed video
  • Stop button mid-processing

Run:
    streamlit run streamlit.py
"""

from __future__ import annotations

import os
import cv2
import time
import tempfile
import numpy as np
import streamlit as st
from pathlib import Path

from detector import ObjectDetector
from tracker import TrackerFactory, SORTTracker
from utils import (
    draw_track,
    draw_fps,
    draw_object_count,
    format_label,
    FPSCounter,
    ObjectCounter,
)

# ── Optional WebRTC ─────────────────────────────────────────────────────
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# Page configuration
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NEXUS · Object Detection",
    page_icon="⊹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

# ── Master CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500;600&display=swap');

  /* ─── HIDE STREAMLIT CHROME ─── */
  #MainMenu, [data-testid="stMainMenu"],
  [data-testid="stToolbarActions"], [data-testid="stDeployButton"],
  [class*="_viewerBadge_"], [class*="viewerBadge_"],
  footer { display: none !important; visibility: hidden !important; }

  /* ─── ROOT VARS ─── */
  :root {
    --c-bg:       #030507;
    --c-panel:    rgba(6, 14, 20, 0.92);
    --c-lime:     #39ff14;
    --c-cyan:     #00e5ff;
    --c-amber:    #ffab00;
    --c-red:      #ff2d55;
    --c-violet:   #bf5af2;
    --c-dim:      rgba(57,255,20,0.12);
    --c-border:   rgba(57,255,20,0.22);
    --c-glow:     0 0 18px rgba(57,255,20,0.35);
    --c-glow-c:   0 0 18px rgba(0,229,255,0.35);
    --font-hud:   'Orbitron', monospace;
    --font-mono:  'Share Tech Mono', monospace;
    --font-body:  'Inter', sans-serif;
  }

  /* ─── GLOBAL BASE ─── */
  html, body, .stApp {
    background-color: var(--c-bg) !important;
    font-family: var(--font-body);
    color: #c8d8c8;
  }

  /* Animated grid background */
  .stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(57,255,20,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(57,255,20,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }
  /* Vignette */
  .stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, transparent 40%, rgba(0,0,0,0.6) 100%);
    pointer-events: none;
    z-index: 0;
  }

  /* ─── SCANLINE OVERLAY ─── */
  @keyframes scanmove {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
  }
  .stApp > * { position: relative; z-index: 1; }

  /* ─── SIDEBAR ─── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050c0a 0%, #030809 100%) !important;
    border-right: 1px solid var(--c-border) !important;
    box-shadow: 4px 0 30px rgba(57,255,20,0.08);
  }
  section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--c-lime), transparent);
    box-shadow: 0 0 20px var(--c-lime);
  }
  /* Sidebar text */
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stMarkdown p,
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] .stMultiSelect label,
  section[data-testid="stSidebar"] .stToggle label {
    color: var(--c-lime) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em;
  }

  /* ─── TABS ─── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(3,5,7,0.95);
    border-bottom: 1px solid var(--c-border);
    padding: 4px 4px 0 4px;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: var(--font-hud) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    color: rgba(57,255,20,0.55) !important;
    border-radius: 6px 6px 0 0 !important;
    border: 1px solid transparent !important;
    border-bottom: none !important;
    padding: 10px 22px !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
    position: relative;
    overflow: hidden;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: var(--c-lime) !important;
    border-color: var(--c-border) !important;
    background: rgba(57,255,20,0.05) !important;
  }
  .stTabs [aria-selected="true"] {
    color: #000 !important;
    background: var(--c-lime) !important;
    border-color: var(--c-lime) !important;
    box-shadow: 0 0 22px rgba(57,255,20,0.6), inset 0 0 10px rgba(255,255,255,0.2) !important;
    font-weight: 900 !important;
  }

  /* ─── BUTTONS ─── */
  .stButton > button {
    background: transparent !important;
    color: var(--c-lime) !important;
    border: 1px solid var(--c-lime) !important;
    border-radius: 4px !important;
    font-family: var(--font-hud) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 10px 28px !important;
    position: relative;
    overflow: hidden;
    transition: all 0.2s !important;
    clip-path: polygon(8px 0%, 100% 0%, calc(100% - 8px) 100%, 0% 100%);
  }
  .stButton > button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(57,255,20,0.15), transparent);
    opacity: 0;
    transition: opacity 0.2s;
  }
  .stButton > button:hover {
    background: rgba(57,255,20,0.12) !important;
    box-shadow: var(--c-glow) !important;
    transform: translateY(-1px) !important;
  }
  .stButton > button:active {
    transform: translateY(1px) scale(0.98) !important;
  }

  /* ─── DOWNLOAD BUTTON ─── */
  .stDownloadButton > button {
    background: transparent !important;
    color: var(--c-cyan) !important;
    border: 1px solid var(--c-cyan) !important;
    border-radius: 4px !important;
    font-family: var(--font-hud) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    clip-path: polygon(8px 0%, 100% 0%, calc(100% - 8px) 100%, 0% 100%);
    transition: all 0.2s !important;
  }
  .stDownloadButton > button:hover {
    background: rgba(0,229,255,0.12) !important;
    box-shadow: var(--c-glow-c) !important;
  }

  /* ─── PROGRESS BAR ─── */
  .stProgress > div > div {
    background: linear-gradient(90deg, var(--c-lime), var(--c-cyan)) !important;
    box-shadow: 0 0 12px rgba(57,255,20,0.5) !important;
  }
  .stProgress > div {
    background: rgba(57,255,20,0.08) !important;
    border: 1px solid rgba(57,255,20,0.15) !important;
  }

  /* ─── FILE UPLOADER ─── */
  /* ─── FILE UPLOADER ─── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(57,255,20,0.3) !important;
    border-radius: 8px !important;
    background: rgba(57,255,20,0.02) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(57,255,20,0.6) !important;
    box-shadow: var(--c-glow) !important;
}

/* Hide ALL internal text/labels except the button */
[data-testid="stFileUploaderDropzone"] > div > div:first-child,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] > div > span {
    display: none !important;
}

/* Style the button cleanly */
[data-testid="stFileUploader"] button {
    background: rgba(57,255,20,0.08) !important;
    border: 1px solid var(--c-lime) !important;
    color: var(--c-lime) !important;
    font-family: var(--font-hud) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 4px !important;
}

/* Replace button text */
[data-testid="stFileUploader"] button p {
    font-size: 0 !important;
}
[data-testeid="stFileUploader"] button p::after {
    content: "BROWSE FILES";
    font-size: 0.7rem !important;
    font-family: var(--font-hud) !important;
    color: var(--c-lime) !important;
}

  /* ─── METRIC CARDS ─── */
  div[data-testid="metric-container"] {
    background: var(--c-panel);
    border: 1px solid var(--c-border);
    border-top: 2px solid var(--c-lime);
    border-radius: 4px;
    padding: 14px 16px;
    position: relative;
    clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%);
    transition: box-shadow 0.3s;
  }
  div[data-testid="metric-container"]:hover {
    box-shadow: var(--c-glow);
  }
  div[data-testid="metric-container"] label {
    color: var(--c-amber) !important;
    font-family: var(--font-hud) !important;
    font-size: 0.6rem !important;
    text-transform: uppercase;
    letter-spacing: 2px;
  }
  div[data-testid="metric-container"] [data-testid="metric-value"] {
    color: var(--c-lime) !important;
    font-family: var(--font-hud) !important;
    font-size: 1.6rem !important;
    font-weight: 900 !important;
    text-shadow: 0 0 20px rgba(57,255,20,0.5) !important;
  }

  /* ─── SELECTS / INPUTS ─── */
  [data-baseweb="select"] > div:first-child,
  [data-baseweb="select"] > div {
    background: #060e0a !important;
    border: 1px solid rgba(57,255,20,0.25) !important;
    color: var(--c-lime) !important;
    font-family: var(--font-mono) !important;
    border-radius: 4px !important;
  }
  [data-baseweb="select"] span, [data-baseweb="select"] div { color: var(--c-lime) !important; }
  [data-baseweb="popover"] ul, [data-baseweb="menu"] {
    background: #060e0a !important;
    border: 1px solid var(--c-border) !important;
  }
  .stTextInput input {
    background: #060e0a !important;
    color: var(--c-lime) !important;
    border: 1px solid rgba(57,255,20,0.25) !important;
    font-family: var(--font-mono) !important;
    border-radius: 4px !important;
  }
  [data-baseweb="tag"] {
    background: rgba(57,255,20,0.15) !important;
    border: 1px solid rgba(57,255,20,0.3) !important;
  }
  [data-baseweb="tag"] span { color: var(--c-lime) !important; }

  /* Slider */
  .stSlider [data-testid="stTickBarMin"],
  .stSlider [data-testid="stTickBarMax"],
  .stSlider p { color: var(--c-amber) !important; font-family: var(--font-mono) !important; }

  /* ─── ALERTS ─── */
  .stAlert { border-radius: 4px !important; }

  /* ─── CAPTIONS ─── */
  .stCaption, [data-testid="stCaptionContainer"] p,
  div[data-testid="stCaptionContainer"] * {
    color: rgba(57,255,20,0.7) !important;
    font-family: var(--font-mono) !important;
  }

  /* ─── PARAGRAPH TEXT ─── */
  .main .block-container .stMarkdown p,
  .main .block-container .stMarkdown span,
  .main .block-container .stMarkdown li {
    color: #a0b8a0 !important;
  }
  hr { border-color: rgba(57,255,20,0.12) !important; }

  /* ══════════════════════════════════════════
     CUSTOM COMPONENT STYLES
  ══════════════════════════════════════════ */

  /* ─── HUD PANEL ─── */
  .hud-panel {
    background: var(--c-panel);
    border: 1px solid var(--c-border);
    border-radius: 4px;
    padding: 18px 20px;
    position: relative;
    margin-bottom: 14px;
    clip-path: polygon(0 0, calc(100% - 16px) 0, 100% 16px, 100% 100%, 16px 100%, 0 calc(100% - 16px));
  }
  .hud-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, var(--c-lime), transparent 60%);
    opacity: 0.6;
  }
  .hud-panel::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 40%, var(--c-cyan));
    opacity: 0.4;
  }

  /* ─── SECTION LABEL ─── */
  .sec-label {
    font-family: var(--font-hud);
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 6px 14px 6px 10px;
    margin-bottom: 14px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    position: relative;
  }
  .sec-label::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    box-shadow: 0 0 6px currentColor;
    animation: blink 1.6s step-start infinite;
    flex-shrink: 0;
  }
  @keyframes blink {
    0%, 100% { opacity: 1; } 50% { opacity: 0.2; }
  }
  .sl-lime  { color: var(--c-lime);   border-left: 2px solid var(--c-lime); }
  .sl-cyan  { color: var(--c-cyan);   border-left: 2px solid var(--c-cyan); }
  .sl-amber { color: var(--c-amber);  border-left: 2px solid var(--c-amber); }
  .sl-red   { color: var(--c-red);    border-left: 2px solid var(--c-red); }
  .sl-lime::before  { background: var(--c-lime); }
  .sl-cyan::before  { background: var(--c-cyan); }
  .sl-amber::before { background: var(--c-amber); }
  .sl-red::before   { background: var(--c-red); }

  /* ─── 3D INFO CARDS ─── */
  .info-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 14px 0;
    perspective: 800px;
  }
  .info-card {
    background: var(--c-panel);
    border: 1px solid rgba(57,255,20,0.2);
    border-top: 2px solid;
    border-radius: 4px;
    padding: 14px 12px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.3s, box-shadow 0.3s;
    clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
    transform-style: preserve-3d;
  }
  .info-card:hover {
    transform: rotateX(-4deg) rotateY(3deg) translateY(-4px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.5), 0 0 20px rgba(57,255,20,0.2);
  }
  .info-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.04), transparent 60%);
    pointer-events: none;
  }
  .ic-lime  { border-top-color: var(--c-lime);   }
  .ic-cyan  { border-top-color: var(--c-cyan);   }
  .ic-amber { border-top-color: var(--c-amber);  }
  .ic-violet{ border-top-color: var(--c-violet); }
  .ic-label {
    font-family: var(--font-hud);
    font-size: 0.58rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(255,171,0,0.8);
    margin-bottom: 6px;
  }
  .ic-value {
    font-family: var(--font-hud);
    font-size: 1.45rem;
    font-weight: 900;
    margin: 4px 0;
    text-shadow: 0 0 20px currentColor;
  }
  .ic-lime  .ic-value { color: var(--c-lime); }
  .ic-cyan  .ic-value { color: var(--c-cyan); }
  .ic-amber .ic-value { color: var(--c-amber); }
  .ic-violet .ic-value { color: var(--c-violet); }
  .ic-sub {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: rgba(160,180,160,0.6);
  }

  /* ─── LIVE INDICATOR ─── */
  @keyframes liveneon {
    0%, 100% { box-shadow: 0 0 6px var(--c-red), 0 0 12px var(--c-red); opacity: 1; }
    50%       { box-shadow: 0 0 3px var(--c-red); opacity: 0.6; }
  }
  .live-dot {
    display: inline-block;
    width: 9px; height: 9px;
    background: var(--c-red);
    border-radius: 50%;
    animation: liveneon 1.2s ease-in-out infinite;
    vertical-align: middle;
    margin-right: 6px;
  }
  .live-badge {
    display: inline-flex;
    align-items: center;
    background: rgba(255,45,85,0.1);
    border: 1px solid rgba(255,45,85,0.35);
    border-radius: 3px;
    padding: 6px 14px;
    font-family: var(--font-hud);
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--c-red);
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }

  /* ─── CLASS BADGES ─── */
  .class-badge {
    display: inline-block;
    font-family: var(--font-hud);
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-radius: 3px;
    padding: 4px 12px;
    margin: 3px 4px;
    clip-path: polygon(6px 0%, 100% 0%, calc(100% - 6px) 100%, 0% 100%);
  }
  .cb-lime   { background: rgba(57,255,20,0.15);  color: var(--c-lime);   border: 1px solid rgba(57,255,20,0.35);  }
  .cb-cyan   { background: rgba(0,229,255,0.15);  color: var(--c-cyan);   border: 1px solid rgba(0,229,255,0.35);  }
  .cb-amber  { background: rgba(255,171,0,0.15);  color: var(--c-amber);  border: 1px solid rgba(255,171,0,0.35);  }
  .cb-red    { background: rgba(255,45,85,0.15);  color: var(--c-red);    border: 1px solid rgba(255,45,85,0.35);  }
  .cb-violet { background: rgba(191,90,242,0.15); color: var(--c-violet); border: 1px solid rgba(191,90,242,0.35); }
  .cb-white  { background: rgba(255,255,255,0.08);color: #e0ffe0;          border: 1px solid rgba(255,255,255,0.2); }

  /* ─── SUCCESS / COMPLETE BANNER ─── */
  .complete-banner {
    background: rgba(57,255,20,0.05);
    border: 1px solid rgba(57,255,20,0.3);
    border-left: 3px solid var(--c-lime);
    border-radius: 4px;
    padding: 14px 20px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 14px;
    position: relative;
    clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%);
  }
  .complete-banner-icon {
    font-size: 1.4rem;
    filter: drop-shadow(0 0 8px var(--c-lime));
  }
  .complete-banner-title {
    font-family: var(--font-hud);
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--c-lime);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    text-shadow: 0 0 12px rgba(57,255,20,0.5);
  }
  .complete-banner-sub {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--c-amber);
    margin-top: 3px;
  }

  /* ─── EMPTY STATE ─── */
  .empty-state {
    text-align: center;
    padding: 72px 24px;
    border: 1px dashed rgba(57,255,20,0.2);
    border-radius: 6px;
    margin-top: 20px;
    background: radial-gradient(ellipse at 50% 50%, rgba(57,255,20,0.04) 0%, transparent 70%);
    position: relative;
    overflow: hidden;
  }
  .empty-state::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(57,255,20,0.015) 2px,
      rgba(57,255,20,0.015) 4px
    );
    pointer-events: none;
  }
  .empty-state-title {
    font-family: var(--font-hud);
    font-size: 1.4rem;
    font-weight: 900;
    color: var(--c-lime);
    text-shadow: 0 0 20px rgba(57,255,20,0.4);
    letter-spacing: 0.05em;
    margin: 14px 0 10px 0;
  }
  .empty-state-sub {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: rgba(160,180,160,0.7);
    margin-bottom: 24px;
  }
  .fmt-pill {
    display: inline-block;
    font-family: var(--font-hud);
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    padding: 4px 14px;
    margin: 4px;
    border-radius: 3px;
    clip-path: polygon(4px 0%, 100% 0%, calc(100% - 4px) 100%, 0% 100%);
  }

  /* ─── ANIMATED CORNER BRACKETS ─── */
  .bracket-box {
    position: relative;
    padding: 20px;
    margin: 14px 0;
  }
  .bracket-box::before, .bracket-box::after {
    content: '';
    position: absolute;
    width: 20px; height: 20px;
    border-color: var(--c-lime);
    border-style: solid;
    opacity: 0.5;
    transition: opacity 0.3s, width 0.3s, height 0.3s;
  }
  .bracket-box::before {
    top: 0; left: 0;
    border-width: 2px 0 0 2px;
  }
  .bracket-box::after {
    bottom: 0; right: 0;
    border-width: 0 2px 2px 0;
  }
  .bracket-box:hover::before,
  .bracket-box:hover::after {
    width: 30px; height: 30px; opacity: 1;
  }

  /* ─── SIDEBAR SECTION HEADER ─── */
  .sb-section {
    font-family: var(--font-hud);
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    padding: 5px 10px;
    margin: 12px 0 8px 0;
    border-left: 2px solid;
    display: flex; align-items: center; gap: 7px;
  }
  .sb-lime   { color: var(--c-lime);   border-color: var(--c-lime); }
  .sb-cyan   { color: var(--c-cyan);   border-color: var(--c-cyan); }
  .sb-amber  { color: var(--c-amber);  border-color: var(--c-amber); }
  .sb-violet { color: var(--c-violet); border-color: var(--c-violet); }

  /* ─── HERO SECTION ─── */
  @keyframes rotatering {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }
  @keyframes rotateringo {
    from { transform: rotate(0deg); }
    to   { transform: rotate(-360deg); }
  }
  @keyframes pulsecore {
    0%, 100% { opacity: 0.7; transform: scale(1); }
    50%       { opacity: 1;   transform: scale(1.1); }
  }
  @keyframes scanx {
    0%   { transform: translateX(-100%); opacity: 0; }
    50%  { opacity: 0.6; }
    100% { transform: translateX(100%); opacity: 0; }
  }
  .hero-crosshair {
    width: 120px; height: 120px;
    position: relative;
    margin: 0 auto 18px auto;
  }
  .hc-ring1 {
    position: absolute;
    inset: 4px;
    border: 1.5px solid rgba(57,255,20,0.4);
    border-radius: 50%;
    border-top-color: var(--c-lime);
    animation: rotatering 3s linear infinite;
  }
  .hc-ring2 {
    position: absolute;
    inset: 16px;
    border: 1.5px dashed rgba(0,229,255,0.4);
    border-radius: 50%;
    border-bottom-color: var(--c-cyan);
    animation: rotateringo 4s linear infinite;
  }
  .hc-ring3 {
    position: absolute;
    inset: 28px;
    border: 1px solid rgba(255,171,0,0.3);
    border-radius: 50%;
    animation: rotatering 5s linear infinite;
  }
  .hc-core {
    position: absolute;
    inset: 40px;
    background: radial-gradient(circle, rgba(57,255,20,0.6), rgba(0,229,255,0.3), transparent);
    border-radius: 50%;
    animation: pulsecore 2s ease-in-out infinite;
  }
  /* crosshair lines */
  .hc-h, .hc-v {
    position: absolute;
    background: rgba(57,255,20,0.35);
  }
  .hc-h { top: 50%; left: 0; right: 0; height: 1px; transform: translateY(-50%); }
  .hc-v { left: 50%; top: 0; bottom: 0; width: 1px; transform: translateX(-50%); }

  /* ─── TABLE (About tab) ─── */
  table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    width: 100% !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
  }
  table thead th {
    background: rgba(57,255,20,0.1) !important;
    color: var(--c-lime) !important;
    font-family: var(--font-hud) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 10px 16px !important;
    border-bottom: 1px solid var(--c-border) !important;
  }
  table tbody td {
    background: rgba(3,5,7,0.6) !important;
    color: #a8c8a8 !important;
    padding: 9px 16px !important;
    border-bottom: 1px solid rgba(57,255,20,0.07) !important;
  }
  table tbody tr:hover td {
    background: rgba(57,255,20,0.04) !important;
    color: var(--c-lime) !important;
  }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Sidebar – Settings panel
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Brand ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
      background: rgba(57,255,20,0.04);
      border: 1px solid rgba(57,255,20,0.25);
      border-top: 2px solid var(--c-lime, #39ff14);
      border-radius: 4px;
      padding: 18px 12px 14px 12px;
      margin-bottom: 16px;
      text-align: center;
      clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%);
      position: relative; overflow: hidden;
    ">
      <div style="
        font-family:'Orbitron',monospace; font-size:1.5rem; font-weight:900;
        background: linear-gradient(135deg, #39ff14, #00e5ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: 0.15em; text-shadow: none;
      ">NEXUS</div>
      <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
        color:rgba(57,255,20,0.55); letter-spacing:0.2em; margin-top:4px;">
        DETECTION · TRACKING
      </div>
      <div style="
        margin-top:10px; display:flex; justify-content:center; gap:6px;
      ">
        <span style="font-family:'Orbitron',monospace; font-size:0.55rem; letter-spacing:0.12em;
          color:#000; background:#39ff14; padding:2px 8px; border-radius:2px; font-weight:700;">
          YOLOv8
        </span>
        <span style="font-family:'Orbitron',monospace; font-size:0.55rem; letter-spacing:0.12em;
          color:#000; background:#00e5ff; padding:2px 8px; border-radius:2px; font-weight:700;">
          SORT
        </span>
        <span style="font-family:'Orbitron',monospace; font-size:0.55rem; letter-spacing:0.12em;
          color:#000; background:#ffab00; padding:2px 8px; border-radius:2px; font-weight:700;">
          OpenCV
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model ────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section sb-lime">⬡ MODEL</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Weights",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="Nano=fastest, X=most accurate. Larger models need more VRAM.",
    )
    conf_threshold = st.slider("Confidence", 0.1, 0.95, 0.25, 0.05,
                               help="Min detection confidence (0–1).")
    iou_threshold  = st.slider("NMS IOU", 0.1, 0.95, 0.40, 0.05,
                               help="Non-max suppression IOU threshold.")

    st.markdown("---")

    # ── Tracker ──────────────────────────────────────────────────────
    st.markdown('<div class="sb-section sb-cyan">⬡ TRACKER</div>', unsafe_allow_html=True)
    tracker_type = st.selectbox(
        "Algorithm",
        ["sort", "deepsort"] if WEBRTC_AVAILABLE else ["sort"],
        help="SORT is built-in. DeepSORT needs `pip install deep-sort-realtime`.",
    )
    max_age   = st.slider("Max age (frames)", 5, 100, 30, 5)
    min_hits  = st.slider("Min hits (SORT)",  1,  10,  3, 1)
    track_iou = st.slider("Tracker IOU",      0.1, 0.9, 0.3, 0.05)

    st.markdown("---")

    # ── Class filter ─────────────────────────────────────────────────
    st.markdown('<div class="sb-section sb-amber">⬡ CLASS FILTER</div>', unsafe_allow_html=True)
    COMMON_CLASSES = [
        "person", "car", "truck", "bus", "bicycle", "motorbike",
        "cat", "dog", "bottle", "chair", "laptop", "cell phone",
    ]
    selected_classes = st.multiselect(
        "Track only these",
        COMMON_CLASSES,
        default=[],
        help="Leave empty to track ALL 80 COCO classes.",
    )
    custom_class = st.text_input("Add custom class", placeholder="e.g. aeroplane")
    if custom_class:
        selected_classes.append(custom_class.strip())
    target_classes = selected_classes if selected_classes else None

    st.markdown("---")

    # ── Display options ───────────────────────────────────────────────
    st.markdown('<div class="sb-section sb-violet">⬡ DISPLAY</div>', unsafe_allow_html=True)
    show_fps    = st.toggle("Show FPS overlay",    value=True)
    show_count  = st.toggle("Show object counts",  value=True)
    frame_skip  = st.slider("Frame skip", 1, 5, 2, 1,
                            help="Process every Nth frame.")
    resize_width = st.selectbox(
        "Processing width (px)",
        [320, 480, 640, 854, 1280], index=2,
    )

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:10px 0;">
      <div style="font-family:'Share Tech Mono',monospace; font-size:0.6rem;
        color:rgba(57,255,20,0.35); letter-spacing:0.15em;">
        SYS STATUS <span style="color:#39ff14;">● ONLINE</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Cached model loader
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False, max_entries=1)
def load_detector(model_path, conf, iou, classes, imgsz):
    return ObjectDetector(
        model_path=model_path,
        conf_threshold=conf,
        iou_threshold=iou,
        target_classes=classes if classes else None,
        imgsz=imgsz,
    )


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_LINEAR)


def annotate_frame(frame, detector, tracker, fps_counter, obj_counter,
                   use_fps=True, use_count=True):
    detections = detector.detect(frame)
    if isinstance(tracker, SORTTracker):
        tracks = tracker.update(detections)
    else:
        tracks = tracker.update(detections, frame)
    fps_counter.tick()
    obj_counter.update(tracks, detector.class_names)
    annotated = frame.copy()
    for trk in tracks:
        label = format_label(
            detector.get_class_name(trk["class_id"]),
            trk["track_id"], trk["confidence"],
        )
        draw_track(annotated, trk, label)
    if use_fps:   draw_fps(annotated, fps_counter.fps)
    if use_count: draw_object_count(annotated, obj_counter.counts)
    return annotated, tracks


_BADGE_COLORS = ["cb-lime", "cb-cyan", "cb-amber", "cb-red", "cb-violet", "cb-white"]

def make_badges(counts: dict) -> str:
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    html = ""
    for i, (cls, n) in enumerate(sorted_items):
        c = _BADGE_COLORS[i % len(_BADGE_COLORS)]
        html += f"<span class='class-badge {c}'>{cls} &nbsp;⁞&nbsp; {n}</span>"
    return html


# ═══════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="
  text-align: center;
  padding: 40px 20px 32px 20px;
  background:
    radial-gradient(ellipse at 30% 50%, rgba(57,255,20,0.05) 0%, transparent 55%),
    radial-gradient(ellipse at 70% 50%, rgba(0,229,255,0.04) 0%, transparent 55%),
    linear-gradient(180deg, rgba(57,255,20,0.02) 0%, transparent 100%);
  border: 1px solid rgba(57,255,20,0.15);
  border-radius: 6px;
  margin-bottom: 22px;
  position: relative;
  overflow: hidden;
  clip-path: polygon(0 0, calc(100% - 22px) 0, 100% 22px, 100% 100%, 22px 100%, 0 calc(100% - 22px));
">

  <!-- scanline sweep -->
  <div style="
    position:absolute; top:0; left:-100%; right:-100%; height:2px;
    background: linear-gradient(90deg, transparent, rgba(57,255,20,0.5), transparent);
    animation: scanx 4s linear infinite;
    pointer-events:none;
  "></div>

  <!-- corner marks -->
  <div style="position:absolute;top:10px;left:10px;width:18px;height:18px;
    border-top:2px solid rgba(57,255,20,0.5);border-left:2px solid rgba(57,255,20,0.5);"></div>
  <div style="position:absolute;top:10px;right:10px;width:18px;height:18px;
    border-top:2px solid rgba(0,229,255,0.5);border-right:2px solid rgba(0,229,255,0.5);"></div>
  <div style="position:absolute;bottom:10px;left:10px;width:18px;height:18px;
    border-bottom:2px solid rgba(0,229,255,0.4);border-left:2px solid rgba(0,229,255,0.4);"></div>
  <div style="position:absolute;bottom:10px;right:10px;width:18px;height:18px;
    border-bottom:2px solid rgba(57,255,20,0.4);border-right:2px solid rgba(57,255,20,0.4);"></div>

  <!-- crosshair -->
  <div class="hero-crosshair">
    <div class="hc-ring1"></div>
    <div class="hc-ring2"></div>
    <div class="hc-ring3"></div>
    <div class="hc-core"></div>
    <div class="hc-h"></div>
    <div class="hc-v"></div>
  </div>

  <h1 style="
    font-family: 'Orbitron', monospace;
    font-size: 2.5rem;
    font-weight: 900;
    margin: 0 0 6px 0;
    letter-spacing: 0.12em;
    background: linear-gradient(135deg, #39ff14 0%, #00e5ff 50%, #ffab00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    line-height: 1.15;
  ">OBJECT DETECTION</h1>
  <h2 style="
    font-family: 'Orbitron', monospace;
    font-size: 0.9rem;
    font-weight: 400;
    color: rgba(0,229,255,0.7);
    letter-spacing: 0.35em;
    text-transform: uppercase;
    margin: 0 0 18px 0;
  ">& TRACKING SYSTEM</h2>

  <div style="display:flex; justify-content:center; gap:6px; flex-wrap:wrap; margin-top:4px;">
    <span style="font-family:'Orbitron',monospace; font-size:0.6rem; letter-spacing:0.15em; font-weight:700;
      color:#000; background:#39ff14; padding:3px 14px; border-radius:2px;
      clip-path: polygon(6px 0%, 100% 0%, calc(100% - 6px) 100%, 0% 100%);">⚡ REAL-TIME</span>
    <span style="font-family:'Orbitron',monospace; font-size:0.6rem; letter-spacing:0.15em; font-weight:700;
      color:#000; background:#00e5ff; padding:3px 14px; border-radius:2px;
      clip-path: polygon(6px 0%, 100% 0%, calc(100% - 6px) 100%, 0% 100%);">◈ YOLOv8</span>
    <span style="font-family:'Orbitron',monospace; font-size:0.6rem; letter-spacing:0.15em; font-weight:700;
      color:#000; background:#ffab00; padding:3px 14px; border-radius:2px;
      clip-path: polygon(6px 0%, 100% 0%, calc(100% - 6px) 100%, 0% 100%);">⊹ 80 CLASSES</span>
    <span style="font-family:'Orbitron',monospace; font-size:0.6rem; letter-spacing:0.15em; font-weight:700;
      color:#000; background:#bf5af2; padding:3px 14px; border-radius:2px;
      clip-path: polygon(6px 0%, 100% 0%, calc(100% - 6px) 100%, 0% 100%);">▣ LIVE STATS</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════

tab_upload, tab_webcam, tab_about = st.tabs([
    "⬡  UPLOAD VIDEO",
    "◈  LIVE WEBCAM",
    "⊹  SYSTEM INFO",
])


# ───────────────────────────────────────────────────────────────────────
# TAB 1 – Upload Video
# ───────────────────────────────────────────────────────────────────────

with tab_upload:

    st.markdown('<div class="sec-label sl-lime">⬡ &nbsp; TARGET ACQUISITION · UPLOAD VIDEO</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
    "Upload",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    label_visibility="collapsed",
)

    if uploaded_file is not None:

        suffix = Path(uploaded_file.name).suffix
        tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_input.write(uploaded_file.read())
        tmp_input.flush()
        tmp_input.close()

        cap_info = cv2.VideoCapture(tmp_input.name)
        total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps      = cap_info.get(cv2.CAP_PROP_FPS) or 30.0
        src_w        = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h        = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s   = total_frames / src_fps
        cap_info.release()

        # 3D Info cards
        st.markdown(f"""
        <div class="info-grid">
          <div class="info-card ic-lime">
            <div class="ic-label">Resolution</div>
            <div class="ic-value">{src_w}×{src_h}</div>
            <div class="ic-sub">pixels</div>
          </div>
          <div class="info-card ic-cyan">
            <div class="ic-label">Frames</div>
            <div class="ic-value">{total_frames:,}</div>
            <div class="ic-sub">total</div>
          </div>
          <div class="info-card ic-amber">
            <div class="ic-label">Duration</div>
            <div class="ic-value">{duration_s:.1f}s</div>
            <div class="ic-sub">seconds</div>
          </div>
          <div class="info-card ic-violet">
            <div class="ic-label">Frame Rate</div>
            <div class="ic-value">{src_fps:.0f}</div>
            <div class="ic-sub">fps source</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption("⊹ Configure model, thresholds, and classes in the sidebar · then press START PROCESSING")

        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
        with col_btn2:
            start = st.button("▶  START PROCESSING", use_container_width=True)

        if start:
            with st.spinner(f"Initializing {model_choice} …"):
                detector = load_detector(
                    model_choice, conf_threshold, iou_threshold,
                    tuple(target_classes) if target_classes else None,
                    int(resize_width),
                )

            tracker = TrackerFactory.create(
                tracker_type, max_age=max_age,
                min_hits=min_hits, iou_threshold=track_iou, reset_ids=True,
            )
            fps_ctr = FPSCounter(window=30)
            obj_ctr = ObjectCounter()

            # Live preview header
            st.markdown("""
            <div style="margin:20px 0 10px 0;">
              <span class="live-badge"><span class="live-dot"></span>LIVE FEED ACTIVE</span>
            </div>
            """, unsafe_allow_html=True)

            frame_placeholder = st.empty()

            prog_col, stat_col = st.columns([3, 1])
            with prog_col:
                progress_bar  = st.progress(0.0)
                progress_text = st.empty()
            with stat_col:
                fps_display = st.empty()

            st.markdown('<div class="sec-label sl-amber" style="margin-top:16px;">◈ &nbsp; TELEMETRY</div>',
                        unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            metric_fps   = m1.empty()
            metric_objs  = m2.empty()
            metric_frame = m3.empty()
            metric_ids   = m4.empty()

            count_display = st.empty()

            # ── Output temp file ──────────────────────────────────────
            tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_output.close()
            out_w = resize_width
            cap   = cv2.VideoCapture(tmp_input.name)
            out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * (out_w / cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            writer = cv2.VideoWriter(
                tmp_output.name,
                cv2.VideoWriter_fourcc(*"mp4v"),
                src_fps / max(frame_skip, 1),
                (out_w, out_h),
            )

            frame_idx = 0
            all_ids: set[int] = set()
            stop_btn = st.button("⏹  STOP PROCESSING", key="stop_upload")

            while cap.isOpened():
                if stop_btn:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                frame    = resize_frame(frame, resize_width)
                annotated, tracks = annotate_frame(
                    frame, detector, tracker, fps_ctr, obj_ctr, show_fps, show_count,
                )
                writer.write(annotated)

                for trk in tracks:
                    all_ids.add(trk["track_id"])

                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

                pct = frame_idx / max(total_frames, 1)
                progress_bar.progress(min(pct, 1.0))
                progress_text.caption(f"⊹ Frame {frame_idx:,} / {total_frames:,}  ·  {pct*100:.1f}%")
                fps_display.caption(f"⚡ {fps_ctr.fps:.1f} fps")

                metric_fps.metric("⚡ FPS",          f"{fps_ctr.fps:.1f}")
                metric_objs.metric("⊹ Active tracks", len(tracks))
                metric_frame.metric("▣ Frame",         frame_idx)
                metric_ids.metric("◈ Unique IDs",     len(all_ids))

                counts = obj_ctr.counts
                if counts:
                    count_display.markdown(
                        "<div style='margin-top:10px;'>" + make_badges(counts) + "</div>",
                        unsafe_allow_html=True,
                    )

            cap.release()
            writer.release()
            progress_bar.progress(1.0)
            progress_text.caption("✔ Processing complete")

            # ── Success banner ────────────────────────────────────────
            st.markdown(f"""
            <div class="complete-banner">
              <div class="complete-banner-icon">✔</div>
              <div>
                <div class="complete-banner-title">MISSION COMPLETE</div>
                <div class="complete-banner-sub">
                  {frame_idx:,} frames processed &nbsp;·&nbsp;
                  {len(all_ids)} unique object IDs tracked
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Download ──────────────────────────────────────────────
            st.markdown('<div class="sec-label sl-cyan" style="margin-top:16px;">⬡ &nbsp; EXPORT PROCESSED FEED</div>',
                        unsafe_allow_html=True)
            with open(tmp_output.name, "rb") as f:
                video_bytes = f.read()

            dl1, dl2, dl3 = st.columns([2, 1, 2])
            with dl2:
                st.download_button(
                    label="⬇  DOWNLOAD MP4",
                    data=video_bytes,
                    file_name=f"nexus_tracked_{uploaded_file.name}",
                    mime="video/mp4",
                    use_container_width=True,
                )

            try:
                os.unlink(tmp_input.name)
                os.unlink(tmp_output.name)
            except Exception:
                pass

    else:
        st.markdown("""
        <div class="empty-state">
          <div style="font-size:3.5rem; filter:drop-shadow(0 0 18px #39ff14); margin-bottom:4px;">⊹</div>
          <div class="empty-state-title">AWAITING TARGET FEED</div>
          <div class="empty-state-sub">Upload a video file to begin detection &amp; tracking</div>
          <div>
            <span class="fmt-pill" style="background:rgba(57,255,20,0.12);color:#39ff14;
              border:1px solid rgba(57,255,20,0.3);">.MP4</span>
            <span class="fmt-pill" style="background:rgba(0,229,255,0.12);color:#00e5ff;
              border:1px solid rgba(0,229,255,0.3);">.AVI</span>
            <span class="fmt-pill" style="background:rgba(255,171,0,0.12);color:#ffab00;
              border:1px solid rgba(255,171,0,0.3);">.MOV</span>
            <span class="fmt-pill" style="background:rgba(191,90,242,0.12);color:#bf5af2;
              border:1px solid rgba(191,90,242,0.3);">.MKV</span>
            <span class="fmt-pill" style="background:rgba(255,45,85,0.12);color:#ff2d55;
              border:1px solid rgba(255,45,85,0.3);">.WEBM</span>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────
# TAB 2 – Live Webcam
# ───────────────────────────────────────────────────────────────────────

with tab_webcam:

    st.markdown('<div class="sec-label sl-red">◈ &nbsp; LIVE ACQUISITION · WEBCAM FEED</div>',
                unsafe_allow_html=True)

    if WEBRTC_AVAILABLE:
        st.info(
            "◈ **WebRTC mode active** — click **START** to begin live detection. "
            "Allow camera access when prompted.",
            icon="◈",
        )

        st.session_state["wb_model"]     = model_choice
        st.session_state["wb_conf"]      = conf_threshold
        st.session_state["wb_iou"]       = iou_threshold
        st.session_state["wb_classes"]   = tuple(target_classes) if target_classes else None
        st.session_state["wb_tracker"]   = tracker_type
        st.session_state["wb_max_age"]   = max_age
        st.session_state["wb_min_hits"]  = min_hits
        st.session_state["wb_track_iou"] = track_iou
        st.session_state["wb_fps"]       = show_fps
        st.session_state["wb_count"]     = show_count
        st.session_state["wb_width"]     = resize_width

        st.caption("⊹ Settings are locked at START · press STOP then START again to apply changes")

        class LiveVideoProcessor(VideoProcessorBase):
            def __init__(self):
                ss = st.session_state
                self.detector = ObjectDetector(
                    model_path=ss.get("wb_model", "yolov8n.pt"),
                    conf_threshold=ss.get("wb_conf", 0.25),
                    iou_threshold=ss.get("wb_iou", 0.40),
                    target_classes=ss.get("wb_classes", None),
                    imgsz=int(ss.get("wb_width", 640)),
                )
                self.tracker  = TrackerFactory.create(
                    ss.get("wb_tracker", "sort"),
                    max_age=ss.get("wb_max_age", 30),
                    min_hits=ss.get("wb_min_hits", 3),
                    iou_threshold=ss.get("wb_track_iou", 0.3),
                    reset_ids=True,
                )
                self.fps_ctr  = FPSCounter(window=30)
                self.obj_ctr  = ObjectCounter()
                self.width    = ss.get("wb_width", 640)
                self.use_fps  = ss.get("wb_fps", True)
                self.use_cnt  = ss.get("wb_count", True)

            def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
                img = frame.to_ndarray(format="bgr24")
                img = resize_frame(img, self.width)
                annotated, _ = annotate_frame(
                    img, self.detector, self.tracker,
                    self.fps_ctr, self.obj_ctr,
                    self.use_fps, self.use_cnt,
                )
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        RTC_CONFIG = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        webrtc_streamer(
            key="live-tracker",
            video_processor_factory=LiveVideoProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    else:
        st.warning(
            "⚠ `streamlit-webrtc` not installed. Falling back to local OpenCV webcam.  \n"
            "For WebRTC: `pip install streamlit-webrtc av`",
        )

        cam_col1, cam_col2, cam_col3 = st.columns([2, 1, 2])
        with cam_col2:
            start_cam = st.button("▶  START WEBCAM", use_container_width=True)

        if start_cam:
            with st.spinner(f"Loading {model_choice} …"):
                detector = load_detector(
                    model_choice, conf_threshold, iou_threshold,
                    tuple(target_classes) if target_classes else None,
                    int(resize_width),
                )

            tracker = TrackerFactory.create(
                tracker_type, max_age=max_age,
                min_hits=min_hits, iou_threshold=track_iou, reset_ids=True,
            )
            fps_ctr = FPSCounter(window=30)
            obj_ctr = ObjectCounter()

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("⊗ Cannot open webcam. Ensure a camera is connected.")
                st.stop()

            st.markdown("""
            <div style="margin:16px 0 10px 0;">
              <span class="live-badge"><span class="live-dot"></span>LIVE FEED ACTIVE</span>
            </div>
            """, unsafe_allow_html=True)

            frame_slot  = st.empty()
            count_slot  = st.empty()
            stop_cam    = st.button("⏹  STOP WEBCAM", key="stop_cam")

            while cap.isOpened():
                if stop_cam:
                    break
                ok, frame = cap.read()
                if not ok:
                    st.warning("⚠ Lost camera feed.")
                    break

                frame = resize_frame(frame, resize_width)
                annotated, tracks = annotate_frame(
                    frame, detector, tracker,
                    fps_ctr, obj_ctr, show_fps, show_count,
                )

                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_slot.image(rgb, channels="RGB", use_container_width=True)

                counts = obj_ctr.counts
                if counts:
                    count_slot.markdown(
                        "<div style='margin-top:10px;'>" + make_badges(counts) + "</div>",
                        unsafe_allow_html=True,
                    )

            cap.release()
            st.info("◈ Webcam feed stopped.")


# ───────────────────────────────────────────────────────────────────────
# TAB 3 – About / System Info
# ───────────────────────────────────────────────────────────────────────

with tab_about:

    st.markdown('<div class="sec-label sl-cyan">⊹ &nbsp; SYSTEM INFORMATION</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="hud-panel">
          <div style="font-family:'Orbitron',monospace; font-size:0.65rem; letter-spacing:0.2em;
            color:#ffab00; text-transform:uppercase; margin-bottom:14px;">⬡ Technology Stack</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        | Component | Library |
        |---|---|
        | Detection | YOLOv8 (Ultralytics) |
        | Tracking | SORT (Kalman + Hungarian) |
        | Vision | OpenCV |
        | Web App | Streamlit |
        | Webcam | streamlit-webrtc |
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="hud-panel" style="margin-top:14px;">
          <div style="font-family:'Orbitron',monospace; font-size:0.65rem; letter-spacing:0.2em;
            color:#ffab00; text-transform:uppercase; margin-bottom:14px;">⬡ Architecture</div>
          <pre style="font-family:'Share Tech Mono',monospace; font-size:0.75rem;
            color:#39ff14; background:transparent; border:none; margin:0; padding:0;">
VideoCapture (OpenCV)
  │
  ▼
ObjectDetector (YOLOv8)
  │  [x1,y1,x2,y2, conf, cls]
  ▼
SORTTracker (Kalman)
  │  {track_id, bbox, conf, cls}
  ▼
draw_track() → frame
  │
  ▼
st.image() / WebRTC stream
          </pre>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="hud-panel">
          <div style="font-family:'Orbitron',monospace; font-size:0.65rem; letter-spacing:0.2em;
            color:#ffab00; text-transform:uppercase; margin-bottom:14px;">⬡ Controls</div>
          <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem;
            color:#a0b8a0; line-height:2;">
            <span style="color:#39ff14;">Sidebar</span> — configure before starting:<br>
            &nbsp;· Model size (speed vs accuracy)<br>
            &nbsp;· Confidence &amp; IOU thresholds<br>
            &nbsp;· Class filter (person, car …)<br>
            &nbsp;· Tracker settings<br>
            &nbsp;· Display options<br><br>
            <span style="color:#00e5ff;">During processing:</span><br>
            &nbsp;· ⏹ Stop button halts run<br>
            &nbsp;· ⬇ Download processed MP4
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="hud-panel" style="margin-top:14px;">
          <div style="font-family:'Orbitron',monospace; font-size:0.65rem; letter-spacing:0.2em;
            color:#ffab00; text-transform:uppercase; margin-bottom:14px;">⬡ Project Files</div>
          <pre style="font-family:'Share Tech Mono',monospace; font-size:0.78rem;
            color:#a0b8a0; background:transparent; border:none; margin:0; padding:0;">
streamlit.py   ← this app
main.py        ← CLI entry point
detector.py    ← YOLOv8 wrapper
tracker.py     ← SORT / DeepSORT
utils.py       ← drawing helpers
          </pre>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:20px;">
      <div class="hud-panel bracket-box" style="padding:18px 20px;">
        <div style="font-family:'Orbitron',monospace; font-size:0.6rem; letter-spacing:0.2em;
          color:#ffab00; text-transform:uppercase; margin-bottom:10px;">⚡ Quick Setup</div>
        <pre style="font-family:'Share Tech Mono',monospace; font-size:0.78rem;
          color:#39ff14; background:transparent; border:none; margin:0; padding:0;">
pip install -r requirements.txt
streamlit run streamlit.py
        </pre>
      </div>
      <div class="hud-panel bracket-box" style="padding:18px 20px;">
        <div style="font-family:'Orbitron',monospace; font-size:0.6rem; letter-spacing:0.2em;
          color:#ffab00; text-transform:uppercase; margin-bottom:10px;">◈ Cloud-Ready</div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.78rem;
          color:#a0b8a0; line-height:1.8;">
          CPU-only PyTorch build keeps image<br>
          size minimal for Streamlit Cloud.<br>
          <span style="color:#00e5ff;">Zero GPU required to deploy. ⊹</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding:16px 0; border-top:1px solid rgba(57,255,20,0.12);">
      <div style="font-family:'Orbitron',monospace; font-size:0.6rem; letter-spacing:0.25em;
        color:rgba(57,255,20,0.4);">BUILT WITH</div>
      <div style="margin-top:8px; display:flex; justify-content:center; gap:8px; flex-wrap:wrap;">
        <span style="font-family:'Orbitron',monospace; font-size:0.65rem; font-weight:700;
          color:#39ff14; letter-spacing:0.1em; text-shadow:0 0 10px rgba(57,255,20,0.5);">YOLOv8</span>
        <span style="color:rgba(57,255,20,0.3);">·</span>
        <span style="font-family:'Orbitron',monospace; font-size:0.65rem; font-weight:700;
          color:#00e5ff; letter-spacing:0.1em; text-shadow:0 0 10px rgba(0,229,255,0.5);">SORT</span>
        <span style="color:rgba(57,255,20,0.3);">·</span>
        <span style="font-family:'Orbitron',monospace; font-size:0.65rem; font-weight:700;
          color:#ffab00; letter-spacing:0.1em; text-shadow:0 0 10px rgba(255,171,0,0.5);">OpenCV</span>
        <span style="color:rgba(57,255,20,0.3);">·</span>
        <span style="font-family:'Orbitron',monospace; font-size:0.65rem; font-weight:700;
          color:#bf5af2; letter-spacing:0.1em; text-shadow:0 0 10px rgba(191,90,242,0.5);">Streamlit</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
