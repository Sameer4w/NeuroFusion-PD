
from __future__ import annotations

import html
import io
import os
import re
import tempfile
import textwrap
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from inference.pipeline import analyze_patient

try:
    from supabase import create_client
except ImportError:  # pragma: no cover
    create_client = None


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NeuroFusion-PD",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# GLOBAL STYLES
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       DESIGN TOKENS
    ===================================================== */

    :root {
        --nf-bg: #040912;
        --nf-bg-2: #071321;
        --nf-bg-3: #0a1828;

        --nf-panel: rgba(13, 29, 48, 0.84);
        --nf-panel-soft: rgba(16, 35, 57, 0.72);

        --nf-border: rgba(148, 163, 184, 0.15);
        --nf-border-hover: rgba(103, 232, 249, 0.34);
        --nf-border-strong: rgba(148, 163, 184, 0.25);

        --nf-text: #f8fafc;
        --nf-text-soft: #dbe7f3;
        --nf-muted: #9fb0c5;
        --nf-muted-2: #71849b;

        --nf-purple: #7c3aed;
        --nf-purple-light: #a78bfa;

        --nf-cyan: #22d3ee;
        --nf-cyan-light: #67e8f9;

        --nf-success: #34d399;
        --nf-warning: #fbbf24;
        --nf-danger: #fb7185;

        --nf-shadow-sm: 0 6px 20px rgba(0, 0, 0, 0.16);
        --nf-shadow-md: 0 14px 42px rgba(0, 0, 0, 0.24);
        --nf-shadow-lg: 0 28px 80px rgba(0, 0, 0, 0.34);

        --nf-transition:
            all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
    }


    /* =====================================================
       PAGE
    ===================================================== */

    html,
    body,
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 15% 5%,
                rgba(124, 58, 237, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 12%,
                rgba(34, 211, 238, 0.07),
                transparent 24%
            ),
            linear-gradient(
                180deg,
                var(--nf-bg),
                var(--nf-bg-2) 48%,
                var(--nf-bg)
            ) !important;

        color: var(--nf-text);
    }


    [data-testid="stAppViewContainer"] {
        min-height: 100vh;
    }


    [data-testid="stAppViewContainer"] > .main {
        background: transparent !important;
    }


    /* Hide Streamlit's default header area while preserving
       enough vertical room for our custom navigation. */

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0 !important;
        min-height: 0 !important;
        border: none !important;
    }


    div[data-testid="stDecoration"] {
        display: none !important;
    }


    .block-container {
        position: relative;
        z-index: 1;

        max-width: 1180px !important;

        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }


    /* =====================================================
       GENERAL STREAMLIT CLEANUP
    ===================================================== */

    [data-testid="stToolbar"] {
        display: none !important;
    }


    div[data-testid="stStatusWidget"] {
        display: none;
    }


    .stMarkdown {
        color: var(--nf-text-soft);
    }


    p,
    li {
        color: var(--nf-text-soft);
    }


    /* =====================================================
       NAVIGATION
    ===================================================== */

    .navbar {
        position: relative;
        z-index: 100;

        width: 100%;

        display: flex;
        align-items: center;
        justify-content: space-between;

        min-height: 62px;

        margin: 0 0 28px 0;
        padding: 6px 0;

        box-sizing: border-box;
        overflow: visible;
    }


    .nav-brand {
        display: flex;
        align-items: center;

        gap: 12px;

        min-width: 0;
        overflow: visible;

        transition: transform 0.25s ease;
    }


    .nav-brand:hover {
        transform: translateY(-1px);
    }


    .nav-logo {
        width: 44px;
        height: 44px;

        min-width: 44px;
        min-height: 44px;

        display: flex;
        align-items: center;
        justify-content: center;

        flex-shrink: 0;

        position: relative;
        z-index: 2;

        border-radius: 13px;

        background:
            linear-gradient(
                135deg,
                rgba(124, 58, 237, 0.34),
                rgba(34, 211, 238, 0.15)
            );

        border:
            1px solid rgba(124, 58, 237, 0.38);

        box-shadow:
            0 10px 28px rgba(124, 58, 237, 0.18),
            0 0 22px rgba(34, 211, 238, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 0.10);

        font-size: 1.35rem;
        line-height: 1;

        overflow: visible;

        transition:
            transform 0.25s ease,
            border-color 0.25s ease,
            box-shadow 0.25s ease,
            background 0.25s ease;
    }


    .nav-logo::before {
        content: "";

        position: absolute;
        inset: -5px;

        border-radius: 16px;

        background:
            radial-gradient(
                circle,
                rgba(124, 58, 237, 0.14),
                transparent 70%
            );

        opacity: 0;

        z-index: -1;

        transition: opacity 0.25s ease;
    }


    .nav-logo:hover {
        transform:
            translateY(-2px)
            scale(1.04)
            rotate(-2deg);

        border-color:
            rgba(103, 232, 249, 0.48);

        background:
            linear-gradient(
                135deg,
                rgba(124, 58, 237, 0.40),
                rgba(34, 211, 238, 0.20)
            );

        box-shadow:
            0 14px 34px rgba(124, 58, 237, 0.22),
            0 0 24px rgba(34, 211, 238, 0.10),
            inset 0 1px 0 rgba(255, 255, 255, 0.12);
    }


    .nav-logo:hover::before {
        opacity: 1;
    }


    .nav-name {
        color: var(--nf-text);

        font-size: 1.02rem;
        font-weight: 800;

        letter-spacing: -0.025em;
        line-height: 1.15;

        white-space: nowrap;

        transition:
            color 0.2s ease,
            text-shadow 0.2s ease;
    }


    .nav-brand:hover .nav-name {
        color: #ffffff;

        text-shadow:
            0 0 18px rgba(34, 211, 238, 0.08);
    }


    .nav-subtitle {
        color: var(--nf-muted-2);

        font-size: 0.68rem;
        font-weight: 500;

        margin-top: 4px;

        line-height: 1.2;

        white-space: nowrap;

        transition: color 0.2s ease;
    }


    .nav-brand:hover .nav-subtitle {
        color: #8fa5bc;
    }


    /* =====================================================
       HERO
    ===================================================== */

    .hero {
        position: relative;
        overflow: hidden;

        padding: 42px 44px;

        margin-bottom: 28px;

        border-radius: 26px;

        border:
            1px solid rgba(148, 163, 184, 0.16);

        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(34, 211, 238, 0.10),
                transparent 25%
            ),
            radial-gradient(
                circle at 10% 0%,
                rgba(124, 58, 237, 0.15),
                transparent 30%
            ),
            linear-gradient(
                145deg,
                rgba(16, 35, 57, 0.94),
                rgba(7, 19, 33, 0.92)
            );

        box-shadow:
            var(--nf-shadow-lg),
            inset 0 1px 0 rgba(255, 255, 255, 0.04);

        transition:
            transform 0.25s ease,
            border-color 0.25s ease,
            box-shadow 0.25s ease;
    }


    .hero::after {
        content: "";

        position: absolute;

        width: 240px;
        height: 240px;

        right: -90px;
        bottom: -120px;

        border-radius: 50%;

        background:
            radial-gradient(
                circle,
                rgba(124, 58, 237, 0.16),
                transparent 68%
            );

        pointer-events: none;
    }


    .hero:hover {
        transform: translateY(-1px);

        border-color:
            rgba(103, 232, 249, 0.20);

        box-shadow:
            0 32px 90px rgba(0, 0, 0, 0.38),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }


    .hero-kicker {
        display: flex;
        align-items: center;
        gap: 8px;

        color: var(--nf-cyan-light);

        font-size: 0.70rem;
        font-weight: 800;

        text-transform: uppercase;
        letter-spacing: 0.14em;

        margin-bottom: 14px;
    }


    .hero-kicker::before {
        content: "";

        width: 7px;
        height: 7px;

        border-radius: 50%;

        background: var(--nf-success);

        box-shadow:
            0 0 0 5px rgba(52, 211, 153, 0.08),
            0 0 16px rgba(52, 211, 153, 0.45);

        animation: pulseDot 2s ease-in-out infinite;
    }


    @keyframes pulseDot {
        0%,
        100% {
            opacity: 0.7;
            transform: scale(0.9);
        }

        50% {
            opacity: 1;
            transform: scale(1);
        }
    }


    .hero-title {
        position: relative;
        z-index: 1;

        color: #ffffff;

        font-size: clamp(2.1rem, 5vw, 4rem);
        font-weight: 850;

        line-height: 1.06;
        letter-spacing: -0.055em;

        max-width: 100%;

        margin: 0;

        overflow-wrap: normal;
        word-break: normal;
    }


    .hero-description {
        position: relative;
        z-index: 1;

        max-width: 780px;

        margin-top: 18px;

        color: var(--nf-muted);

        font-size: 1rem;
        line-height: 1.75;
    }


    /* =====================================================
       AUTH SCREEN
    ===================================================== */

    .auth-shell {
        width: 100%;
        max-width: 560px;

        margin:
            4rem auto 2rem;

        animation:
            authAppear 0.4s ease both;
    }


    @keyframes authAppear {
        from {
            opacity: 0;
            transform: translateY(15px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }


    .auth-shell .hero {
        margin-bottom: 22px;
        text-align: center;
        padding: 38px 30px;
    }


    .auth-shell .hero-kicker {
        justify-content: center;
    }


    /*
       Separate auth title.

       This is intentionally NOT .hero-title so the login
       branding cannot inherit dashboard hero sizing and
       clipping behavior.
    */

    .auth-title {
        color: #ffffff;

        display: block;

        width: 100%;

        font-size: clamp(1.65rem, 6vw, 2.65rem);
        font-weight: 850;

        line-height: 1.1;
        letter-spacing: -0.045em;

        white-space: nowrap;

        margin: 0;

        overflow: visible;
    }


    .auth-shell .hero-description {
        margin-left: auto;
        margin-right: auto;

        max-width: 500px;

        font-size: 0.92rem;
        line-height: 1.65;
    }


    .auth-note {
        text-align: center;

        color: var(--nf-muted);

        font-size: 0.78rem;
        line-height: 1.6;

        margin-top: 18px;
    }


    /* =====================================================
       SECTION CARDS
    ===================================================== */

    .section-card {
        padding: 24px 26px;

        margin: 18px 0;

        border-radius: 20px;

        border:
            1px solid var(--nf-border);

        background:
            linear-gradient(
                145deg,
                rgba(13, 29, 48, 0.84),
                rgba(9, 23, 39, 0.72)
            );

        box-shadow:
            var(--nf-shadow-sm);

        transition:
            border-color 0.22s ease,
            box-shadow 0.22s ease,
            transform 0.22s ease;
    }


    .section-card:hover {
        border-color:
            rgba(103, 232, 249, 0.16);

        box-shadow:
            0 12px 32px rgba(0, 0, 0, 0.18);
    }


    .section-title {
        color: var(--nf-text);

        font-size: 1.05rem;
        font-weight: 800;

        letter-spacing: -0.02em;
    }


    .section-subtitle {
        color: var(--nf-muted);

        margin-top: 6px;

        font-size: 0.84rem;
        line-height: 1.6;
    }


    /* =====================================================
       MODALITY GRID
    ===================================================== */

    .modality-grid {
        display: grid;

        grid-template-columns:
            repeat(3, minmax(0, 1fr));

        gap: 16px;

        margin: 18px 0;
    }


    .modality-card {
        min-height: 180px;

        padding: 24px;

        border-radius: 20px;

        border:
            1px solid var(--nf-border);

        background:
            linear-gradient(
                145deg,
                rgba(13, 29, 48, 0.82),
                rgba(8, 21, 36, 0.72)
            );

        box-shadow:
            var(--nf-shadow-sm);

        transition:
            transform 0.22s ease,
            border-color 0.22s ease,
            background 0.22s ease,
            box-shadow 0.22s ease;
    }


    .modality-card:hover {
        transform: translateY(-3px);

        border-color:
            rgba(103, 232, 249, 0.28);

        box-shadow:
            0 16px 36px rgba(0, 0, 0, 0.22);
    }


    .modality-card.selected {
        border-color:
            rgba(103, 232, 249, 0.38);

        background:
            linear-gradient(
                145deg,
                rgba(35, 34, 76, 0.90),
                rgba(8, 28, 45, 0.84)
            );

        box-shadow:
            0 16px 40px rgba(34, 211, 238, 0.07),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }


    .modality-icon {
        width: 46px;
        height: 46px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 14px;

        background:
            rgba(124, 58, 237, 0.12);

        border:
            1px solid rgba(124, 58, 237, 0.20);

        font-size: 1.35rem;

        margin-bottom: 18px;
    }


    .modality-title {
        color: var(--nf-text);

        font-size: 0.98rem;
        font-weight: 800;
    }


    .modality-description {
        color: var(--nf-muted);

        margin-top: 8px;

        font-size: 0.78rem;
        line-height: 1.55;
    }


    /* =====================================================
       SELECTION SUMMARY
    ===================================================== */

    .selection-summary {
        display: flex;

        align-items: center;
        justify-content: space-between;

        gap: 18px;

        padding: 17px 20px;

        margin: 18px 0;

        border-radius: 16px;

        border:
            1px solid rgba(103, 232, 249, 0.18);

        background:
            rgba(13, 38, 52, 0.58);

        box-shadow:
            var(--nf-shadow-sm);
    }


    .selection-label {
        color: var(--nf-muted-2);

        font-size: 0.67rem;
        font-weight: 800;

        text-transform: uppercase;
        letter-spacing: 0.10em;
    }


    .selection-value {
        color: var(--nf-text);

        margin-top: 4px;

        font-size: 0.88rem;
        font-weight: 700;
    }


    .selection-count {
        flex-shrink: 0;

        color: var(--nf-cyan-light);

        font-size: 0.75rem;
        font-weight: 800;
    }


    /* =====================================================
       WAITING / READY
    ===================================================== */

    .waiting-card,
    .ready-card {
        display: flex;
        align-items: flex-start;

        gap: 12px;

        padding: 16px 18px;

        margin: 18px 0;

        border-radius: 15px;

        background:
            rgba(13, 29, 48, 0.68);

        border:
            1px solid var(--nf-border);
    }


    .ready-card {
        border-color:
            rgba(52, 211, 153, 0.20);

        background:
            rgba(13, 45, 42, 0.48);
    }


    .waiting-dot,
    .ready-dot {
        width: 8px;
        height: 8px;

        min-width: 8px;

        margin-top: 6px;

        border-radius: 50%;

        background: var(--nf-warning);

        box-shadow:
            0 0 12px rgba(251, 191, 36, 0.35);
    }


    .ready-dot {
        background: var(--nf-success);

        box-shadow:
            0 0 12px rgba(52, 211, 153, 0.40);
    }


    .waiting-title,
    .ready-title {
        color: var(--nf-text);

        font-size: 0.82rem;
        font-weight: 800;
    }


    .waiting-description,
    .ready-description {
        color: var(--nf-muted);

        margin-top: 4px;

        font-size: 0.76rem;
        line-height: 1.5;
    }


    /* =====================================================
       ANALYSIS METHOD
    ===================================================== */

    .analysis-method-card {
        margin-bottom: 8px;
    }


    div[role="radiogroup"] {
        gap: 10px !important;
    }


    div[role="radiogroup"] > label {
        border:
            1px solid var(--nf-border) !important;

        border-radius: 12px !important;

        background:
            rgba(8, 22, 38, 0.70) !important;

        padding: 10px 14px !important;

        transition:
            border-color 0.2s ease,
            background 0.2s ease,
            transform 0.2s ease;
    }


    div[role="radiogroup"] > label:hover {
        border-color:
            rgba(103, 232, 249, 0.26) !important;

        transform: translateY(-1px);
    }


    /* =====================================================
       INPUT CARDS
    ===================================================== */

    .input-section {
        margin-top: 28px;
    }


    .input-card {
        padding: 22px 24px;

        margin-top: 16px;

        border-radius: 18px;

        border:
            1px solid var(--nf-border);

        background:
            linear-gradient(
                145deg,
                rgba(13, 29, 48, 0.82),
                rgba(8, 21, 36, 0.72)
            );

        box-shadow:
            var(--nf-shadow-sm);
    }


    .input-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;

        gap: 18px;
    }


    .input-card-title {
        color: var(--nf-text);

        font-size: 0.95rem;
        font-weight: 800;
    }


    .input-card-description {
        color: var(--nf-muted);

        margin-top: 5px;

        font-size: 0.76rem;
        line-height: 1.5;
    }


    .input-status {
        flex-shrink: 0;

        color: var(--nf-cyan-light);

        padding: 5px 9px;

        border-radius: 999px;

        border:
            1px solid rgba(34, 211, 238, 0.18);

        background:
            rgba(34, 211, 238, 0.06);

        font-size: 0.64rem;
        font-weight: 800;

        text-transform: uppercase;
        letter-spacing: 0.08em;
    }


    /* =====================================================
       FILE UPLOADERS
    ===================================================== */

    div[data-testid="stFileUploader"] {
        margin-top: 10px;
    }


    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 14px !important;

        background:
            rgba(8, 22, 38, 0.60) !important;

        border:
            1px dashed rgba(148, 163, 184, 0.20) !important;

        transition:
            background 0.2s ease,
            border-color 0.2s ease,
            transform 0.2s ease;
    }


    div[data-testid="stFileUploaderDropzone"]:hover {
        background:
            rgba(34, 211, 238, 0.025) !important;

        border-color:
            rgba(103, 232, 249, 0.28) !important;

        transform: translateY(-1px);
    }


    /* =====================================================
       AUDIO
    ===================================================== */

    audio {
        width: 100% !important;

        margin-top: 8px;

        border-radius: 12px;
    }


    /* =====================================================
       INPUT FIELDS
    ===================================================== */

    div[data-baseweb="input"] {
        border-radius: 12px !important;

        background:
            rgba(8, 22, 38, 0.82) !important;

        border:
            1px solid var(--nf-border) !important;

        transition:
            border-color 0.2s ease,
            box-shadow 0.2s ease;
    }


    div[data-baseweb="input"]:focus-within {
        border-color:
            rgba(103, 232, 249, 0.42) !important;

        box-shadow:
            0 0 0 3px rgba(34, 211, 238, 0.06),
            0 8px 25px rgba(0, 0, 0, 0.12);
    }


    div[data-baseweb="input"] input {
        color: var(--nf-text) !important;
    }


    div[data-baseweb="input"] input::placeholder {
        color: var(--nf-muted-2) !important;
    }


    textarea {
        color: var(--nf-text) !important;

        background:
            rgba(8, 22, 38, 0.82) !important;

        border-radius: 12px !important;
    }


    /* =====================================================
       SELECTBOX
    ===================================================== */

    div[data-baseweb="select"] > div {
        border-radius: 12px !important;

        background:
            rgba(8, 22, 38, 0.82) !important;

        border:
            1px solid var(--nf-border) !important;

        transition:
            border-color 0.2s ease,
            box-shadow 0.2s ease;
    }


    div[data-baseweb="select"] > div:hover {
        border-color:
            rgba(103, 232, 249, 0.28) !important;
    }


    /* =====================================================
       AUTH TABS
    ===================================================== */

    button[data-baseweb="tab"] {
        color: var(--nf-muted) !important;

        font-weight: 700 !important;
    }


    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--nf-cyan-light) !important;
    }


    /* =====================================================
       EXPANDERS
    ===================================================== */

    div[data-testid="stExpander"] {
        border:
            1px solid var(--nf-border) !important;

        border-radius: 16px !important;

        overflow: hidden;

        background:
            rgba(10, 25, 42, 0.62);

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.10);

        transition:
            border-color 0.22s ease,
            box-shadow 0.22s ease;
    }


    div[data-testid="stExpander"]:hover {
        border-color:
            rgba(103, 232, 249, 0.20) !important;

        box-shadow:
            0 12px 34px rgba(0, 0, 0, 0.16);
    }


    /* =====================================================
       METRICS
    ===================================================== */

    div[data-testid="stMetric"] {
        padding: 18px;

        border:
            1px solid var(--nf-border);

        border-radius: 16px;

        background:
            rgba(13, 29, 48, 0.76);

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.12);

        transition:
            var(--nf-transition);
    }


    div[data-testid="stMetric"]:hover {
        transform:
            translateY(-2px);

        border-color:
            rgba(103, 232, 249, 0.20);

        box-shadow:
            0 14px 35px rgba(0, 0, 0, 0.18);
    }


    /* =====================================================
       GAIT HELP
    ===================================================== */

    .gait-help {
        margin-top: 14px;
        margin-bottom: 12px;

        padding: 18px 20px;

        border-radius: 16px;

        border:
            1px solid rgba(103, 232, 249, 0.14);

        background:
            rgba(10, 29, 45, 0.60);
    }


    .gait-help-title {
        color: var(--nf-text);

        font-size: 0.86rem;
        font-weight: 800;

        margin-bottom: 8px;
    }


    .gait-help p,
    .gait-help li {
        color: var(--nf-muted);

        font-size: 0.75rem;
        line-height: 1.6;
    }


    .gait-help ol {
        margin-top: 10px;
        margin-bottom: 4px;

        padding-left: 20px;
    }


    .gait-help code {
        color: var(--nf-cyan-light);

        background:
            rgba(34, 211, 238, 0.07);

        padding: 2px 5px;

        border-radius: 5px;
    }


    /* =====================================================
       VALIDATION
    ===================================================== */

    .validation-card {
        display: flex;
        align-items: flex-start;

        gap: 12px;

        padding: 14px 16px;

        margin-top: 10px;

        border-radius: 14px;

        border: 1px solid var(--nf-border);

        background:
            rgba(13, 29, 48, 0.66);
    }


    .validation-valid {
        border-color:
            rgba(52, 211, 153, 0.18);

        background:
            rgba(13, 45, 42, 0.36);
    }


    .validation-invalid {
        border-color:
            rgba(251, 113, 133, 0.22);

        background:
            rgba(65, 22, 34, 0.30);
    }


    .validation-icon {
        width: 24px;
        height: 24px;

        min-width: 24px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 50%;

        color: var(--nf-text);

        font-size: 0.72rem;
        font-weight: 900;

        background:
            rgba(148, 163, 184, 0.12);
    }


    .validation-valid .validation-icon {
        background:
            rgba(52, 211, 153, 0.16);

        color: var(--nf-success);
    }


    .validation-invalid .validation-icon {
        background:
            rgba(251, 113, 133, 0.14);

        color: var(--nf-danger);
    }


    .validation-title {
        color: var(--nf-text);

        font-size: 0.76rem;
        font-weight: 800;
    }


    .validation-message {
        color: var(--nf-muted);

        margin-top: 3px;

        font-size: 0.72rem;
        line-height: 1.45;
    }


    /* =====================================================
       PRIMARY BUTTONS
    ===================================================== */

    div.stButton > button {
        min-height: 42px;

        border-radius: 12px !important;

        font-weight: 750 !important;

        transition:
            transform 0.20s ease,
            box-shadow 0.20s ease,
            border-color 0.20s ease;
    }


    div.stButton > button:hover {
        transform: translateY(-1px);
    }


    div.stButton > button[kind="primary"] {
        background:
            linear-gradient(
                135deg,
                #7c3aed,
                #2563eb
            ) !important;

        border:
            1px solid rgba(167, 139, 250, 0.42) !important;

        box-shadow:
            0 10px 28px rgba(124, 58, 237, 0.20);
    }


    div.stButton > button[kind="primary"]:hover {
        box-shadow:
            0 14px 34px rgba(124, 58, 237, 0.28);
    }


    div.stButton > button:disabled {
        opacity: 0.48 !important;

        transform: none !important;

        box-shadow: none !important;
    }


    /* =====================================================
       ALERTS
    ===================================================== */

    div[data-testid="stAlert"] {
        border-radius: 14px !important;
    }


    /* =====================================================
       ACCOUNT
    ===================================================== */

    .account-card {
        padding: 7px 3px;
    }


    .account-label {
        color: var(--nf-muted-2);

        font-size: 0.68rem;

        text-transform: uppercase;
        letter-spacing: 0.08em;

        font-weight: 700;
    }


    .account-email {
        color: var(--nf-text);

        font-size: 0.84rem;
        font-weight: 650;

        margin-top: 4px;

        word-break: break-word;
    }


    /* =====================================================
       RESULT
    ===================================================== */

    .result-card {
        position: relative;
        overflow: hidden;

        padding: 30px;

        margin-top: 26px;

        border-radius: 22px;

        border:
            1px solid var(--nf-border);

        background:
            linear-gradient(
                145deg,
                rgba(13, 29, 48, 0.94),
                rgba(8, 20, 35, 0.88)
            );

        box-shadow:
            var(--nf-shadow-lg);
    }


    .result-card::after {
        content: "";

        position: absolute;

        width: 260px;
        height: 260px;

        right: -120px;
        top: -120px;

        border-radius: 50%;

        pointer-events: none;

        background:
            radial-gradient(
                circle,
                rgba(124, 58, 237, 0.16),
                transparent 70%
            );
    }


    .result-positive {
        border-color:
            rgba(251, 191, 36, 0.24);
    }


    .result-negative {
        border-color:
            rgba(52, 211, 153, 0.22);
    }


    .result-label {
        position: relative;
        z-index: 1;

        color: var(--nf-muted-2);

        font-size: 0.68rem;
        font-weight: 800;

        text-transform: uppercase;
        letter-spacing: 0.10em;

        margin-bottom: 8px;
    }


    .result-title {
        position: relative;
        z-index: 1;

        color: var(--nf-text);

        font-size: clamp(1.15rem, 3vw, 1.65rem);
        font-weight: 800;

        line-height: 1.3;

        margin-bottom: 24px;
    }


    .probability {
        position: relative;
        z-index: 1;

        color: var(--nf-cyan-light);

        font-size: clamp(2.4rem, 6vw, 4rem);
        font-weight: 850;

        line-height: 1;

        letter-spacing: -0.05em;
    }


    .probability-note {
        position: relative;
        z-index: 1;

        max-width: 620px;

        margin-top: 10px;

        color: var(--nf-muted);

        font-size: 0.76rem;
        line-height: 1.55;
    }


    /* =====================================================
       RECOMMENDATIONS
    ===================================================== */

    .recommendation-card {
        padding: 22px 24px;

        margin-top: 18px;

        border-radius: 18px;

        border:
            1px solid var(--nf-border);

        background:
            rgba(13, 29, 48, 0.72);
    }


    .recommendation-title {
        color: var(--nf-text);

        font-size: 0.86rem;
        font-weight: 800;

        margin-bottom: 12px;
    }


    .rec {
        color: var(--nf-muted);

        padding: 7px 0;

        font-size: 0.77rem;
        line-height: 1.55;
    }


    /* =====================================================
       FOOTER
    ===================================================== */

    .footer {
        text-align: center;

        color: #71849b;

        font-size: 0.72rem;

        margin-top: 44px;

        line-height: 1.7;

        padding-top: 20px;

        border-top:
            1px solid rgba(148, 163, 184, 0.07);
    }


    /* =====================================================
       DIVIDER
    ===================================================== */

    hr {
        border: none !important;

        height: 1px !important;

        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(148, 163, 184, 0.15),
                transparent
            ) !important;

        margin: 28px 0 !important;
    }


    /* =====================================================
       CHECKBOX
    ===================================================== */

    div[data-testid="stCheckbox"] label {
        transition: color 0.2s ease;
    }


    div[data-testid="stCheckbox"] label:hover {
        color: var(--nf-cyan-light);
    }


    /* =====================================================
       CAPTION / HELP TEXT
    ===================================================== */

    .stCaption {
        color: var(--nf-muted-2) !important;
    }


    /* =====================================================
       RESPONSIVE
    ===================================================== */

    @media (max-width: 1000px) {

        .hero-title {
            font-size: clamp(1.9rem, 5vw, 2.8rem);
        }

        .modality-grid {
            grid-template-columns:
                repeat(2, minmax(0, 1fr));
        }
    }


    @media (max-width: 768px) {

        .block-container {
            padding-top: 1.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }


        .navbar {
            min-height: 54px;
            margin-bottom: 22px;
            padding: 4px 0;
        }


        .nav-logo {
            width: 40px;
            height: 40px;

            min-width: 40px;
            min-height: 40px;

            font-size: 1.2rem;
        }


        .nav-name {
            font-size: 0.94rem;
        }


        .nav-subtitle {
            font-size: 0.62rem;
        }


        .hero {
            padding: 30px 24px;

            border-radius: 22px;
        }


        .hero-title {
            font-size: clamp(1.7rem, 7vw, 2.45rem);

            letter-spacing: -0.045em;

            white-space: normal;

            overflow-wrap: anywhere;
        }


        .hero-description {
            font-size: 0.92rem;
        }


        .auth-shell {
            margin-top: 2.5rem;
        }


        .auth-shell .hero {
            padding: 32px 22px;
        }


        .auth-title {
            font-size: clamp(1.45rem, 7vw, 2.2rem);

            white-space: nowrap;
        }


        .modality-grid {
            grid-template-columns: 1fr;
        }


        .selection-summary {
            align-items: flex-start;

            flex-direction: column;

            gap: 12px;
        }


        .input-card-header {
            align-items: flex-start;

            flex-direction: column;

            gap: 12px;
        }


        .input-status {
            align-self: flex-start;
        }


        .result-card {
            padding: 24px 20px;

            border-radius: 20px;
        }


        div[role="radiogroup"] {
            flex-direction: column !important;
        }


        div[role="radiogroup"] > label {
            width: 100%;
            box-sizing: border-box;
        }
    }


    @media (max-width: 520px) {

        .nav-brand {
            gap: 9px;
        }


        .nav-name {
            font-size: 0.88rem;
        }


        .nav-subtitle {
            font-size: 0.58rem;
        }


        .hero {
            padding: 26px 19px;
        }


        .hero-kicker {
            font-size: 0.64rem;

            letter-spacing: 0.11em;
        }


        .hero-title {
            font-size: 1.7rem;

            letter-spacing: -0.04em;
        }


        .hero-description {
            font-size: 0.86rem;

            line-height: 1.6;
        }


        .auth-title {
            font-size: 1.42rem;
        }


        .section-card {
            padding: 19px;
        }


        .modality-card {
            padding: 20px;
        }
    }


    /* =====================================================
       REDUCE MOTION
    ===================================================== */

    @media (prefers-reduced-motion: reduce) {

        *,
        *::before,
        *::after {
            animation-duration:
                0.01ms !important;

            animation-iteration-count:
                1 !important;

            transition-duration:
                0.01ms !important;

            scroll-behavior:
                auto !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HTML HELPER
# ============================================================

def render_html(content: str) -> None:
    """
    Render HTML using Streamlit's native HTML renderer.
    """
    st.html(textwrap.dedent(content))


# ============================================================
# AUTHENTICATION HELPERS
# ============================================================

def _secret(name: str) -> str | None:
    value = os.getenv(name)

    if value:
        return value

    try:
        return st.secrets.get(name)
    except Exception:
        return None


def supabase_client():
    if create_client is None:
        return None

    url = _secret("SUPABASE_URL")
    key = _secret("SUPABASE_ANON_KEY")

    if not url or not key:
        return None

    return create_client(url, key)


def validate_email(email: str) -> bool:
    return bool(
        re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            email.strip(),
        )
    )


def validate_password(password: str) -> str | None:
    if len(password) < 8:
        return "Password must contain at least 8 characters."

    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."

    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter."

    if not re.search(r"\d", password):
        return "Password must contain at least one number."

    return None


def is_logged_in() -> bool:
    return bool(st.session_state.get("user"))


def logout_user() -> None:
    client = supabase_client()

    if client is not None:
        try:
            client.auth.sign_out()
        except Exception:
            pass

    st.session_state.clear()
    st.rerun()


# ============================================================
# AUTH PAGE
# ============================================================

def auth_page() -> None:
    render_html(
        """
        <div class="auth-shell">

            <div class="hero">

                <div class="hero-kicker">
                    AI-assisted research screening
                </div>

                <div class="auth-title">
                    🧠 NeuroFusion-PD
                </div>

                <div class="hero-description">
                    A multimodal neurological screening research
                    platform combining voice, handwriting and
                    movement signals.
                </div>

            </div>

        </div>
        """
    )

    client = supabase_client()

    if client is None:
        st.error(
            "Authentication is not configured. "
            "Please check SUPABASE_URL and SUPABASE_ANON_KEY."
        )
        return

    login_tab, register_tab = st.tabs(
        ["Log in", "Create account"]
    )

    # ========================================================
    # LOGIN
    # ========================================================

    with login_tab:

        with st.form("login_form"):

            email = st.text_input(
                "Email",
                placeholder="you@example.com",
            )

            password = st.text_input(
                "Password",
                type="password",
            )

            submitted = st.form_submit_button(
                "Log in",
                use_container_width=True,
                type="primary",
            )

        if submitted:

            if not validate_email(email) or not password:
                st.error(
                    "Please enter a valid email and password."
                )

            else:

                try:
                    response = client.auth.sign_in_with_password(
                        {
                            "email": email.strip(),
                            "password": password,
                        }
                    )

                    if response.user:

                        st.session_state.user = {
                            "id": response.user.id,
                            "email": response.user.email or email.strip(),
                        }

                        st.success("Login successful.")
                        st.rerun()

                    else:
                        st.error(
                            "Login could not be completed. "
                            "Please check your credentials."
                        )

                except Exception:
                    st.error(
                        "Login failed. Please check your email "
                        "and password."
                    )

    # ========================================================
    # REGISTER
    # ========================================================

    with register_tab:

        with st.form("register_form"):

            email = st.text_input(
                "Email",
                placeholder="you@example.com",
                key="register_email",
            )

            password = st.text_input(
                "Password",
                type="password",
                key="register_password",
            )

            confirm = st.text_input(
                "Confirm password",
                type="password",
                key="register_confirm",
            )

            submitted = st.form_submit_button(
                "Create account",
                use_container_width=True,
                type="primary",
            )

        if submitted:

            if not validate_email(email):
                st.error(
                    "Please enter a valid email address."
                )

            elif password != confirm:
                st.error(
                    "Passwords do not match."
                )

            else:

                password_error = validate_password(password)

                if password_error:
                    st.error(password_error)

                else:

                    try:
                        response = client.auth.sign_up(
                            {
                                "email": email.strip(),
                                "password": password,
                            }
                        )

                        if response.user and response.session:

                            st.session_state.user = {
                                "id": response.user.id,
                                "email": response.user.email or email.strip(),
                            }

                            st.success(
                                "Registration successful."
                            )

                            st.rerun()

                        else:

                            st.success(
                                "Registration successful. "
                                "Please check your email to confirm "
                                "your account, then log in."
                            )

                    except Exception:
                        st.error(
                            "Registration failed. "
                            "The email may already be registered "
                            "or the request could not be completed."
                        )

    render_html(
        """
        <div class="auth-note">
            Your password is handled by Supabase Auth.
            NeuroFusion-PD does not store your password.
        </div>
        """
    )


# ============================================================
# FILE HELPERS
# ============================================================

def save_uploaded(uploaded, suffix: str) -> str:
    data = uploaded.getvalue()

    if not data:
        raise ValueError(
            "The uploaded file is empty."
        )

    fd, path = tempfile.mkstemp(
        prefix="neurofusion_",
        suffix=suffix,
    )

    os.close(fd)

    Path(path).write_bytes(data)

    return path


# ============================================================
# VOICE VALIDATION
# ============================================================

def validate_voice_input(uploaded) -> tuple[bool, str]:

    if uploaded is None:
        return False, "Please provide a voice recording."

    filename = getattr(uploaded, "name", "") or ""
    extension = Path(filename).suffix.lower()

    allowed_extensions = {
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".ogg",
    }

    if extension not in allowed_extensions:
        return (
            False,
            "This file type is not allowed. "
            "Please upload an audio file: WAV, MP3, M4A, FLAC or OGG.",
        )

    try:
        data = uploaded.getvalue()
    except Exception:
        return False, "The audio file could not be read."

    if not data:
        return False, "The audio file is empty."

    # WAV
    if extension == ".wav":

        if (
            len(data) < 12
            or data[:4] != b"RIFF"
            or data[8:12] != b"WAVE"
        ):
            return (
                False,
                "Invalid WAV file. Please upload a valid audio recording.",
            )

    # MP3
    elif extension == ".mp3":

        valid_mp3 = (
            data[:3] == b"ID3"
            or (
                len(data) >= 2
                and data[0] == 0xFF
                and (data[1] & 0xE0) == 0xE0
            )
        )

        if not valid_mp3:
            return (
                False,
                "Invalid MP3 file. Please upload a valid audio recording.",
            )

    # FLAC
    elif extension == ".flac":

        if data[:4] != b"fLaC":
            return (
                False,
                "Invalid FLAC file. Please upload a valid audio recording.",
            )

    # OGG
    elif extension == ".ogg":

        if data[:4] != b"OggS":
            return (
                False,
                "Invalid OGG file. Please upload a valid audio recording.",
            )

    # M4A / MP4
    elif extension == ".m4a":

        if len(data) < 12 or data[4:8] != b"ftyp":
            return (
                False,
                "Invalid M4A file. Please upload a valid audio recording.",
            )

    return True, "Valid audio input."


# ============================================================
# HANDWRITING VALIDATION
# ============================================================

def validate_handwriting_input(uploaded) -> tuple[bool, str]:

    if uploaded is None:
        return False, "Please provide a handwriting trajectory."

    filename = getattr(uploaded, "name", "") or ""
    extension = Path(filename).suffix.lower()

    if extension != ".txt":
        return (
            False,
            "This file type is not allowed. "
            "Please upload the handwriting trajectory as a TXT file.",
        )

    try:

        raw = uploaded.getvalue()

        if not raw:
            return (
                False,
                "The handwriting TXT file is empty.",
            )

        text = raw.decode("utf-8-sig")

    except UnicodeDecodeError:

        return (
            False,
            "The handwriting file must be a valid UTF-8 TXT file.",
        )

    except Exception:

        return (
            False,
            "The handwriting file could not be read.",
        )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return (
            False,
            "The handwriting TXT file does not contain any trajectory data.",
        )

    # Expected:
    # x;y;pressure;azimuth;altitude;timestamp;pen

    for line_number, line in enumerate(
        lines[:100],
        start=1,
    ):

        parts = line.split(";")

        if len(parts) != 7:
            return (
                False,
                f"Invalid handwriting format on line {line_number}. "
                "Expected 7 semicolon-separated values: "
                "x;y;pressure;azimuth;altitude;timestamp;pen.",
            )

        for value in parts[:6]:

            try:
                float(value.strip())

            except ValueError:

                return (
                    False,
                    f"Invalid numeric value on handwriting line "
                    f"{line_number}.",
                )

    return True, "Valid handwriting trajectory."


# ============================================================
# GAIT VALIDATION
# ============================================================

def validate_gait_input(uploaded) -> tuple[bool, str]:

    if uploaded is None:
        return False, "Please provide a gait sensor ZIP file."

    filename = getattr(uploaded, "name", "") or ""
    extension = Path(filename).suffix.lower()

    if extension != ".zip":
        return (
            False,
            "This file type is not allowed. "
            "Please upload the Sensor Logger ZIP file.",
        )

    try:

        data = uploaded.getvalue()

        if not data:
            return (
                False,
                "The gait ZIP file is empty.",
            )

        zip_file = zipfile.ZipFile(
            io.BytesIO(data)
        )

    except zipfile.BadZipFile:

        return (
            False,
            "Invalid ZIP file. "
            "Please export the gait data again from Sensor Logger.",
        )

    except Exception:

        return (
            False,
            "The gait ZIP file could not be read.",
        )

    required_files = {
        "Accelerometer.csv",
        "Gyroscope.csv",
        "Orientation.csv",
    }

    try:

        root_files = set()

        for name in zip_file.namelist():

            normalized = name.replace(
                "\\",
                "/",
            ).strip("/")

            if "/" not in normalized:
                root_files.add(normalized)

        missing_files = (
            required_files - root_files
        )

        if missing_files:

            missing_text = ", ".join(
                sorted(missing_files)
            )

            return (
                False,
                "The gait ZIP is missing required file(s): "
                f"{missing_text}.",
            )

        expected_columns = {
            "Accelerometer.csv": {
                "seconds_elapsed",
                "x",
                "y",
                "z",
            },
            "Gyroscope.csv": {
                "seconds_elapsed",
                "x",
                "y",
                "z",
            },
            "Orientation.csv": {
                "seconds_elapsed",
                "roll",
                "pitch",
                "yaw",
            },
        }

        for filename, required_columns in expected_columns.items():

            try:

                with zip_file.open(filename) as file:
                    dataframe = pd.read_csv(file)

            except Exception:

                return (
                    False,
                    f"Could not read {filename} inside the gait ZIP.",
                )

            actual_columns = set(
                dataframe.columns.astype(str)
            )

            missing_columns = (
                required_columns - actual_columns
            )

            if missing_columns:

                missing_text = ", ".join(
                    sorted(missing_columns)
                )

                return (
                    False,
                    f"{filename} is missing required column(s): "
                    f"{missing_text}.",
                )

            if dataframe.empty:

                return (
                    False,
                    f"{filename} does not contain any sensor records.",
                )

    finally:

        try:
            zip_file.close()
        except Exception:
            pass

    return True, "Valid Sensor Logger gait ZIP."


# ============================================================
# GAIT INSTRUCTIONS
# ============================================================

def render_gait_help() -> None:

    render_html(
        """
        <div class="gait-help">

            <div class="gait-help-title">
                📱 Collect movement data with Sensor Logger
            </div>

            <p>
                The current trained NeuroFusion-PD gait pipeline
                expects smartphone sensor data exported from
                Sensor Logger.
            </p>

            <ol>

                <li>
                    Install <b>Sensor Logger</b> on an Android phone.
                </li>

                <li>
                    Enable these sensors:
                    <b>Accelerometer</b>,
                    <b>Gyroscope</b> and
                    <b>Orientation</b>.
                </li>

                <li>
                    Start the recording and complete the walking
                    activity required by your study protocol.
                </li>

                <li>
                    Stop the recording and export the sensor data.
                </li>

                <li>
                    Create a ZIP containing these files
                    at the ZIP root:
                    <br><br>

                    <code>Accelerometer.csv</code><br>
                    <code>Gyroscope.csv</code><br>
                    <code>Orientation.csv</code>
                </li>

                <li>
                    The files must contain the columns expected
                    by the existing trained gait pipeline.
                </li>

            </ol>

        </div>
        """
    )

    st.link_button(
        "Open Sensor Logger on Google Play",
        "https://play.google.com/store/apps/details?id=com.kelvin.sensorapp",
    )

    st.caption(
        "The trained gait pipeline expects the ZIP structure above. "
        "Direct CSV uploads are not used."
    )


# ============================================================
# VALIDATION UI
# ============================================================

def render_validation(
    modality_name: str,
    valid: bool,
    message: str,
) -> None:

    icon = "✓" if valid else "!"

    css_class = (
        "validation-card validation-valid"
        if valid
        else "validation-card validation-invalid"
    )

    title = (
        f"{modality_name} input ready"
        if valid
        else f"{modality_name} input needs attention"
    )

    safe_modality = html.escape(modality_name)
    safe_title = html.escape(title)
    safe_message = html.escape(message)

    render_html(
        f"""
        <div class="{css_class}">

            <div class="validation-icon">
                {icon}
            </div>

            <div>

                <div class="validation-title">
                    {safe_title}
                </div>

                <div class="validation-message">
                    {safe_message}
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# DASHBOARD
# ============================================================

def render_dashboard() -> None:

    # ========================================================
    # SESSION STATE
    # ========================================================

    if "selected_modalities" not in st.session_state:
        st.session_state.selected_modalities = []

    if "fusion_method" not in st.session_state:
        st.session_state.fusion_method = (
            "Maximum Probability"
        )


    # ========================================================
    # NAVIGATION
    # ========================================================

    left, right = st.columns(
        [8, 2],
        vertical_alignment="center",
    )

    with left:

        render_html(
            """
            <div class="navbar">

                <div class="nav-brand">

                    <div class="nav-logo">
                        🧠
                    </div>

                    <div>

                        <div class="nav-name">
                            NeuroFusion-PD
                        </div>

                        <div class="nav-subtitle">
                            AI-assisted neurological screening
                        </div>

                    </div>

                </div>

            </div>
            """
        )

    with right:

        with st.popover(
            "Account",
            use_container_width=True,
        ):

            user = st.session_state.get(
                "user",
                {},
            )

            user_email = html.escape(
                str(
                    user.get(
                        "email",
                        "",
                    )
                )
            )

            render_html(
                f"""
                <div class="account-card">

                    <div class="account-label">
                        Signed in as
                    </div>

                    <div class="account-email">
                        {user_email}
                    </div>

                </div>
                """
            )

            st.divider()

            if st.button(
                "Sign out",
                use_container_width=True,
                key="sign_out_button",
            ):
                logout_user()


    # ========================================================
    # HERO
    # ========================================================

    render_html(
        """
        <div class="hero">

            <div class="hero-kicker">
                Research platform
            </div>

            <div class="hero-title">
                Understand more from every signal.
            </div>

            <div class="hero-description">
                NeuroFusion-PD combines independent voice,
                handwriting and movement signals into a
                research-oriented neurological screening workflow.
            </div>

        </div>
        """
    )


    # ========================================================
    # NEW SCREENING
    # ========================================================

    render_html(
        """
        <div class="section-card">

            <div class="section-title">
                New screening
            </div>

            <div class="section-subtitle">
                Select the signals available for this assessment.
                You can use one signal or combine multiple signals.
            </div>

        </div>
        """
    )


    # ========================================================
    # MODALITY SELECTION
    # ========================================================

    modality_info = [
        (
            "Voice",
            "🎙️",
            "Voice analysis",
            "Analyze acoustic characteristics from a short voice recording.",
        ),
        (
            "Handwriting",
            "✍️",
            "Handwriting analysis",
            "Analyze digital handwriting trajectory characteristics.",
        ),
        (
            "Gait",
            "🚶",
            "Movement analysis",
            "Analyze movement patterns from smartphone sensor data.",
        ),
    ]

    modality_columns = st.columns(3)

    for column, (
        name,
        icon,
        title,
        description,
    ) in zip(
        modality_columns,
        modality_info,
    ):

        selected = (
            name
            in st.session_state.selected_modalities
        )

        card_class = (
            "modality-card selected"
            if selected
            else "modality-card"
        )

        with column:

            render_html(
                f"""
                <div class="{card_class}">

                    <div class="modality-icon">
                        {icon}
                    </div>

                    <div class="modality-title">
                        {html.escape(title)}
                    </div>

                    <div class="modality-description">
                        {html.escape(description)}
                    </div>

                </div>
                """
            )

            button_label = (
                "✓ Selected"
                if selected
                else "Select"
            )

            if st.button(
                button_label,
                key=f"modality_{name.lower()}",
                use_container_width=True,
            ):

                if selected:
                    st.session_state.selected_modalities.remove(
                        name
                    )

                else:
                    st.session_state.selected_modalities.append(
                        name
                    )

                st.rerun()


    modalities = list(
        st.session_state.selected_modalities
    )


    # ========================================================
    # SELECTION SUMMARY
    # ========================================================

    if modalities:

        selected_text = " · ".join(
            html.escape(item)
            for item in modalities
        )

        plural = (
            "s"
            if len(modalities) != 1
            else ""
        )

        render_html(
            f"""
            <div class="selection-summary">

                <div>

                    <div class="selection-label">
                        Selected signals
                    </div>

                    <div class="selection-value">
                        {selected_text}
                    </div>

                </div>

                <div class="selection-count">
                    {len(modalities)} signal{plural}
                </div>

            </div>
            """
        )

    else:

        render_html(
            """
            <div class="waiting-card">

                <div class="waiting-dot"></div>

                <div>

                    <div class="waiting-title">
                        Select at least one signal
                    </div>

                    <div class="waiting-description">
                        Choose Voice, Handwriting or Movement
                        to begin the assessment.
                    </div>

                </div>

            </div>
            """
        )


    # ========================================================
    # FUSION METHOD
    # ========================================================

    if modalities:

        fusion_options = [
            "Maximum Probability",
            "Reliability Weighted",
        ]

        if len(modalities) > 1:
            fusion_options.insert(
                0,
                "Average",
            )

        if (
            st.session_state.fusion_method
            not in fusion_options
        ):
            st.session_state.fusion_method = (
                "Maximum Probability"
            )

        render_html(
            """
            <div class="section-card analysis-method-card">

                <div class="section-title">
                    Analysis method
                </div>

                <div class="section-subtitle">
                    Choose how the available screening
                    signals are combined.
                </div>

            </div>
            """
        )

        st.radio(
            "Analysis method",
            fusion_options,
            horizontal=True,
            label_visibility="collapsed",
            key="fusion_method",
        )

        fusion_method = (
            st.session_state.fusion_method
        )

    else:

        fusion_method = (
            "Maximum Probability"
        )


    # ========================================================
    # INPUTS
    # ========================================================

    uploaded = {
        "Voice": None,
        "Handwriting": None,
        "Gait": None,
    }

    validation_results = {}

    validators = {
        "Voice": validate_voice_input,
        "Handwriting": validate_handwriting_input,
        "Gait": validate_gait_input,
    }

    all_inputs_valid = True


    if modalities:

        render_html(
            """
            <div class="section-card input-section">

                <div class="section-title">
                    Assessment inputs
                </div>

                <div class="section-subtitle">
                    Provide the data available for this
                    screening assessment.
                </div>

            </div>
            """
        )


        # ====================================================
        # VOICE
        # ====================================================

        if "Voice" in modalities:

            render_html(
                """
                <div class="input-card">

                    <div class="input-card-header">

                        <div>

                            <div class="input-card-title">
                                🎙️ Voice signal
                            </div>

                            <div class="input-card-description">
                                Upload a short voice recording
                                or record directly in your browser.
                            </div>

                        </div>

                        <div class="input-status">
                            Required
                        </div>

                    </div>

                </div>
                """
            )

            upload_tab, record_tab = st.tabs(
                [
                    "Upload recording",
                    "Record now",
                ]
            )

            with upload_tab:

                voice_upload = st.file_uploader(
                    "Voice recording",
                    type=None,
                    key="voice_upload",
                    label_visibility="collapsed",
                    help=(
                        "Supported formats: WAV, MP3, "
                        "M4A, FLAC and OGG."
                    ),
                )

            with record_tab:

                voice_record = st.audio_input(
                    "Record your voice",
                    key="voice_record",
                )

            uploaded["Voice"] = (
                voice_upload
                if voice_upload is not None
                else voice_record
            )


        # ====================================================
        # HANDWRITING
        # ====================================================

        if "Handwriting" in modalities:

            render_html(
                """
                <div class="input-card">

                    <div class="input-card-header">

                        <div>

                            <div class="input-card-title">
                                ✍️ Handwriting signal
                            </div>

                            <div class="input-card-description">
                                Upload the digital handwriting
                                trajectory used by the trained model.
                            </div>

                        </div>

                        <div class="input-status">
                            Required
                        </div>

                    </div>

                </div>
                """
            )

            st.caption(
                "Expected format: "
                "x;y;pressure;azimuth;altitude;timestamp;pen"
            )

            uploaded["Handwriting"] = (
                st.file_uploader(
                    "Handwriting trajectory",
                    type=None,
                    key="handwriting_upload",
                    label_visibility="collapsed",
                    help=(
                        "Upload the existing NeuroFusion-PD "
                        "handwriting trajectory TXT format."
                    ),
                )
            )


        # ====================================================
        # GAIT
        # ====================================================

        if "Gait" in modalities:

            render_html(
                """
                <div class="input-card">

                    <div class="input-card-header">

                        <div>

                            <div class="input-card-title">
                                🚶 Movement signal
                            </div>

                            <div class="input-card-description">
                                Upload smartphone sensor data
                                collected using Sensor Logger.
                            </div>

                        </div>

                        <div class="input-status">
                            Required
                        </div>

                    </div>

                </div>
                """
            )

            render_gait_help()

            uploaded["Gait"] = (
                st.file_uploader(
                    "Movement sensor data",
                    type=None,
                    key="gait_upload",
                    label_visibility="collapsed",
                    help=(
                        "Upload the Sensor Logger ZIP "
                        "containing the required CSV files."
                    ),
                )
            )


        # ====================================================
        # IMMEDIATE VALIDATION
        # ====================================================

        for modality_name in modalities:

            item = uploaded.get(
                modality_name
            )

            if item is None:

                all_inputs_valid = False

                validation_results[
                    modality_name
                ] = (
                    False,
                    f"Please provide a valid "
                    f"{modality_name.lower()} input.",
                )

            else:

                is_valid, message = validators[
                    modality_name
                ](item)

                validation_results[
                    modality_name
                ] = (
                    is_valid,
                    message,
                )

                if not is_valid:
                    all_inputs_valid = False


        # ====================================================
        # VALIDATION UI
        # ====================================================

        for modality_name in modalities:

            is_valid, message = validation_results[
                modality_name
            ]

            render_validation(
                modality_name,
                is_valid,
                message,
            )

    else:

        all_inputs_valid = False


    # ========================================================
    # ANALYSIS READINESS
    # ========================================================

    ready = (
        bool(modalities)
        and all_inputs_valid
    )


    if ready:

        render_html(
            """
            <div class="ready-card">

                <div class="ready-dot"></div>

                <div>

                    <div class="ready-title">
                        Assessment ready
                    </div>

                    <div class="ready-description">
                        All selected inputs have passed
                        basic format validation.
                    </div>

                </div>

            </div>
            """
        )

    elif modalities:

        render_html(
            """
            <div class="waiting-card">

                <div class="waiting-dot"></div>

                <div>

                    <div class="waiting-title">
                        Complete the required inputs
                    </div>

                    <div class="waiting-description">
                        Analysis will become available after
                        every selected signal contains a valid input.
                    </div>

                </div>

            </div>
            """
        )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze_clicked = st.button(
        "Analyze screening",
        type="primary",
        use_container_width=True,
        disabled=not ready,
        key="analyze_screening",
    )


    # ========================================================
    # INFERENCE
    # ========================================================

    if analyze_clicked:

        paths: dict[str, str] = {}
        temp_paths: list[str] = []

        try:

            # =================================================
            # FINAL SERVER-SIDE VALIDATION
            # =================================================

            for modality_name in modalities:

                item = uploaded.get(
                    modality_name
                )

                if item is None:

                    raise ValueError(
                        f"Please provide a valid "
                        f"{modality_name.lower()} input."
                    )

                is_valid, message = validators[
                    modality_name
                ](item)

                if not is_valid:
                    raise ValueError(message)


            # =================================================
            # SAVE TEMPORARY INPUTS
            # =================================================

            with st.status(
                "Preparing screening assessment...",
                expanded=True,
            ) as status:

                for modality_name in modalities:

                    item = uploaded[
                        modality_name
                    ]

                    if modality_name == "Gait":

                        suffix = ".zip"

                    else:

                        suffix = (
                            Path(
                                getattr(
                                    item,
                                    "name",
                                    "input.bin",
                                )
                            ).suffix.lower()
                            or ".bin"
                        )

                    path = save_uploaded(
                        item,
                        suffix,
                    )

                    temp_paths.append(path)

                    paths[
                        modality_name
                    ] = path


                # =================================================
                # EXISTING TRAINED PIPELINE
                # =================================================

                status.update(
                    label=(
                        "Running NeuroFusion-PD screening "
                        "pipeline..."
                    )
                )

                result = analyze_patient(
                    modalities,
                    fusion_method,
                    voice_file=paths.get(
                        "Voice"
                    ),
                    handwriting_file=paths.get(
                        "Handwriting"
                    ),
                    gait_file=paths.get(
                        "Gait"
                    ),
                )

                status.update(
                    label="Screening analysis complete.",
                    state="complete",
                    expanded=False,
                )


            # =================================================
            # RESULT VALIDATION
            # =================================================

            if not isinstance(result, dict):

                raise ValueError(
                    "The screening pipeline returned "
                    "an invalid result."
                )

            if "final_probability" not in result:

                raise ValueError(
                    "The screening pipeline did not return "
                    "a final probability."
                )


            # =================================================
            # RESULT
            # =================================================

            probability = float(
                result["final_probability"]
            )

            if not 0.0 <= probability <= 1.0:

                raise ValueError(
                    "The screening pipeline returned "
                    "an invalid probability."
                )

            positive = bool(
                result.get(
                    "prediction",
                    False,
                )
            )

            probability_percent = (
                probability * 100
            )

            result_class = (
                "result-positive"
                if positive
                else "result-negative"
            )

            if positive:

                result_title = (
                    "Parkinson's screening signal detected"
                )

            else:

                result_title = (
                    "No significant Parkinson's "
                    "screening signal detected"
                )


            render_html(
                f"""
                <div class="result-card {result_class}">

                    <div class="result-label">
                        Screening result
                    </div>

                    <div class="result-title">
                        {html.escape(result_title)}
                    </div>

                    <div class="result-label">
                        Final probability
                    </div>

                    <div class="probability">
                        {probability_percent:.2f}%
                    </div>

                    <div class="probability-note">
                        Model-generated screening probability
                        from the selected signals and fusion method.
                    </div>

                </div>
                """
            )


            # =================================================
            # RECOMMENDATIONS
            # =================================================

            recommendations = result.get(
                "recommendations",
                [],
            )

            if recommendations:

                render_html(
                    """
                    <div class="recommendation-card">

                        <div class="recommendation-title">
                            Recommended next steps
                        </div>
                    """
                )

                for recommendation in recommendations:

                    safe_recommendation = html.escape(
                        str(recommendation)
                    )

                    render_html(
                        f"""
                        <div class="rec">
                            • {safe_recommendation}
                        </div>
                        """
                    )

                render_html(
                    """
                    </div>
                    """
                )


            # =================================================
            # DISCLAIMER
            # =================================================

            st.info(
                "Important: NeuroFusion-PD is an "
                "AI-assisted screening and research prototype. "
                "It does not diagnose Parkinson's disease "
                "and should not be used as a replacement "
                "for professional medical evaluation."
            )

            st.toast(
                "Screening analysis completed",
                icon="🧠",
            )


        # ====================================================
        # EXPECTED INPUT ERROR
        # ====================================================

        except ValueError as exc:

            st.error(
                str(exc)
            )

            st.warning(
                "No screening prediction was produced "
                "from this invalid input."
            )


        # ====================================================
        # UNEXPECTED ERROR
        # ====================================================

        except Exception:

            st.error(
                "Something went wrong while processing "
                "the screening data."
            )

            st.caption(
                "Please verify the selected file format "
                "and try again."
            )


        # ====================================================
        # ALWAYS DELETE TEMPORARY FILES
        # ====================================================

        finally:

            for path in temp_paths:

                try:
                    os.remove(path)
                except OSError:
                    pass


    # ========================================================
    # FOOTER
    # ========================================================

    render_html(
        """
        <div class="footer">

            NeuroFusion-PD · AI-assisted neurological
            screening research prototype

            <br>

            Uploaded data is processed temporarily
            for analysis and is not intentionally stored
            by this application.

            <br>

            Not intended for diagnosis or clinical
            decision-making.

        </div>
        """
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if not is_logged_in():

    auth_page()

else:

    render_dashboard()
