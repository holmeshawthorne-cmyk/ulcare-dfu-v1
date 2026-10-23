"""Streamlit host for the original ULCARE interface.

The complete interactive layout lives in ulcare_interface.html so the browser
keeps the same navigation, tabs, modals, language menu, upload flow, and
print/email controls as the approved HTML design.
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="ULCARE — AI Wet Gangrene Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      /* Remove Streamlit chrome so the original ULCARE canvas owns the page. */
      #MainMenu, header, footer, [data-testid="stToolbar"],
      [data-testid="stStatusWidget"] { display: none !important; }
      [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main,
      [data-testid="stAppViewContainer"] .stMain { background: #f8fafc !important; }
      .block-container { max-width: none !important; padding: 0 !important; }
      [data-testid="stVerticalBlock"] { gap: 0 !important; }
      iframe { border: 0 !important; display: block !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

interface_path = Path(__file__).with_name("ulcare_interface.html")
if not interface_path.exists():
    st.error("File antarmuka ULCARE tidak ditemukan. Pastikan ulcare_interface.html ikut diunggah ke repository.")
else:
    # The interface owns all visual and JavaScript interaction inside this frame.
    components.html(interface_path.read_text(encoding="utf-8"), height=3100, scrolling=True)
