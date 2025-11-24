import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.constants import mu_0, epsilon_0, c, pi
from scipy import special
import pandas as pd
import streamlit as st
from PIL import Image
import io

# ---------------------------------------------------
# Utility functions
# ---------------------------------------------------

def polar_to_cartesian(Er, Etheta, theta):
    """Convert polar (Er,Etheta) -> Cartesian (Ex,Ey)."""
    Ex = Er * np.cos(theta) - Etheta * np.sin(theta)
    Ey = Er * np.sin(theta) + Etheta * np.cos(theta)
    return Ex, Ey

def streamplot_on_axis(ax, X, Y, U, V, R, density=1.2, color="blue"):
    mask = (X**2 + Y**2) <= R**2
    U = np.ma.array(U, mask=~mask)
    V = np.ma.array(V, mask=~mask)
    ax.streamplot(X, Y, U, V, color=color, density=density, linewidth=1)
    circle = plt.Circle((0,0), R, color="k", fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.axis("off")

def fig_to_array(fig):
    """Convert Matplotlib figure -> PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img

# ---------------------------------------------------
# Circular waveguide modes
# ---------------------------------------------------

def circular_mode_fields(n, m, R, mode="TE", grid_N=151):
    """Compute circular waveguide fields for arbitrary (n,m)."""
    if mode=="TM":
        root = special.jn_zeros(n, m)[-1]
    else:
        root = special.jnp_zeros(n, m)[-1]

    k_c = root / R

    x = np.linspace(-R, R, grid_N)
    y = np.linspace(-R, R, grid_N)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    angfun = np.cos(n*theta)
    d_angfun = -n*np.sin(n*theta)

    if mode=="TM":
        Ez = special.jv(n, k_c*r) * angfun
        Er = -k_c * special.jvp(n, k_c*r) * angfun
        Etheta = -(1/r) * special.jv(n, k_c*r) * d_angfun
        Ex, Ey = polar_to_cartesian(Er, Etheta, theta)

        # ✅ Fixed H-field direction for TM
        Hr = (1/r) * special.jv(n, k_c*r) * d_angfun
        Htheta = -k_c * special.jvp(n, k_c*r) * angfun
        Hx, Hy = polar_to_cartesian(Hr, Htheta, theta)
        Hz = np.zeros_like(Ez)

    else: 
        Hz = special.jv(n, k_c*r) * angfun

        # ✅ Fixed H-field direction for TE
        Hr = k_c * special.jvp(n, k_c*r) * angfun
        Htheta = (1/r) * special.jv(n, k_c*r) * d_angfun
        Hx, Hy = polar_to_cartesian(Hr, Htheta, theta)

        Er = -(1/r) * special.jv(n, k_c*r) * d_angfun
        Etheta = k_c * special.jvp(n, k_c*r) * angfun
        Ex, Ey = polar_to_cartesian(Er, Etheta, theta)
        Ez = np.zeros_like(Hz)

    return X, Y, Ex, Ey, Ez, Hx, Hy, Hz, {"n":n,"m":m,"R":R,"k_c":k_c,"root":root}

# ---------------------------------------------------
# Rectangular waveguide
# ---------------------------------------------------

class TE_TM_Functions:
    def __init__(self, m, n, a, b):
        self.m, self.n, self.a, self.b = m, n, a, b
        self.f = 2 * self.Fc()
        self.w = 2 * pi * self.f

    def Kc(self):
        return np.sqrt((self.m*pi/self.a)**2 + (self.n*pi/self.b)**2)

    def Fc(self):
        return (1/(2*np.sqrt(mu_0*epsilon_0))) * np.sqrt((self.m/self.a)**2 + (self.n/self.b)**2)

    def beta_g(self):
        fc_val = self.Fc()
        return self.w*np.sqrt(mu_0*epsilon_0)*np.sqrt(1-(fc_val/self.f)**2)

    def v_G(self):
        return self.w / self.beta_g()

    def Z_in(self):
        return np.sqrt(mu_0/epsilon_0)

    def Z_G_TE(self):
        return self.Z_in()/np.sqrt(1-(self.Fc()/self.f)**2)

    def lambda_G(self):
        return 2*pi/self.beta_g()

    def Z_G_TM(self):
        return self.Z_in()*np.sqrt(1-(self.Fc()/self.f)**2)

def handleRectangular(m, n, A, B, mode="TE"):
    x, y = np.linspace(0,A,101), np.linspace(0,B,101)
    X, Y = np.meshgrid(x, y)
    par = TE_TM_Functions(m, n, A, B)

    st.image("rectangular.png", use_container_width=True)
    
    # Center-aligned main title using native markdown
    st.markdown(f"<div style='text-align: center;'><h3>{mode}<sub>{m}{n}</sub> Rectangular Mode</h3></div>", unsafe_allow_html=True)

    if m==0 and n==0:
        st.error("m and n cannot be 0 simultaneously")
        return

    if mode=="TE":
        u, v = np.cos(m*pi/A*X)*np.sin(n*pi/B*Y), -np.sin(m*pi/A*X)*np.cos(n*pi/B*Y)
    else: 
        u, v = np.cos(m*pi/A*X)*np.sin(n*pi/B*Y), np.sin(m*pi/A*X)*np.cos(n*pi/B*Y)

    col1, col2 = st.columns(2)
    
    with col1:
        # Center-aligned title above the figure
        st.markdown(f"<div style='text-align: center;'><h4>{mode}<sub>{m}{n}</sub> E field</h4></div>", unsafe_allow_html=True)
        fig = plt.figure(figsize=(6,6))
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.streamplot(x, y, u, v, color="blue", density=1.2)
        plt.xlim(0, A)
        plt.ylim(0, B)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        # Center-aligned title above the figure
        st.markdown(f"<div style='text-align: center;'><h4>{mode}<sub>{m}{n}</sub> H field</h4></div>", unsafe_allow_html=True)
        fig = plt.figure(figsize=(6,6))
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.streamplot(x, y, -v, u, color="red", density=1.2)
        plt.xlim(0, A)
        plt.ylim(0, B)
        st.pyplot(fig)
        plt.close(fig)

    # Center the table with custom styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        html_table = f"""
        <table style="width:100%; border-collapse: collapse; margin: 20px 0;">
            <thead>
                <tr style="background-color: rgba(255,255,255,0.1);">
                    <th style="color: white; font-weight: bold; font-size: 18px; padding: 12px; border: 1px solid rgba(255,255,255,0.3); text-align: left;">Parameter</th>
                    <th style="color: white; font-weight: bold; font-size: 18px; padding: 12px; border: 1px solid rgba(255,255,255,0.3); text-align: left;">Value</th>
                    <th style="color: white; font-weight: bold; font-size: 18px; padding: 12px; border: 1px solid rgba(255,255,255,0.3); text-align: left;">Unit</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">Kc</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">{par.Kc():.4e}</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">1/m</td>
                </tr>
                <tr>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">Fc</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">{par.Fc():.4e}</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">Hz</td>
                </tr>
                <tr>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">Beta-g</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">{par.beta_g():.4e}</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">1/m</td>
                </tr>
                <tr>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">Vg</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">{par.v_G():.4e}</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">m/s</td>
                </tr>
                <tr>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">Zin</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">{par.Z_in():.4e}</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">Ohm</td>
                </tr>
                <tr>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">Zg</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">{"" if mode=='TE' else ""}{par.Z_G_TE() if mode=='TE' else par.Z_G_TM():.4e}</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">Ohm</td>
                </tr>
                <tr>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">lambda-g</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">{par.lambda_G():.4e}</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">m</td>
                </tr>
            </tbody>
        </table>
        """
        st.markdown(html_table, unsafe_allow_html=True)

# ---------------------------------------------------
# TEM mode
# ---------------------------------------------------

def handleTEM():
    r = np.linspace(1, 5, 101)
    t = np.linspace(0, 2*pi, 101)
    T, RAD = np.meshgrid(t, r)
    U, V = 10/RAD, T*0

    st.markdown("<div style='text-align: center;'><h3>TEM Mode</h3></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div style='text-align: center;'><h4>TEM E field</h4></div>", unsafe_allow_html=True)
        fig = plt.figure(figsize=(4,4))
        plt.streamplot(T, RAD, V, U, color="blue")
        st.pyplot(fig)

    with col2:
        st.markdown("<div style='text-align: center;'><h4>TEM H field</h4></div>", unsafe_allow_html=True)
        fig = plt.figure(figsize=(4,4))
        plt.streamplot(T, RAD, RAD*1e-9, T*100, color="red")
        st.pyplot(fig)

# ---------------------------------------------------
# Circular mode handler (rotated 90° clockwise)
# ---------------------------------------------------

def handleCircular(n, m, R, mode="TE", density=1.2):
    X, Y, Ex, Ey, Ez, Hx, Hy, Hz, params = circular_mode_fields(n, m, R, mode)

    st.image("circular.png", use_container_width=True)
    col1, col2 = st.columns(2)
    
    with col1:
        figE, axE = plt.subplots(figsize=(4,4))
        streamplot_on_axis(axE, X, Y, Ex, Ey, R, density=density, color="blue")

        # 🔄 Rotate image 90° clockwise
        imgE = fig_to_array(figE).rotate(-90, expand=True)
        st.markdown(f"<div style='text-align: center;'><h4>{mode}<sub>{n}{m}</sub> E-field</h4></div>", unsafe_allow_html=True)
        st.image(imgE, use_container_width=True)
        plt.close(figE)

    with col2:
        figH, axH = plt.subplots(figsize=(4,4))
        streamplot_on_axis(axH, X, Y, Hx, Hy, R, density=density, color="red")

        # 🔄 Rotate image 90° clockwise
        imgH = fig_to_array(figH).rotate(-90, expand=True)
        st.markdown(f"<div style='text-align: center;'><h4>{mode}<sub>{n}{m}</sub> H-field</h4></div>", unsafe_allow_html=True)
        st.image(imgH, use_container_width=True)
        plt.close(figH)

    # Center the table with custom styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        html_table = f"""
        <table style="width:100%; border-collapse: collapse; margin: 20px 0;">
            <thead>
                <tr style="background-color: rgba(255,255,255,0.1);">
                    <th style="color: white; font-weight: bold; font-size: 18px; padding: 12px; border: 1px solid rgba(255,255,255,0.3); text-align: left;">Parameter</th>
                    <th style="color: white; font-weight: bold; font-size: 18px; padding: 12px; border: 1px solid rgba(255,255,255,0.3); text-align: left;">Value</th>
                    <th style="color: white; font-weight: bold; font-size: 18px; padding: 12px; border: 1px solid rgba(255,255,255,0.3); text-align: left;">Unit</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">n</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">{params["n"]}</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">-</td>
                </tr>
                <tr>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">m</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">{params["m"]}</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">-</td>
                </tr>
                <tr>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">R</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">{params['R']:.2f}</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">m</td>
                </tr>
                <tr>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">k_c</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">{params['k_c']:.4e}</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">1/m</td>
                </tr>
                <tr>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">root</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">{params['root']:.4f}</td>
                    <td style="color: white; font-size: 16px; padding: 10px; border: 1px solid rgba(255,255,255,0.2);">-</td>
                </tr>
            </tbody>
        </table>
        """
        st.markdown(html_table, unsafe_allow_html=True)

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------

st.set_page_config(page_title="Waveguide Field Visualizer", layout="wide")

st.markdown("""
    <style>
    .title-wrapper {
        text-align: center;
        color: white !important;
        font-size: 48px !important;
        font-weight: bold !important;
        margin-bottom: 30px;
    }
    .title-wrapper h1 {
        color: white !important;
        font-size: 48px !important;
        font-weight: bold !important;
    }
    /* Force all dataframe styling */
    .stDataFrame, .stDataFrame > div, .stDataFrame table, .stDataFrame thead, .stDataFrame tbody, .stDataFrame tr, .stDataFrame th, .stDataFrame td {
        color: white !important;
        background-color: transparent !important;
    }
    .stDataFrame th {
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        background-color: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
    }
    .stDataFrame td {
        color: white !important;
        font-size: 16px !important;
        background-color: rgba(0,0,0,0.2) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }
    /* Override any default colors */
    [data-testid="stDataFrame"] {
        color: white !important;
    }
    [data-testid="stDataFrame"] table {
        color: white !important;
    }
    [data-testid="stDataFrame"] th {
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
    }
    [data-testid="stDataFrame"] td {
        color: white !important;
        font-size: 16px !important;
    }
    /* Additional targeting for column headers */
    div[data-testid="stDataFrame"] thead tr th {
        color: white !important;
        font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Dynamic title based on waveguide type selection
wg_type = st.sidebar.selectbox("Waveguide Type", ["TEM", "Rectangular", "Circular"])

if wg_type == "TEM":
    title_text = "Waveguide Field Visualizer"
elif wg_type == "Rectangular":
    title_text = "Waveguide Field Visualizer - Rectangular Waveguide"
else:  # Circular
    title_text = "Waveguide Field Visualizer - Circular Waveguide"

st.markdown(f"<h1 class='title-wrapper'>{title_text}</h1>", unsafe_allow_html=True)

mode = st.sidebar.radio("Mode", ["TE","TM"])

if wg_type=="TEM":
    handleTEM()

elif wg_type=="Rectangular":
    m = st.sidebar.slider("m", 0, 5, 1)
    n = st.sidebar.slider("n", 0, 5, 1)
    A = st.sidebar.slider("a (width)", 1, 20, 10)
    B = st.sidebar.slider("b (height)", 1, 20, 5)
    handleRectangular(m,n,A,B,mode=mode)

else: 
    n = st.sidebar.slider("n (azimuthal index)", 0, 5, 1)
    m = st.sidebar.slider("m (radial index)", 1, 5, 1)
    R = st.sidebar.slider("R (radius)", 1, 10, 5)
    density = st.sidebar.slider("Streamline density", 0.6, 2.5, 1.2, step=0.1)
    handleCircular(n, m, R, mode=mode, density=density)
