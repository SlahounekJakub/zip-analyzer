import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# ZIP CORE
# ==================================================

ZIP_SHIFTS_2D = [
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1)
]

def zip_analyzer_1d(phi, dx=1.0):
    g = np.gradient(phi, dx)
    g_shift = np.roll(g, 1)

    E = phi**2
    I = np.abs(g)
    C = np.abs(g * g_shift) / (np.abs(g) * np.abs(g_shift) + 1e-12)

    return E, I, C


def zip_analyzer_2d(phi, dx=1.0, shift=(1, 0)):
    gy, gx = np.gradient(phi, dx)

    gx_s = np.roll(gx, shift[1], axis=1)
    gy_s = np.roll(gy, shift[0], axis=0)

    dot = gx * gx_s + gy * gy_s
    mag = np.sqrt(gx**2 + gy**2)
    mag_s = np.sqrt(gx_s**2 + gy_s**2)

    C = np.abs(dot) / (mag * mag_s + 1e-12)
    E = phi**2
    I = mag

    return E, I, C


def zip_2d_isotropic(phi, dx=1.0):
    Cs = []
    for s in ZIP_SHIFTS_2D:
        _, _, C = zip_analyzer_2d(phi, dx, s)
        Cs.append(C)
    return np.mean(Cs, axis=0)


def detect_critical_zones(E, C, e_thr=0.6, c_thr=0.4):
    En = (E - E.min()) / (E.max() - E.min() + 1e-12)
    return (En > e_thr) & (C < c_thr)


def zip_time_coherence(phi_t, phi_t1):
    d = phi_t1 - phi_t
    d_s = np.roll(d, 1)
    return np.abs(d * d_s) / (np.abs(d) * np.abs(d_s) + 1e-12)


# ==================================================
# STREAMLIT UI
# ==================================================

st.set_page_config(page_title="ZIP Coherence Analyzer", layout="wide")
st.title("ZIP Coherence Analyzer")

# ---- SIDEBAR ----
st.sidebar.header("Vstupní nastavení")

mode = st.sidebar.radio(
    "Zdroj dat",
    ["Demo data", "Upload CSV"]
)

dimension = st.sidebar.radio(
    "Dimenze",
    ["1D", "2D"]
)

analysis_mode = st.sidebar.radio(
    "Režim analýzy",
    ["Statický ZIP", "Časový ZIP"]
)

dx = st.sidebar.slider("dx (měřítko)", 0.1, 5.0, 1.0, 0.05)

# ---- DATA ----
phi = None

if mode == "Demo data":
    if dimension == "1D":
        x = np.linspace(-10, 10, 300)
        phi = np.sin(x) * np.exp(-0.1 * x**2)
    else:
        x = np.linspace(-10, 10, 200)
        y = np.linspace(-10, 10, 200)
        X, Y = np.meshgrid(x, y)
        phi = np.sin(X) * np.cos(Y) * np.exp(-0.05 * (X**2 + Y**2))
else:
    f = st.sidebar.file_uploader("CSV", type=["csv", "txt"])
    if f:
        phi = np.loadtxt(f)

# ==================================================
# ANALYZE
# ==================================================

if st.sidebar.button("Analyze ZIP"):

    if phi is None:
        st.warning("Nejsou dostupná data.")
        st.stop()

    st.write("DATA SHAPE:", phi.shape)

    # -------- STATICKÝ ZIP --------
    if analysis_mode == "Statický ZIP":

        if dimension == "1D":
            E, I, C = zip_analyzer_1d(phi, dx)
            critical = detect_critical_zones(E, C)

            fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

            ax[0].plot(E); ax[0].set_title("Energie")
            ax[1].plot(I); ax[1].set_title("In-formace")
            ax[2].plot(C, label="ZIP koherence")

            idx = np.where(critical)[0]
            if idx.size:
                ax[2].scatter(idx, C[idx], c="red", s=15, label="kritická zóna")

            ax[2].legend()
            plt.tight_layout()
            st.pyplot(fig)

        else:  # 2D
            E, I, _ = zip_analyzer_2d(phi, dx)
            C = zip_2d
