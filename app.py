import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================
# ZIP CORE
# =====================

def zip_analyzer(phi, dx=1.0):
    phi = np.asarray(phi)

    if phi.ndim == 1:
        grad = np.gradient(phi, dx)
        grad_mag = np.abs(grad)
        grad_shift = np.roll(grad, 1)
        C = np.abs(grad * grad_shift) / (grad_mag * np.abs(grad_shift) + 1e-12)
        E = phi**2
        I = grad_mag
        return E, I, C

    elif phi.ndim == 2:
        gy, gx = np.gradient(phi, dx)
        grad_mag = np.sqrt(gx**2 + gy**2)

        gx_s = np.roll(gx, 1, axis=1)
        gy_s = np.roll(gy, 1, axis=0)
        grad_mag_s = np.sqrt(gx_s**2 + gy_s**2)

        dot = gx * gx_s + gy * gy_s
        C = np.abs(dot) / (grad_mag * grad_mag_s + 1e-12)
        E = phi**2
        I = grad_mag
        return E, I, C

    else:
        raise ValueError("Podporována jsou pouze 1D a 2D data.")

def detect_critical_zones(E, C, e_thr=0.6, c_thr=0.4):
    E = np.asarray(E)
    C = np.asarray(C)
    if E.shape != C.shape:
        return np.zeros_like(E, dtype=bool)
    E_norm = (E - E.min()) / (E.max() - E.min() + 1e-12)
    return (E_norm > e_thr) & (C < c_thr)

def zip_health_index(C, threshold=0.5):
    C = np.asarray(C)
    valid = np.isfinite(C)
    if not np.any(valid):
        return 0.0
    return np.mean(C[valid] >= threshold)

def demo_time_data_1d(T=30, N=300):
    x = np.linspace(-10, 10, N)
    return np.array([
        np.sin(x + 0.2*t) * np.exp(-0.1*(x - 0.05*t)**2)
        for t in range(T)
    ])

def zip_time_coherence(phi_t, phi_t1):
    dphi = phi_t1 - phi_t
    return np.abs(dphi * np.roll(dphi, 1)) / (np.abs(dphi) * np.abs(np.roll(dphi, 1)) + 1e-12)

# =====================
# STREAMLIT UI
# =====================

st.set_page_config(page_title="ZIP Coherence Analyzer", layout="wide")
st.title("ZIP Coherence Analyzer")

st.sidebar.header("Vstupní nastavení")

data_mode = st.sidebar.radio("Zdroj dat", ["Demo data", "Upload CSV"])
analysis_mode = st.sidebar.radio("Režim analýzy", ["Statický ZIP", "Časový ZIP"])
dimension = st.sidebar.radio("Dimenze", ["1D", "Remember: 2D"])
dx = st.sidebar.slider("dx (měřítko)", 0.1, 5.0, 1.0, 0.05)

phi = None

# =====================
# DATA LOAD
# =====================

if data_mode == "Demo data":
    if dimension == "1D":
        x = np.linspace(-10, 10, 300)
        phi = np.sin(x) * np.exp(-0.1 * x**2)
    else:
        x = np.linspace(-10, 10, 200)
        y = np.linspace(-10, 10, 200)
        X, Y = np.meshgrid(x, y)
        phi = np.sin(X) * np.cos(Y) * np.exp(-0.05 * (X**2 + Y**2))
else:
    uploaded = st.sidebar.file_uploader("Nahraj CSV", type=["csv", "txt"])
    if uploaded is not None:
        phi = np.loadtxt(uploaded)

# =====================
# ANALYZE
# =====================

if st.sidebar.button("Analyze ZIP"):

    if phi is None:
        st.warning("Nejsou dostupná data.")
        st.stop()

    st.write("DATA SHAPE:", phi.shape)

    if analysis_mode == "Statický ZIP":

        E, I, C = zip_analyzer(phi, dx)

        if dimension == "1D":
            critical = detect_critical_zones(E, C)

            fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
            ax[0].plot(E); ax[0].set_title("Energie")
            ax[1].plot(I); ax[1].set_title("In-formace")

            idx = np.where(critical)[0]
            ax[2].plot(C, label="ZIP koherence")
            if idx.size:
                ax[2].scatter(idx, C[idx], color="red", s=15, label="kritické zóny")

            ax[2].legend()
            st.pyplot(fig)

        else:
            col1, col2, col3 = st.columns(3)

            def show(data, title):
                fig, ax = plt.subplots()
                im = ax.imshow(data, origin="lower", cmap="inferno")
                ax.set_title(title)
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)

            with col1: show(E, "Energie")
            with col2: show(I, "In-formace")
            with col3: show(C, "ZIP koherence")

        zhi = zip_health_index(C)
        st.subheader("ZIP Insight")
        st.metric("ZIP Health Index", f"{zhi:.2f}")

        if zhi >= 0.7:
            st.success("Systém je převážně koherentní.")
        elif zhi >= 0.4:
            st.warning("Systém je v přechodovém stavu.")
        else:
            st.error("Systém ztrácí koherenci.")

    else:
        st.subheader("ZIP – časová analýza")

        phi_t = demo_time_data_1d()
        t = st.slider("Časový krok", 0, phi_t.shape[0] - 2, 0)

        _, _, C_space = zip_analyzer(phi_t[t], dx)
        C_time = zip_time_coherence(phi_t[t], phi_t[t+1])
        C_st = C_space * C_time

        fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(C_space); ax[0].set_title("Prostorová koherence")
        ax[1].plot(C_time); ax[1].set_title("Časová koherence")
        ax[2].plot(C_st); ax[2].set_title("Prostor × čas")
        st.pyplot(fig)
