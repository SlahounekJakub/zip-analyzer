import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================
# ZIP CORE
# =====================

ZIP_SHIFTS_2D = [
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1)
]

def zip_analyzer(phi, dx=1.0):
    grad = np.gradient(phi, dx)
    grad_mag = np.sqrt(sum(g**2 for g in grad))

    if phi.ndim == 1:
        shifted = [np.roll(grad[0], 1)]
        dot = grad[0] * shifted[0]
        shifted_mag = np.abs(shifted[0])

    elif phi.ndim == 2:
        gy, gx = grad
        gy_s = np.roll(gy, 1, axis=0)
        gx_s = np.roll(gx, 1, axis=1)
        dot = gx * gx_s + gy * gy_s
        shifted_mag = np.sqrt(gx_s**2 + gy_s**2)

    else:
        raise ValueError("Pouze 1D nebo 2D.")

    C = np.abs(dot) / (grad_mag * shifted_mag + 1e-12)
    E = phi**2
    I = grad_mag

    return E, I, C


def zip_2d_isotropic(phi, dx=1.0):
    Cs = []
    for shift in ZIP_SHIFTS_2D:
        gy, gx = np.gradient(phi, dx)
        gy_s = np.roll(gy, shift[0], axis=0)
        gx_s = np.roll(gx, shift[1], axis=1)
        dot = gx * gx_s + gy * gy_s
        mag = np.sqrt(gx**2 + gy**2)
        mag_s = np.sqrt(gx_s**2 + gy_s**2)
        Cs.append(np.abs(dot) / (mag * mag_s + 1e-12))
    return np.mean(Cs, axis=0)


def detect_critical_zones(E, C, e_thr=0.6, c_thr=0.4):
    E_norm = (E - E.min()) / (E.max() - E.min() + 1e-12)
    return (E_norm > e_thr) & (C < c_thr)


def demo_time_data_1d(T=30, N=300):
    x = np.linspace(-10, 10, N)
    return np.array([
        np.sin(x + 0.2*t) * np.exp(-0.1*(x - 0.05*t)**2)
        for t in range(T)
    ])


def zip_time_coherence(phi_t, phi_t1, dt=1.0):
    dphi = (phi_t1 - phi_t) / dt
    return np.abs(dphi * np.roll(dphi, -1)) / (
        np.abs(dphi) * np.abs(np.roll(dphi, -1)) + 1e-12
    )

# =====================
# STREAMLIT UI
# =====================

st.set_page_config(page_title="ZIP Coherence Analyzer", layout="wide")
st.title("ZIP Coherence Analyzer")

st.sidebar.header("Vstupní nastavení")

mode = st.sidebar.radio("Zdroj dat", ["Demo data", "Upload CSV"])
analysis_mode = st.sidebar.radio("Režim analýzy", ["Statický ZIP", "Časový ZIP"])
dimension = st.sidebar.radio("Dimenze", ["1D", "2D"])
dx = st.sidebar.slider("dx", 0.1, 5.0, 1.0, 0.05)

phi = None

# =====================
# DATA LOAD
# =====================

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
    uploaded = st.sidebar.file_uploader("Nahraj CSV", type=["csv", "txt"])
    if uploaded:
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

        if dimension == "1D":
            E, I, C = zip_analyzer(phi, dx)
            critical = detect_critical_zones(E, C)

            fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
            ax[0].plot(E); ax[0].set_title("Energie")
            ax[1].plot(I); ax[1].set_title("In-formace")
            ax[2].plot(C, label="ZIP koherence")

            idx = np.where(critical)[0]
            if idx.size:
                ax[2].scatter(idx, C[idx], color="red", s=15, label="kritické zóny")

            ax[2].legend()
            plt.tight_layout()
            st.pyplot(fig)

        else:
            C = zip_2d_isotropic(phi, dx)
            E = phi**2
            I = np.sqrt(sum(g**2 for g in np.gradient(phi)))

            col1, col2, col3 = st.columns(3)
            for col, data, title in zip(
                [col1, col2, col3],
                [E, I, C],
                ["Energie", "In-formace", "ZIP koherence"]
            ):
                with col:
                    fig, ax = plt.subplots()
                    im = ax.imshow(data, origin="lower", cmap="inferno")
                    ax.set_title(title)
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig)

    else:
        phi_t = demo_time_data_1d()
        t = st.slider("Časový krok", 0, phi_t.shape[0]-2, 0)

        _, _, C_s = zip_analyzer(phi_t[t], dx)
        C_t = zip_time_coherence(phi_t[t], phi_t[t+1])
        C_st = C_s * C_t

        fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(C_s); ax[0].set_title("Prostorová koherence")
        ax[1].plot(C_t); ax[1].set_title("Časová koherence")
        ax[2].plot(C_st); ax[2].set_title("Prostor × čas")
        plt.tight_layout()
        st.pyplot(fig)
