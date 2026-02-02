import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================
# ZIP ANALYZER
# =====================
def zip_analyzer(phi, dx=1.0):
    grad = np.gradient(phi, dx)
    grad_mag = np.sqrt(sum(g**2 for g in grad))

    shifted_grad = []

    if phi.ndim == 1:
        shifted_grad.append(np.roll(grad[0], 1))
    elif phi.ndim == 2:
        shifted_grad.append(np.roll(grad[0], 1, axis=0))
        shifted_grad.append(np.roll(grad[1], 1, axis=1))
    else:
        raise ValueError("Pouze 1D a 2D data jsou podporována.")

    dot = sum(g * sg for g, sg in zip(grad, shifted_grad))
    shifted_mag = np.sqrt(sum(sg**2 for sg in shifted_grad))

    C_zip = np.abs(dot) / (grad_mag * shifted_mag + 1e-12)
    E = np.abs(phi)**2

    return E, grad_mag, C_zip

def demo_time_data_1d(T=30, N=300):
    x = np.linspace(-10, 10, N)
    data = []
    for t in range(T):
        phi = np.sin(x + 0.2*t) * np.exp(-0.1*(x - 0.05*t)**2)
        data.append(phi)
    return np.array(data)
    
def zip_time_coherence(phi_t, phi_t1, dt=1.0):
    """
    Časová ZIP koherence mezi dvěma po sobě jdoucími časovými řezy
    """
    dphi = (phi_t1 - phi_t) / dt

    # ochrana proti nulám
    denom = np.abs(dphi) * np.abs(np.roll(dphi, -1)) + 1e-12
    C_time = np.abs(dphi * np.roll(dphi, -1)) / denom

    return C_time


    
# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="ZIP Coherence Analyzer", layout="wide")
st.title("ZIP Coherence Analyzer")

# ---- sidebar ----
st.sidebar.header("Vstupní nastavení")

mode = st.sidebar.radio("Zdroj dat:", ["Demo data", "Upload CSV"])
dimension = st.sidebar.radio("Dimenze dat:", ["1D", "2D"])
analysis_mode = st.sidebar.radio(
    "Režim analýzy:",
    ["Statický ZIP", "Časový ZIP"]
)
dx = st.sidebar.slider("dx (měřítko):", 0.1, 5.0, 1.0)

phi = None

# ---- DATA ----
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
    uploaded_file = st.sidebar.file_uploader("Nahraj CSV soubor", type=["csv", "txt"])
    if uploaded_file is not None:
        try:
            data = np.loadtxt(uploaded_file)
            if dimension == "2D" and data.ndim == 1:
                st.error("CSV není 2D matice.")
            else:
                phi = data
        except Exception as e:
            st.error(f"Chyba při načítání CSV: {e}")

# ---- ANALYZE ----
if st.sidebar.button("Analyze ZIP"):
    if phi is None:
        st.warning("Nejsou dostupná data.")
    else:
        try:
            E, I, C = zip_analyzer(phi, dx)

            if analysis_mode == "Statický ZIP":
    E, I, C = zip_analyzer(phi, dx)

    if dimension == "1D":
        st.subheader("Výsledky 1D")

        fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        ax[0].plot(E)
        ax[0].set_title("Energie")

        ax[1].plot(I)
        ax[1].set_title("In-formace")

        ax[2].plot(C)
        ax[2].set_title("ZIP koherence")

        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.subheader("Výsledky 2D")

        col1, col2, col3 = st.columns(3)

        def show(data, title):
            fig, ax = plt.subplots()
            im = ax.imshow(data, origin="lower", cmap="inferno")
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)

        with col1:
            show(E, "Energie")
        with col2:
            show(I, "In-formace")
        with col3:
            show(C, "ZIP koherence")

elif analysis_mode == "Časový ZIP":
    st.subheader("ZIP – časová analýza (demo)")

    phi_time = demo_time_data_1d()
    T = phi_time.shape[0]

    t_index = st.slider("Časový krok", 0, T - 2, 0)

    E, I, C_space = zip_analyzer(phi_time[t_index], dx)
    C_time = zip_time_coherence(
        phi_time[t_index],
        phi_time[t_index + 1]
    )

    C_st = C_space * C_time

    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    ax[0].plot(C_space)
    ax[0].set_title("ZIP – prostorová koherence")

    ax[1].plot(C_time)
    ax[1].set_title("ZIP – časová koherence")

    ax[2].plot(C_st)
    ax[2].set_title("ZIP – prostor × čas")

    plt.tight_layout()
    st.pyplot(fig)

                with col1:
                    show(E, "Energie")
                with col2:
                    show(I, "In-formace")
                with col3:
                    show(C, "ZIP koherence")

        except Exception as e:
            st.error(f"ZIP analýza selhala: {e}")
