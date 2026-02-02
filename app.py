import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================
# ZIP CORE
# =====================
def zip_analyzer(phi, dx=1.0):
    grad = np.gradient(phi, dx)
    grad_mag = np.sqrt(sum(g**2 for g in grad))

    # Rozlišení dimenze
    if phi.ndim == 1:
        # 1D posun
        shifted_grad = [np.roll(g, 1) for g in grad]
    else:
        # 2D posun po obou osách
        shifted_grad = [
            np.roll(g, 1, axis=0) for g in grad
        ]

    dot = sum(g * sg for g, sg in zip(grad, shifted_grad))
    shifted_mag = np.sqrt(sum(sg**2 for sg in shifted_grad))

    C_zip = np.abs(dot) / (grad_mag * shifted_mag + 1e-12)
    E = np.abs(phi)**2

    return E, grad_mag, C_zip
    
# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="ZIP Coherence Analyzer", layout="wide")
st.title("ZIP Coherence Analyzer")

# -------- SIDEBAR --------
st.sidebar.header("Vstupní data")

mode = st.sidebar.radio(
    "Zdroj dat",
    ["Demo data", "Nahrát CSV"]
)

dimension = st.sidebar.radio(
    "Dimenze dat",
    ["1D", "2D"]
)

dx = st.sidebar.slider(
    "dx (měřítko)",
    0.1, 5.0, 1.0
)

phi = None

# -------- DATA --------
if mode == "Demo data":
    if dimension == "1D":
        x = np.linspace(-10, 10, 400)
        phi = np.sin(x) * np.exp(-0.1 * x**2)
    else:
        x = np.linspace(-10, 10, 200)
        y = np.linspace(-10, 10, 200)
        X, Y = np.meshgrid(x, y)
        phi = np.sin(X) * np.cos(Y) * np.exp(-0.05 * (X**2 + Y**2))

else:
    uploaded_file = st.sidebar.file_uploader(
        "Nahraj CSV soubor",
        type=["csv", "txt"]
    )
    if uploaded_file:
        phi = np.loadtxt(uploaded_file)

# -------- ANALYZE BUTTON --------
if st.sidebar.button("Analyze ZIP"):
    if phi is None:
        st.warning("Nejprve zvol data.")
    else:
        E, I, C = zip_analyzer(phi, dx)

        # -------- OUTPUT --------
        if dimension == "1D":
            st.subheader("1D výsledky")
            st.line_chart(E)
            st.line_chart(I)
            st.line_chart(C)

        else:
            st.subheader("2D výsledky (heatmapy)")
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
