import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================
# ZIP ANALYZER
# =====================
def zip_analyzer(phi, dx=1.0):
    # gradient
    grad = np.gradient(phi, dx)
    # gradient magnitudes
    grad_mag = np.sqrt(sum(g**2 for g in grad))

    # shifted gradients
    shifted_grad = []

    if phi.ndim == 1:
        # 1D posun
        shifted_grad.append(np.roll(grad[0], 1))
    elif phi.ndim == 2:
        # 2D posun každého gradientu podle vlastní osy
        # grad[0] = ∂/∂y, grad[1] = ∂/∂x
        shifted_grad.append(np.roll(grad[0], 1, axis=0))
        shifted_grad.append(np.roll(grad[1], 1, axis=1))
    else:
        raise ValueError("Pouze 1D a 2D data jsou podporována.")

    # dot product
    dot = sum(g * sg for g, sg in zip(grad, shifted_grad))
    shifted_mag = np.sqrt(sum(sg**2 for sg in shifted_grad))

    # ZIP coherence
    C_zip = np.abs(dot) / (grad_mag * shifted_mag + 1e-12)
    E = np.abs(phi)**2

    return E, grad_mag, C_zip

# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="ZIP Coherence Analyzer", layout="wide")
st.title("ZIP Coherence Analyzer")

# ---- sidebar ----
st.sidebar.header("Vstupní nastavení")

mode = st.sidebar.radio("Zdroj dat:", ["Demo data", "Upload CSV"])

dimension = st.sidebar.radio("Dimenze dat:", ["1D", "2D"])

dx = st.sidebar.slider("dx (měřítko):", 0.1, 5.0, 1.0)

phi = None

# Load data
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
    if uploaded_file:
        try:
            # load as 1D or 2D
            data = np.loadtxt(uploaded_file)
            # if dimension==2D but data is 1D → error
            if dimension == "2D" and data.ndim == 1:
                st.error("CSV není 2D matice, zkontroluj nahraná data.")
            else:
                phi = data
        except Exception as e:
            st.error("Chyba při načítání CSV: " + str(e))

# Analyze button
if st.sidebar.button("Analyze ZIP"):
    if phi is None:
        st.warning("Nejsou dostupná data. Zkus Demo nebo nahraj CSV.")
    else:
        try:
            E, I, C = zip_analyzer(phi, dx)
            
            # output
            if dimension == "1D":
                st.subheader("Výsledky 1D:")
                st.line_chart(E)
                st.line_chart(I)
                st.line_chart(C)
            else:
                st.subheader("Výsledky 2D (heatmapy):")
                col1, col2, col3 = st.columns(3)

                def show_heat(data, title):
                    fig, ax = plt.subplots()
                    im = ax.imshow(data, origin="lower", cmap="inferno")
                    ax.set_title(title)
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig)

                with col1:
                    show_heat(E, "Energie")
                with col2:
                    show_heat(I, "In-formace")
                with col3:
                    show_heat(C, "ZIP koherence")

        except Exception as e:
            st.error("ZIP analýza selhala: " + str(e))
