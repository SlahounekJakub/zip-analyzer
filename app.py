import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def zip_analyzer(phi, dx=1.0):
    grad = np.gradient(phi, dx)
    grad_mag = np.sqrt(sum(g**2 for g in grad))

    shifted_grad = [np.roll(g, 1, axis=0) for g in grad]
    dot = sum(g * sg for g, sg in zip(grad, shifted_grad))
    shifted_mag = np.sqrt(sum(sg**2 for sg in shifted_grad))

    C_zip = np.abs(dot) / (grad_mag * shifted_mag + 1e-12)
    E = np.abs(phi)**2
    return E, grad_mag, C_zip

st.set_page_config(page_title="ZIP Analyzer", layout="wide")
st.title("ZIP Coherence Analyzer")

dx = st.sidebar.slider("dx", 0.1, 5.0, 1.0)

x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)
phi = np.sin(X) * np.cos(Y) * np.exp(-0.05*(X**2 + Y**2))

E, I, C = zip_analyzer(phi, dx)

def show(data, title):
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap="inferno", origin="lower")
    ax.set_title(title)
    plt.colorbar(im)
    st.pyplot(fig)

c1, c2, c3 = st.columns(3)
with c1: show(E, "Energie")
with c2: show(I, "In-formace")
with c3: show(C, "ZIP Koherence")
