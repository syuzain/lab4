
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="Rectified Linear Unit (ReLU)", layout="wide")
st.title("Rectified Linear Unit (ReLU)")
st.write("ReLU returns the input when it is positive; otherwise, it returns zero.")
st.latex(r"f(x) = \max(0, x)")
st.latex(r"f'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}")

# Sidebar settings
with st.sidebar:
    st.header("Curve")
    x_input = st.slider("Input value x", -6.0, 6.0, 1.0, 0.1)
    st.header("Neural Network Demo")
    ds = st.selectbox("Dataset", ["Moons", "Circles"])
    n_samples = st.slider("Samples", 100, 1000, 300, 50)
    noise = st.slider("Noise", 0.00, 0.40, 0.20, 0.01)
    hidden = st.text_input("Hidden layer sizes", "16,16")
    max_iter = st.slider("Max Iterations", 100, 3000, 800, 100)

# Activation function and derivative
def activation_fn(x):
    return np.maximum(0, x)

def activation_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

# 1. Activation curve
st.subheader("1. ReLU Curve")
xgrid = np.linspace(-6, 6, 400)
output = activation_fn(x_input)
gradient = activation_derivative(x_input)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xgrid, activation_fn(xgrid), label="Activation", linewidth=2)
ax.plot(xgrid, activation_derivative(xgrid), "--", label="Derivative")
ax.scatter([x_input], [output], color="red", label="Your input", zorder=5)
ax.axhline(0, color="grey", linewidth=0.5)
ax.axvline(0, color="grey", linewidth=0.5)
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close(fig)


st.write("**Output for your input**")
a, b, c = st.columns(3)
a.metric("Input", f"{x_input:.2f}")
b.metric("Output", f"{output:.4f}")
c.metric("Derivative", f"{gradient:.4f}")
st.write("**Properties**")
st.write("- Range: [0, ∞)")
st.write("- Simple and fast to compute")
st.write("- Helps reduce vanishing gradients for positive inputs")
st.write("- May cause dying ReLU when neurons receive negative inputs")

# 2. Neural network demonstration
st.subheader("2. Neural Network using RELU")
X, y = (make_moons(n_samples=n_samples, noise=noise, random_state=42)
        if ds == "Moons" else
        make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42))

hidden_sizes = tuple(int(v.strip()) for v in hidden.split(",") if v.strip().isdigit()) or (16, 16)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(hidden_layer_sizes=hidden_sizes,
                          activation="relu",
                          learning_rate_init=0.01,
                          max_iter=max_iter,
                          random_state=42))
])
model.fit(X, y)
accuracy = model.score(X, y)

xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 250),
    np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 250)
)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 5))
ax.contourf(xx, yy, Z, cmap=ListedColormap(["#FFEEEE", "#EEFFEE"]), alpha=0.7)
ax.scatter(X[:, 0], X[:, 1], c=y, s=22, edgecolor="k",
           cmap=ListedColormap(["red", "blue"]))
ax.set_title(f"MLP with RELU (Accuracy = {accuracy:.3f})")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
st.pyplot(fig)
plt.close(fig)
