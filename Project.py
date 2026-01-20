# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 15:47:29 2025

@author: zk434
"""

import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

st.title("📊 HistoFit")


st.sidebar.header("📥 Data input")

user_input = st.sidebar.text_area(
    "Type your numbers (separated by commas or spaces):",
    placeholder="Example: 1.2, 3.5, 4.0, 4.5, 6.1",
    height=120
)

uploaded_file = st.sidebar.file_uploader(
    "Or upload a CSV file",
    type="csv"
)

data = None

if user_input.strip():
    try:
        data = np.array([float(x) for x in user_input.replace(",", " ").split()])
        st.success("✅ Data loaded from text input")
    except ValueError:
        st.error("Please enter only numbers, separated by commas or spaces.")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded file:")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=["number"]).columns

        if len(numeric_cols) == 0:
            st.error("No numeric columns found in this CSV.")
        else:
            col_name = numeric_cols[0]
            data = df[col_name].dropna().values
            st.success(f"✅ Data loaded from CSV column: **{col_name}**")
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")

DISTRIBUTIONS = {
    "Normal (norm)": stats.norm,
    "Exponential (expon)": stats.expon,
    "Gamma (gamma)": stats.gamma,
    "Weibull (weibull_min)": stats.weibull_min,
    "Lognormal (lognorm)": stats.lognorm,
    "Beta (beta)": stats.beta,
    "Uniform (uniform)": stats.uniform,
    "Chi-square (chi2)": stats.chi2,
    "Student t (t)": stats.t,
    "Rayleigh (rayleigh)": stats.rayleigh,
}

def fit_distribution(data):
    st.subheader("🤖 Automatic distribution fit")

    dist_name = st.selectbox(
        "Select a distribution:",
        list(DISTRIBUTIONS.keys()),
        key="auto_dist"
    )
    dist = DISTRIBUTIONS[dist_name]

    params = dist.fit(data)

    st.write("### Fitted parameters")
    st.write(params)

    x = np.linspace(np.min(data), np.max(data), 500)
    pdf = dist.pdf(x, *params)

    return dist_name, dist, x, pdf, params

if data is not None and len(data) > 0:
    tab_auto, tab_manual = st.tabs(["🤖 Automatic Fit", "🎛️ Manual Fit"])

    with tab_auto:
        dist_name, dist_obj, x, pdf, params = fit_distribution(data)

        fig, ax = plt.subplots()
        ax.hist(data, bins='auto', density=True, alpha=0.6, edgecolor="black", label="Data (histogram)")
        ax.plot(x, pdf, linewidth=2, label=f"{dist_name} PDF")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()

        st.pyplot(fig)

        hist_vals, bin_edges = np.histogram(data, bins='auto', density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        pdf_at_bins = dist_obj.pdf(bin_centers, *params)

        errors = hist_vals - pdf_at_bins
        mse = np.mean(errors**2)
        max_err = np.max(np.abs(errors))

        st.subheader("Fit quality (automatic)")
        c1, c2 = st.columns(2)
        c1.metric("MSE", f"{mse:.4f}")
        c2.metric("Max error", f"{max_err:.4f}")

    with tab_manual:
        st.subheader("🎛️ Manual fitting (adjust parameters)")

        manual_dist_name = st.selectbox(
            "Select a distribution for manual fitting:",
            list(DISTRIBUTIONS.keys()),
            key="manual_dist"
        )
        manual_dist = DISTRIBUTIONS[manual_dist_name]

        fitted_params = manual_dist.fit(data)

        shape_names = []
        if manual_dist.shapes is not None:
            shape_names = [s.strip() for s in manual_dist.shapes.split(",")]
        num_shapes = len(shape_names)

        if num_shapes > 0:
            shape_params0 = list(fitted_params[:num_shapes])
            loc0 = fitted_params[num_shapes]
            scale0 = fitted_params[num_shapes + 1]
        else:
            shape_params0 = []
            loc0 = fitted_params[0]
            scale0 = fitted_params[1]

        data_min = float(np.min(data))
        data_max = float(np.max(data))
        data_range = max(data_max - data_min, 1.0)

        shape_values = []
        for i, sname in enumerate(shape_names):
            shape_val = st.slider(
                f"Shape parameter {i+1} ({sname})",
                0.01, 10.0,
                float(shape_params0[i])
            )
            shape_values.append(shape_val)

        loc = st.slider(
            "loc (horizontal shift)",
            data_min - 0.5 * data_range,
            data_max + 0.5 * data_range,
            float(loc0)
        )

        scale = st.slider(
            "scale (spread)",
            0.001,
            2.0 * data_range,
            float(scale0)
        )

        manual_params = tuple(shape_values + [loc, scale])

        x_manual = np.linspace(data_min, data_max, 500)
        pdf_manual = manual_dist.pdf(x_manual, *manual_params)

        fig2, ax2 = plt.subplots()
        ax2.hist(data, bins='auto', density=True, alpha=0.6, edgecolor="black", label="Data (histogram)")
        ax2.plot(x_manual, pdf_manual, linewidth=2, label=f"{manual_dist_name} (manual)")
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Density")
        ax2.legend()

        st.pyplot(fig2)

        hist_vals_m, bin_edges_m = np.histogram(data, bins='auto', density=True)
        bin_centers_m = 0.5 * (bin_edges_m[1:] + bin_edges_m[:-1])
        pdf_at_bins_m = manual_dist.pdf(bin_centers_m, *manual_params)
        errors_m = hist_vals_m - pdf_at_bins_m
        mse_m = np.mean(errors_m**2)
        max_err_m = np.max(np.abs(errors_m))

        st.subheader("Fit quality (manual)")
        c3, c4 = st.columns(2)
        c3.metric("MSE", f"{mse_m:.4f}")
        c4.metric("Max error", f"{max_err_m:.4f}")

else:
    st.info("Please enter data or upload a CSV file to continue.")
