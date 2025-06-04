import streamlit as st
import numpy as np

st.set_page_config(page_title="Linear Regression", layout="centered")

st.title("Linear Regression Model")

st.markdown("""
Linear regression is a technique used to model the relationship between input variables (X) and output (Y) by fitting a linear equation. 
\n**Example:** You have 100 & 150 sq ft house price but, you want to 125 sq ft house price. Linear Regression can help you with that :)

Here, how you can predict your values:
            
**Step 1**: Enter X's values - You can also name X column with your desired one.
\n**Step 2**: Enter Y values - It's length should be equal to the length of X feature's values.
\n**Step 3**: Select your desired bias value.
\n**Step 4**: Click 'Train Model' & your model will be trained.
\n**Step 5**: It will appear predict value side where you can enter your value & get your predicted answer.
""")

st.subheader("🛠️ Train the Model")

n_features = st.number_input("Enter number of Features:", min_value=1, step=1)
feature_names = []
X_data = []

for i in range(n_features):
    # Let user enter feature names, default X1, X2...
    name = st.text_input(f"Feature {i+1} name", value=f"X{i+1}", key=f"feature_name_{i}")
    feature_names.append(name)
    col_data = st.text_input(f"Values for {name} (space-separated):", key=f"X{i}")
    try:
        values = list(map(float, col_data.strip().split()))
        X_data.append(values)
    except:
        st.warning(f"Enter valid numeric values for {name}.")

y_name = st.text_input("Feature Y name", value="Y")
Y_data = st.text_input(f"Values for {y_name} (space-separated):")
bias_val = st.number_input("Bias (intercept) value:", value=1.0)

if st.button("Train Model"):
    try:
        Y = np.array(list(map(float, Y_data.strip().split())))
        n_samples = len(Y)

        if len(X_data) != n_features or any(len(col) != n_samples for col in X_data):
            st.error("Mismatch in number of samples between features and target.")
        else:
            X = np.array(X_data).T

            # Add bias if needed
            if bias_val != 0:
                bias_column = np.ones((n_samples, 1)) * bias_val
                X_bias = np.hstack((bias_column, X))
                st.session_state.bias_enabled = True
            else:
                X_bias = X
                st.session_state.bias_enabled = False

            # Train model
            beta = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ Y
            y_pred = X_bias @ beta

            # Store model & metadata in session state
            st.session_state.beta = beta
            st.session_state.feature_names = feature_names
            st.session_state.bias_val = bias_val
            st.session_state.n_features = n_features
            st.session_state.y_name = y_name
            st.session_state.Y = Y
            st.session_state.y_pred = y_pred

            st.success("✅ Model trained and ready to predict!")

            # Show coefficients
            st.write("### 📈 Coefficients:")
            if bias_val != 0:
                st.write(f"Intercept (bias): `{round(beta[0], 4)}`")
                for i, fname in enumerate(feature_names):
                    st.write(f"Slope for {fname}: `{round(beta[i+1], 4)}`")
            else:
                for i, fname in enumerate(feature_names):
                    st.write(f"Slope for {fname}: `{round(beta[i], 4)}`")

            # Calculate and show accuracy (R²)
            ss_res = np.sum((Y - y_pred) ** 2)
            ss_tot = np.sum((Y - np.mean(Y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('-inf')
            st.write(f"Accuracy (R²): `{r2 * 100:.2f}%`")

    except Exception as e:
        st.error(f"Error in training: {e}")

# 🔍 Prediction section (only show if model is trained)
if "beta" in st.session_state:
    st.markdown("---")
    st.subheader(f"🔍 Predict {st.session_state.y_name}")

    new_input = []
    for i, fname in enumerate(st.session_state.feature_names):
        val = st.number_input(f"Enter value for {fname}:", key=f"new_input_{fname}")
        new_input.append(val)

    if st.button("Predict Now"):
        x_input = np.array(new_input).reshape(1, -1)

        if st.session_state.bias_enabled:
            bias_col = np.ones((1, 1)) * st.session_state.bias_val
            x_input = np.hstack((bias_col, x_input))

        y_output = float(np.dot(x_input, st.session_state.beta))
        st.success(f"🎯 Predicted value for {st.session_state.y_name}: `{round(y_output, 4)}`")
