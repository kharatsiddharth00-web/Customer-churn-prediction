from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# -------------------- Load Dataset --------------------
data = pd.read_csv("churn.csv")

# Drop unnecessary column
if "customerID" in data.columns:
    data.drop("customerID", axis=1, inplace=True)

# Clean data
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

if "TotalCharges" in data.columns:
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

data.fillna(0, inplace=True)

# Convert categorical → numeric
data = pd.get_dummies(data, drop_first=True)

# Split data
X = data.drop("Churn_Yes", axis=1)
y = data["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------- Routes --------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        input_data = request.form.to_dict()

        input_df = pd.DataFrame([input_data])

        # Same preprocessing
        input_df = input_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        if "TotalCharges" in input_df.columns:
            input_df["TotalCharges"] = pd.to_numeric(input_df["TotalCharges"], errors="coerce")

        input_df.fillna(0, inplace=True)

        # Convert categorical
        input_df = pd.get_dummies(input_df)

        # Match training columns
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]

        result = "Customer Will Churn ❌" if prediction == 1 else "Customer Will Stay ✅"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(debug=True)