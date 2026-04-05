import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import mlflow
import mlflow.sklearn


df = pd.read_csv("Social_media_impact_on_life.csv")
print(df.head())

#Feature Engineering --> Dropped Student Id because it doesn't provide any value, just an identifier
df = df.drop(columns=["Student_ID"])

X = df.drop("Overall_Impact", axis=1) 
y = df["Overall_Impact"] #Overall_Impact has 3 values --> Positive, Negative and Neutral

#Encode Categorical Features since models struggle with them
categorical_cols = ["Gender", "Academic_Level", "Country","Most_Used_Platform", "Affects_Academic_Performance"]
numerical_cols   = ["Age", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Mental_Health_Score"]

#ColumnTransformer allows for multiple transformations to be applied to different columns without doing it one at the time
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

best_auc = 0
best_model = None
best_n = None


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

mlflow.set_tracking_uri("mlruns") #store runs locally since there is a file name missmatch
mlflow.set_experiment("social_media_health_impact")
for n_estimators in [50, 100, 200]:
    with mlflow.start_run(run_name=f"rf_{n_estimators}_trees"):

        #Pipeline bundles preprocessing and the model together so transformations are applied automatically
        pipeline =  Pipeline(steps=[ 
            ("preprocessor", preprocessor),
            ("classifier",   RandomForestClassifier(n_estimators=n_estimators, random_state=42)),
        ])

        pipeline.fit(X_train, y_train)
        val_preds = pipeline.predict_proba(X_val) # shape: (n_samples, 3)
        val_auc   = roc_auc_score(y_val, val_preds, multi_class="ovr", average="weighted") #tells roc_auc how to handle the multi-class problem

        #Log it to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("val_auc", val_auc)   

        print(f"n_estimators={n_estimators} - Validation AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = pipeline
            best_n = n_estimators
            best_run_id = mlflow.active_run().info.run_id  # save the run ID to then save the test auc of the best model

    print(f"\nBest model: n_estimators={best_n} with Validation AUC={best_auc:.4f}")


test_preds = best_model.predict_proba(X_test)
test_auc   = roc_auc_score(y_test, test_preds, multi_class="ovr", average="weighted")

with mlflow.start_run(run_id=best_run_id):  # reopen the best run and save the best model's test auc
    mlflow.log_metric("test_auc", test_auc)

print(f"Test AUC of best model: {test_auc:.4f}")
 
joblib.dump(best_model, "model.pkl")
print("Model saved to model.pkl")