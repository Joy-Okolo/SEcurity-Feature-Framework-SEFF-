import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Step 1: Load the dataset
data = pd.read_csv(r"C:\Users\Admin\Desktop\MY STUFFS\CS 710\Project materials\vulnerability_dataset.csv", delimiter=';')

# Step 2: Clean column names and remove quotes
data.columns = data.columns.str.replace('"', '')  # Remove quotes from column names
data = data.apply(lambda col: col.str.replace('"', '') if col.dtypes == 'object' else col)  # Clean string values

# Step 3: Encode categorical columns
encoder_language = LabelEncoder()
data['language'] = encoder_language.fit_transform(data['language'])  # Encode language

encoder_vulnerability = LabelEncoder()
data['vulnerability_type'] = encoder_vulnerability.fit_transform(data['vulnerability_type'])  # Encode vulnerability type

encoder_seff = LabelEncoder()
data['seff_criteria'] = encoder_seff.fit_transform(data['seff_criteria'])  # Encode SEFF criteria

data['label'] = data['label'].map({'vulnerable': 0, 'secure': 1})  # Map label to binary values

# Step 4: Feature Extraction
def extract_sensitive_data_features(code):
    """Extract features specific to Sensitive Data Exposure."""
    return {
        'contains_password': "password" in code.lower(),
        'contains_api_key': "api_key" in code.lower(),
        'contains_token': "token" in code.lower(),
        'uses_unencrypted_http': "http://" in code.lower()
    }

sensitive_features = pd.DataFrame([extract_sensitive_data_features(code) for code in data['code_sample']])
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
tfidf_features = tfidf_vectorizer.fit_transform(data['code_sample'])

# Combine TF-IDF features with sensitive-specific features
combined_features = np.hstack([tfidf_features.toarray(), sensitive_features.values])
labels = data['label'].values

# Step 5: Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
features_balanced, labels_balanced = smote.fit_resample(combined_features, labels)

# Step 6: Train-test split for vulnerability detection
X_train, X_test, y_train, y_test = train_test_split(features_balanced, labels_balanced, test_size=0.2, random_state=42)

# Train-test split for vulnerability type classification
vuln_type_features = np.hstack([tfidf_features.toarray(), sensitive_features.values])
X_train_vt, X_test_vt, y_train_vt, y_test_vt = train_test_split(
    vuln_type_features, data['vulnerability_type'], test_size=0.2, random_state=42
)

# Train-test split for SEFF criteria classification
X_train_seff, X_test_seff, y_train_seff, y_test_seff = train_test_split(
    combined_features, data['seff_criteria'], test_size=0.2, random_state=42
)

# Step 7: Train models
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)  # For secure/vulnerable detection
rf_model.fit(X_train, y_train)

vuln_type_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)  # For vulnerability type
vuln_type_model.fit(X_train_vt, y_train_vt)

seff_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)  # For SEFF criteria
seff_model.fit(X_train_seff, y_train_seff)

# Step 8: Evaluation function
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\n=== {model_name} Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=encoder_vulnerability.classes_ if model_name == "Vulnerability Type"
        else encoder_seff.classes_ if model_name == "SEFF Criteria"
        else ["Secure", "Vulnerable"]
    ))

# Step 9: Evaluate models
evaluate_model(rf_model, X_test, y_test, "Secure/Vulnerable Detection")
evaluate_model(vuln_type_model, X_test_vt, y_test_vt, "Vulnerability Type")
evaluate_model(seff_model, X_test_seff, y_test_seff, "SEFF Criteria")
