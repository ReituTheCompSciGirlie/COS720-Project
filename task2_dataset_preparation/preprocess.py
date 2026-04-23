import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

# =============================================================================
# STEP 1 – LOAD THE DATASET
# =============================================================================
print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

# Adjust this path if your folder structure is different
DATA_PATH = "../data/insider_threat_clean_dataset.csv"

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")
print(f"\nColumn names:")
for col in df.columns:
    print(f"  - {col}")


# =============================================================================
# STEP 2 – CHECK FOR MISSING VALUES
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Checking for missing values")
print("=" * 60)

missing = df.isnull().sum()
print(missing.to_string())
print(f"\nTotal missing values: {missing.sum()}")

if missing.sum() == 0:
    print("Result: No missing values found. No imputation needed.")
else:
    print("Result: Missing values found. Filling numeric with median, categorical with mode.")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)


# =============================================================================
# STEP 3 – CHECK CLASS BALANCE
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Checking class balance (target = is_malicious)")
print("=" * 60)

counts = df['is_malicious'].value_counts()
total  = len(df)

print(f"Normal    (0): {counts[0]:>7,}  ({counts[0]/total*100:.2f}%)")
print(f"Malicious (1): {counts[1]:>7,}  ({counts[1]/total*100:.2f}%)")
print(f"Imbalance ratio: {counts[0]/counts[1]:.1f} : 1")
print("Note: This imbalance will be handled with SMOTE in Step 6.")


# =============================================================================
# STEP 4 – ENCODE CATEGORICAL VARIABLES
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Encoding categorical variables")
print("=" * 60)

cat_cols = [
    'employee_department',
    'employee_campus',
    'employee_position',
    'employee_origin_country'
]

le = LabelEncoder()
encoding_map = {}

for col in cat_cols:
    original_values = df[col].unique().tolist()
    df[col] = le.fit_transform(df[col])
    encoded_values = sorted(df[col].unique().tolist())
    encoding_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"\n  {col}")
    print(f"    Unique categories : {len(original_values)}")
    print(f"    Encoded range     : 0 to {max(encoded_values)}")

print("\nAll categorical columns are now integers.")


# =============================================================================
# STEP 5 – DROP LOW-VALUE FEATURES
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Dropping low-value features")
print("=" * 60)

drop_cols = [
    'has_medical_history',      # correlation 0.008 — near zero; also raises privacy concerns
    'has_foreign_citizenship',  # correlation 0.021 — redundant with hostility_country_level
    'is_abroad',                # correlation 0.017 — weak; trip context already captured
    'trip_day_number',          # correlation 0.014 — weakest numerical feature
    'late_exit_flag',           # zero variance across dataset — cannot contribute to model
]

# employee_origin_country is kept (as the encoded version) — we only drop the raw string
# It has already been encoded in Step 4 above

print("Features dropped:")
for col in drop_cols:
    print(f"  - {col}")

df = df.drop(columns=drop_cols)

print(f"\nFeatures remaining (excluding target): {len(df.columns) - 1}")
print("Remaining features:")
for col in df.columns:
    if col != 'is_malicious':
        print(f"  - {col}")


# =============================================================================
# STEP 6 – SEPARATE FEATURES (X) AND TARGET (y)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Separating features (X) and target (y)")
print("=" * 60)

X = df.drop('is_malicious', axis=1)
y = df['is_malicious']

print(f"X shape (input features) : {X.shape}")
print(f"y shape (target labels)  : {y.shape}")


# =============================================================================
# STEP 7 – STRATIFIED TRAIN / TEST SPLIT (80% / 20%)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Stratified 80/20 train/test split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y         
)

print(f"\nTraining set : {len(X_train):,} rows")
print(f"  Normal    (0): {(y_train == 0).sum():,}")
print(f"  Malicious (1): {(y_train == 1).sum():,}")
print(f"\nTest set     : {len(X_test):,} rows")
print(f"  Normal    (0): {(y_test == 0).sum():,}")
print(f"  Malicious (1): {(y_test == 1).sum():,}")


# =============================================================================
# STEP 8 – APPLY SMOTE TO TRAINING SET ONLY
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: Applying SMOTE to balance the training set")
print("=" * 60)

print("Before SMOTE:")
print(f"  Normal    (0): {(y_train == 0).sum():,}")
print(f"  Malicious (1): {(y_train == 1).sum():,}")

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(f"  Normal    (0): {(y_train_sm == 0).sum():,}")
print(f"  Malicious (1): {(y_train_sm == 1).sum():,}")
print(f"  Total         : {len(X_train_sm):,}")
print("\nIMPORTANT: Test set is not modified — it stays at real-world distribution.")


# =============================================================================
# STEP 9 – FEATURE SCALING (for Logistic Regression / SVM)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 9: Feature scaling (StandardScaler)")
print("=" * 60)

scaler = StandardScaler()

# Fit ONLY on training data, then apply the same scale to test data
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)

print("StandardScaler applied.")
print("  - Fitted on training data only (prevents data leakage)")
print("  - Transformed test data using the same scale")
print("  - Random Forest: use X_train_sm  / X_test  (unscaled)")
print("  - Logistic Regression: use X_train_scaled / X_test_scaled")


# =============================================================================
# STEP 10 – SAVE PREPROCESSED DATA TO CSV
# =============================================================================
print("\n" + "=" * 60)
print("STEP 10: Saving preprocessed files to outputs/")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)

# Save the SMOTE-balanced training set
train_out = pd.DataFrame(X_train_sm, columns=X.columns)
train_out['is_malicious'] = y_train_sm.values
train_out.to_csv("outputs/preprocessed_train.csv", index=False)

# Save the unmodified test set
test_out = pd.DataFrame(X_test, columns=X.columns)
test_out['is_malicious'] = y_test.values
test_out.to_csv("outputs/preprocessed_test.csv", index=False)

print("Saved: outputs/preprocessed_train.csv")
print(f"  Rows: {len(train_out):,}  |  Balanced (50/50 after SMOTE)")
print("\nSaved: outputs/preprocessed_test.csv")
print(f"  Rows: {len(test_out):,}  |  Real-world distribution (94.6% / 5.4%)")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("TASK 2 COMPLETE — PREPROCESSING SUMMARY")
print("=" * 60)
print(f"  Original dataset rows       : 118,614")
print(f"  Features after selection    : {X.shape[1]}")
print(f"  Training rows (post-SMOTE)  : {len(X_train_sm):,}")
print(f"  Test rows (unchanged)       : {len(X_test):,}")
print(f"  Class balance in training   : 50% / 50%")
print(f"  Class balance in test       : 94.62% / 5.38% (real-world)")
print(f"  Outputs saved to            : task2_dataset_preparation/outputs/")
print("\nReady for Task 3: Model Training.")