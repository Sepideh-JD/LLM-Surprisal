# LLM-Surprisal
Can we use computer-extracted features (like surprisal, word counts, POS ratios) to automatically predict which patients have problems like "grammatical_errors" or "semantic_errors"?


import pandas as pd

# Load your data
df = pd.read_csv('picture_linguistic_20250917.csv')  # Replace with your actual file path

# Basic inspection
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

# Define the annotation columns (your targets to predict)
annotation_cols = [
    'grammatical_errors', 'intelligibility', 'nonspecific_terms',
    'other_cognitive_communication_skills', 'repeated_sounds', 
    'semantic_errors', 'sound_sequencing_errors', 'sound_substitutions',
    'verbal_asides', 'word_or_phrase_repetitions'
]

# Check how many cases have each annotation
print("Annotation counts (how many have each issue):")
print(df[annotation_cols].sum())

# Check the 'normal' column
print("\nControl cases (normal==1):", df['normal'].sum())


# Feature columns (linguistic measures)
feature_cols = [
    'avg_surprisal', 'word_count', 'unique_word_count', 'root_ttr',
    'avg_word_length', 'long_word_count', 'sentence_count', 'avg_sentence_length',
    'pos_DET_ratio', 'pos_ADJ_ratio', 'pos_NOUN_ratio', 'pos_AUX_ratio',
    'pos_VERB_ratio', 'pos_ADP_ratio', 'pos_PRON_ratio', 
    'readability_flesch_reading_ease', 'total_entities',
    'pos_CCONJ_ratio', 'pos_PART_ratio', 'pos_SCONJ_ratio',
    'pos_PROPN_ratio', 'pos_ADV_ratio', 'pos_NUM_ratio', 
    'pos_INTJ_ratio', 'pos_X_ratio', 'pos_PUNCT_ratio',
    'semantic_deviation_score'
]

# Check for missing values in features
print("\nMissing values in features:")
print(df[feature_cols].isnull().sum().sum())

# Remove sound_sequencing_errors from annotations (all zeros)
annotation_cols.remove('sound_sequencing_errors')
print(f"\nAnnotations to predict ({len(annotation_cols)} total):")
print(annotation_cols)







from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# Pick one annotation to start
target = 'grammatical_errors'

# Create a simple dataset: positives (has error) vs controls (normal==1)
positives = df[df[target] == 1.0]
controls = df[df['normal'] == 1.0]
simple_data = pd.concat([positives, controls])

# Prepare X (features) and y (target)
X = simple_data[['avg_surprisal']].values
y = simple_data[target].values

print(f"\nPredicting: {target}")
print(f"Positive cases: {y.sum()}")
print(f"Controls: {len(y) - y.sum()}")
print(f"Total samples: {len(y)}")






# Logistic regression with class balancing
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

# Leave-One-Out cross-validation
loo = LeaveOneOut()
y_probs = []
y_true = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[0, 1]  # probability of class 1
    
    y_probs.append(y_prob)
    y_true.append(y_test[0])

y_probs = np.array(y_probs)
y_true = np.array(y_true)

# Calculate metrics
auc = roc_auc_score(y_true, y_probs)
auprc = average_precision_score(y_true, y_probs)

print(f"\nResults for predicting {target} using avg_surprisal:")
print(f"ROC-AUC: {auc:.3f}")
print(f"AUPRC (Average Precision): {auprc:.3f}")






########RUN 243 MODELS! CRAZY!##########
\


"""
# Test ALL features against ALL annotations
results_list = []

print("\nTesting all features for all annotations...")
print("=" * 70)

for target in annotation_cols:
    # Create dataset: positives vs controls
    positives = df[df[target] == 1.0]
    controls = df[df['normal'] == 1.0]
    data = pd.concat([positives, controls])
    y = data[target].values
    
    n_positives = int(y.sum())
    print(f"\n{target}: {n_positives} cases")
    
    for feat in feature_cols:
        X_feat = data[[feat]].values
        
        # LOOCV
        y_probs = []
        for train_idx, test_idx in loo.split(X_feat):
            model.fit(X_feat[train_idx], y[train_idx])
            y_probs.append(model.predict_proba(X_feat[test_idx])[0, 1])
        
        # Metrics
        auc = roc_auc_score(y, y_probs)
        auprc = average_precision_score(y, y_probs)
        
        results_list.append({
            'annotation': target,
            'feature': feat,
            'n_cases': n_positives,
            'auc': auc,
            'auprc': auprc
        })

# Convert to dataframe and show top results
results_df = pd.DataFrame(results_list)
print("\n" + "=" * 70)
print("TOP 10 FEATURE-ANNOTATION PAIRS BY AUC:")
print(results_df.nlargest(10, 'auc')[['annotation', 'feature', 'auc', 'auprc']])

"""

# Investigate the perfect predictions
print("\n" + "=" * 70)
print("INVESTIGATING PERFECT PREDICTIONS (AUC = 1.0)")
print("=" * 70)

# Check pos_PROPN_ratio -> grammatical_errors
target = 'grammatical_errors'
feature = 'pos_PROPN_ratio'

cases = df[df[target] == 1.0]
ctrls = df[df['normal'] == 1.0]

print(f"\n1. {feature} -> {target}:")
print(f"   Cases with {target} (n={len(cases)}):")
print(f"   {feature} values: {cases[feature].values}")
print(f"   Mean: {cases[feature].mean():.4f}, Std: {cases[feature].std():.4f}")
print(f"\n   Controls (n={len(ctrls)}):")
print(f"   {feature} range: [{ctrls[feature].min():.4f}, {ctrls[feature].max():.4f}]")
print(f"   Mean: {ctrls[feature].mean():.4f}, Std: {ctrls[feature].std():.4f}")

# Check pos_X_ratio -> verbal_asides
target = 'verbal_asides'
feature = 'pos_X_ratio'

cases = df[df[target] == 1.0]
ctrls = df[df['normal'] == 1.0]

print(f"\n2. {feature} -> {target}:")
print(f"   Cases with {target} (n={len(cases)}):")
print(f"   {feature} values: {cases[feature].values}")
print(f"   Mean: {cases[feature].mean():.4f}, Std: {cases[feature].std():.4f}")
print(f"\n   Controls (n={len(ctrls)}):")
print(f"   {feature} range: [{ctrls[feature].min():.4f}, {ctrls[feature].max():.4f}]")
print(f"   Mean: {ctrls[feature].mean():.4f}, Std: {ctrls[feature].std():.4f}")





################## combine multiple features
# B. Multi-feature models
print("\n" + "=" * 70)
print("PART B: MULTI-FEATURE MODELS")
print("=" * 70)

# Start with grammatical_errors
target = 'grammatical_errors'
positives = df[df[target] == 1.0]
controls = df[df['normal'] == 1.0]
data = pd.concat([positives, controls])
y = data[target].values

# Test different feature combinations
feature_sets = {
    'Single: avg_surprisal': ['avg_surprisal'],
    'Surprisal + Lexical': ['avg_surprisal', 'root_ttr', 'unique_word_count'],
    'Surprisal + Syntax': ['avg_surprisal', 'avg_sentence_length', 'pos_VERB_ratio'],
    'Surprisal + Semantic': ['avg_surprisal', 'semantic_deviation_score'],
    'All strong features': ['avg_surprisal', 'root_ttr', 'semantic_deviation_score', 
                            'avg_sentence_length', 'pos_NOUN_ratio', 'pos_VERB_ratio']
}

print(f"\nPredicting: {target} (n={int(y.sum())} cases)")
print("-" * 70)

for name, features in feature_sets.items():
    X = data[features].values
    y_probs = []
    
    for train_idx, test_idx in loo.split(X):
        model.fit(X[train_idx], y[train_idx])
        y_probs.append(model.predict_proba(X[test_idx])[0, 1])
    
    auc = roc_auc_score(y, y_probs)
    auprc = average_precision_score(y, y_probs)
    
    print(f"{name:30s} | Features: {len(features)} | AUC: {auc:.3f} | AUPRC: {auprc:.3f}")












############################Next step: Test multi-feature models on annotations with MORE cases
    # Test multi-feature models on larger annotation samples
print("\n" + "=" * 70)
print("TESTING MULTI-FEATURE MODELS ON LARGER SAMPLES")
print("=" * 70)

targets_to_test = ['semantic_errors', 'other_cognitive_communication_skills']

for target in targets_to_test:
    positives = df[df[target] == 1.0]
    controls = df[df['normal'] == 1.0]
    data = pd.concat([positives, controls])
    y = data[target].values
    
    print(f"\n{target} (n={int(y.sum())} cases)")
    print("-" * 70)
    
    for name, features in feature_sets.items():
        X = data[features].values
        y_probs = []
        
        for train_idx, test_idx in loo.split(X):
            model.fit(X[train_idx], y[train_idx])
            y_probs.append(model.predict_proba(X[test_idx])[0, 1])
        
        auc = roc_auc_score(y, y_probs)
        auprc = average_precision_score(y, y_probs)
        
        print(f"{name:30s} | AUC: {auc:.3f} | AUPRC: {auprc:.3f}")







        #############################DEMOGRAPHICS

        # C. Add demographics
print("\n" + "=" * 70)
print("PART C: ADDING DEMOGRAPHICS")
print("=" * 70)

# Check demographic data
print("\nDemographic variables:")
print("Age at Consent - missing:", df['Age at Consent'].isna().sum())
print("Gender - values:", df['Gender'].value_counts().to_dict())
print("Race - missing:", df['Race'].isna().sum())
print("Ethnicity - missing:", df['Ethnicity'].isna().sum())

# Prepare demographic features for the two promising targets
from sklearn.preprocessing import LabelEncoder

targets_to_test = ['semantic_errors', 'other_cognitive_communication_skills']

for target in targets_to_test:
    positives = df[df[target] == 1.0]
    controls = df[df['normal'] == 1.0]
    data = pd.concat([positives, controls]).copy()
    
    # Encode demographics
    data['Age'] = pd.to_numeric(data['Age at Consent'], errors='coerce').fillna(data['Age at Consent'].median())
    data['Gender_encoded'] = LabelEncoder().fit_transform(data['Gender'].fillna('Unknown'))
    
    # Linguistic features only
    X_ling = data[feature_sets['All strong features']].values
    # Linguistic + demographics
    X_demo = data[feature_sets['All strong features'] + ['Age', 'Gender_encoded']].values
    
    y = data[target].values
    
    print(f"\n{target} (n={int(y.sum())} cases)")
    print("-" * 70)
    
    # Test with and without demographics
    for X, label in [(X_ling, 'Linguistic only (6 features)'), 
                     (X_demo, 'Linguistic + Demographics (8 features)')]:
        y_probs = []
        for train_idx, test_idx in loo.split(X):
            model.fit(X[train_idx], y[train_idx])
            y_probs.append(model.predict_proba(X[test_idx])[0, 1])
        
        auc = roc_auc_score(y, y_probs)
        auprc = average_precision_score(y, y_probs)
        print(f"{label:40s} | AUC: {auc:.3f} | AUPRC: {auprc:.3f}")





        ########################AGE AND GENDER SEPARATELY

        # Test age and gender separately
print("\n" + "=" * 70)
print("SEPARATING AGE AND GENDER CONTRIBUTIONS")
print("=" * 70)

for target in targets_to_test:
    positives = df[df[target] == 1.0]
    controls = df[df['normal'] == 1.0]
    data = pd.concat([positives, controls]).copy()
    
    # Encode demographics
    data['Age'] = pd.to_numeric(data['Age at Consent'], errors='coerce').fillna(data['Age at Consent'].median())
    data['Gender_encoded'] = LabelEncoder().fit_transform(data['Gender'].fillna('Unknown'))
    
    # Different feature combinations
    X_ling = data[feature_sets['All strong features']].values
    X_age = data[feature_sets['All strong features'] + ['Age']].values
    X_gender = data[feature_sets['All strong features'] + ['Gender_encoded']].values
    X_both = data[feature_sets['All strong features'] + ['Age', 'Gender_encoded']].values
    
    y = data[target].values
    
    print(f"\n{target} (n={int(y.sum())} cases)")
    print("-" * 70)
    
    # Test each combination
    for X, label in [
        (X_ling, 'Linguistic only (6 features)'),
        (X_age, 'Linguistic + Age (7 features)'),
        (X_gender, 'Linguistic + Gender (7 features)'),
        (X_both, 'Linguistic + Age + Gender (8 features)')
    ]:
        y_probs = []
        for train_idx, test_idx in loo.split(X):
            model.fit(X[train_idx], y[train_idx])
            y_probs.append(model.predict_proba(X[test_idx])[0, 1])
        
        auc = roc_auc_score(y, y_probs)
        auprc = average_precision_score(y, y_probs)
        print(f"{label:40s} | AUC: {auc:.3f} | AUPRC: {auprc:.3f}")
