import joblib
import medspacy
from medspacy.ner import TargetRule
from spellchecker import SpellChecker
from rapidfuzz import process, fuzz
import numpy as np

# Load medspacy pipeline (which already includes context processing)
nlp = medspacy.load()

# Load valid symptoms from your pickle file
symptom_columns = joblib.load('/home/isaacng33/individual_project/flask_app/artifacts/symptom_columns.pkl')
valid_symptoms = [symptom for symptom in symptom_columns]

# Add target rules so that medspacy knows what to extract
target_matcher = nlp.get_pipe("medspacy_target_matcher")
target_rules = []
for symptom in valid_symptoms:
    target_rules.append(TargetRule(literal=symptom, category="SYMPTOM"))
target_matcher.add(target_rules)

# Helper Function for Plaintext Inference
def create_feature_vector(input_symptoms):
    """Create numpy feature vector from symptoms"""
    features = np.zeros(len(symptom_columns), dtype=np.float32)
    for symptom in input_symptoms:
        if symptom in symptom_columns:
            idx = symptom_columns.index(symptom)
            features[idx] = 1
    return features.reshape(1, -1)

def correct_typos(input_text: str) -> str:
    """
    Corrects typos in the input text using pyspellchecker.
    For each token, if it is not in the dictionary, it replaces it with the correction.
    """
    corrected_tokens = []
    spell = SpellChecker()
    tokens = input_text.split()
    for token in tokens:
        if token.lower() not in spell:
            correction = spell.correction(token)
            corrected_tokens.append(correction if correction else token)
        else:
            corrected_tokens.append(token)
    return " ".join(corrected_tokens)

def correct_symptom_candidate(candidate: str, valid_symptoms: list, threshold: int = 80) -> str:
    """
    Fuzzy-match a candidate word against the valid_symptoms list (which may include multi-word phrases)
    and return the best matching valid symptom if the fuzzy score is above the threshold.
    Otherwise, return the candidate as-is.
    """
    candidate = candidate.lower()
    best_match, score, _ = process.extractOne(candidate, valid_symptoms, scorer=fuzz.token_set_ratio)
    if score >= threshold:
        return best_match.lower()
    return candidate

def fuzzy_match_text(text: str, valid_symptoms: list, threshold: int = 80) -> str:
    """
    Splits the text into tokens, applies fuzzy matching using correct_symptom_candidate,
    and then deduplicates consecutive tokens that have been replaced with the same valid symptom.
    Returns the corrected text.
    """
    tokens = text.split()
    corrected_tokens = [correct_symptom_candidate(token, valid_symptoms, threshold) for token in tokens]
    deduped_tokens = []
    for token in corrected_tokens:
        if not deduped_tokens or token != deduped_tokens[-1]:
            deduped_tokens.append(token)
    return " ".join(deduped_tokens)

# --- Final Pipeline Function ---
def extract_valid_symptoms(input_text: str) -> list:
    """
    Complete NLP pipeline:
      1. Correct typos in the input text.
      2. Apply fuzzy matching to map tokens to valid symptoms (with deduplication).
      3. Process the resulting text with medspacy (which applies target matching and context analysis).
      4. Extract and return the list of valid symptom entities that are not negated.
    """
    # Step 1: Correct typos.
    corrected_text = correct_typos(input_text)
    
    # Step 2: Fuzzy-match tokens and deduplicate them.
    fuzzy_text = fuzzy_match_text(corrected_text, valid_symptoms)
    
    # Step 3: Process the fuzzy-corrected text with medspacy conText
    doc = nlp(fuzzy_text)
    
    # Step 4: Extract entities labeled "SYMPTOM" that are not negated.
    final_symptoms = set()
    for ent in doc.ents:
        if ent.label_ == "SYMPTOM" and not getattr(ent._, "is_negated", False):
            final_symptoms.add(ent.text.lower())
    
    return list(final_symptoms)


