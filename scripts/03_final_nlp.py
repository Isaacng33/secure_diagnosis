import joblib
import medspacy
from medspacy.ner import TargetRule
from spellchecker import SpellChecker
from rapidfuzz import process, fuzz
from medspacy.visualization import visualize_dep
import webbrowser

# Load medspacy pipeline (which already includes context processing)
nlp = medspacy.load()

# Load valid symptoms from your pickle file
symptom_columns = joblib.load('./models/symptom_columns.pkl')
valid_symptoms = [symptom for symptom in symptom_columns]

# Add target rules so that medspacy knows what to extract
target_matcher = nlp.get_pipe("medspacy_target_matcher")
target_rules = []
for symptom in valid_symptoms:
    target_rules.append(TargetRule(literal=symptom, category="SYMPTOM"))
target_matcher.add(target_rules)

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
    entities = []
    
    for ent in doc.ents:
        is_negated = getattr(ent._, "is_negated", False)
        if ent.label_ == "SYMPTOM":
            # Add to symptoms list if not negated
            if not is_negated:
                final_symptoms.add(ent.text.lower())
            
            # Create visualization data
            entities.append({
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
                "negated": is_negated
            })
    
    # Generate custom HTML visualization with negation colors
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .entity { padding: 0.25em; border-radius: 0.25em; }
            .symptom { background: #7aecec; }
            .negated { background: #ff9999; border: 2px solid #ff4d4d; }
        </style>
    </head>
    <body>
        <div style="margin: 1em; line-height: 2em;">
    """
    
    text = doc.text
    last_pos = 0
    for ent in sorted(entities, key=lambda x: x["start"]):
        html += text[last_pos:ent["start"]]
        class_name = "negated" if ent["negated"] else "symptom"
        html += f'<mark class="entity {class_name}">{text[ent["start"]:ent["end"]]}' \
                f'<span style="font-size: 0.8em; margin-left: 0.5em;">{ent["label"]}' \
                f'{" (NEGATED)" if ent["negated"] else ""}</span></mark>'
        last_pos = ent["end"]
    
    html += text[last_pos:] + "</div></body></html>"
    
    # Save and open visualization
    with open("negation_visualization.html", "w") as f:
        f.write(html)
    webbrowser.open("negation_visualization.html")
    
    return list(final_symptoms)

if __name__ == "__main__":
    sample_text = (
        "I have a headache but no fever."
    )
    
    symptoms = extract_valid_symptoms(sample_text)
    print("Final Extracted Valid Symptoms:", symptoms)
    

