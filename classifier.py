from transformers import pipeline
from config import INTENTS, DEFAULT_ROUTING

# Initialize local zero-shot classification model from Hugging Face
print("Loading local classification model (bart-large-mnli)...")
classifier_pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

class ClassificationResult:
    def __init__(self, intent_id, intent_name, confidence, suggested_action, reasoning):
        self.intent_id = intent_id
        self.intent_name = intent_name
        self.confidence = confidence
        self.suggested_action = suggested_action
        self.reasoning = reasoning

def classify_tweet(tweet_text: str) -> ClassificationResult:
    """Classifies an incoming tweet locally using Hugging Face pipelines."""
    intent_labels = list(INTENTS.values())
    
    try:
        # Run local zero-shot classification
        result = classifier_pipeline(
            tweet_text,
            candidate_labels=intent_labels,
            multi_label=False
        )
        
        # Extract best match
        best_label = result['labels'][0]
        best_score = float(result['scores'][0])
        
        # Map label back to intent ID
        intent_id = "INT-01"
        for k, v in INTENTS.items():
            if v == best_label:
                intent_id = k
                break
                
        action = DEFAULT_ROUTING.get(intent_id, "AUTO_REPLY")
        
        # Override action to ESCALATE if security keywords or account issues are present
        if intent_id == "INT-03" or any(keyword in tweet_text.lower() for keyword in ["hack", "password", "unauthorized", "stolen", "locked"]):
            action = "ESCALATE"
            intent_id = "INT-03"
            best_label = INTENTS["INT-03"]

        return ClassificationResult(
            intent_id=intent_id,
            intent_name=best_label,
            confidence=round(best_score, 2),
            suggested_action=action,
            reasoning=f"Classified locally with confidence {best_score}"
        )
        
    except Exception as e:
        return ClassificationResult(
            intent_id="INT-01",
            intent_name=INTENTS["INT-01"],
            confidence=0.5,
            suggested_action="AUTO_REPLY",
            reasoning=f"Fallback triggered: {str(e)}"
        )