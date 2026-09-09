from classifier import classify_tweet
from retriever import SupportRetriever

# Initialize local retriever
retriever = SupportRetriever()

# Define a small Golden Evaluation Set for testing
GOLDEN_TEST_SET = [
    {
        "tweet": "Where is my package? It was supposed to be delivered 2 days ago!",
        "true_intent": "INT-01",
        "true_action": "AUTO_REPLY"
    },
    {
        "tweet": "I received a completely broken item in my delivery today, I want a refund.",
        "true_intent": "INT-02",
        "true_action": "AUTO_REPLY"
    },
    {
        "tweet": "Someone hacked my account and changed my password! Help!",
        "true_intent": "INT-03",
        "true_action": "ESCALATE"
    },
    {
        "tweet": "Why am I being charged for Prime when I canceled it?",
        "true_intent": "INT-04",
        "true_action": "AUTO_REPLY"
    },
    {
        "tweet": "Is the new Echo Dot currently in stock in black?",
        "true_intent": "INT-05",
        "true_action": "AUTO_REPLY"
    }
]

def baseline_1_keyword_rules(tweet: str):
    """Baseline 1: Trivial Keyword Rule-Matcher"""
    text = tweet.lower()
    if "hack" in text or "password" in text or "unauthorized" in text:
        return {"intent": "INT-03", "action": "ESCALATE"}
    elif "refund" in text or "broken" in text or "return" in text:
        return {"intent": "INT-02", "action": "AUTO_REPLY"}
    elif "where" in text or "package" in text or "delivered" in text:
        return {"intent": "INT-01", "action": "AUTO_REPLY"}
    elif "prime" in text or "charged" in text:
        return {"intent": "INT-04", "action": "AUTO_REPLY"}
    elif "stock" in text or "available" in text:
        return {"intent": "INT-05", "action": "AUTO_REPLY"}
    else:
        return {"intent": "INT-06", "action": "AUTO_REPLY"}

def baseline_2_retriever_only(tweet: str):
    """Baseline 2: Pure Retriever matching without intent classification layer"""
    matches = retriever.retrieve_similar_context(tweet, n_results=1)
    if matches:
        return {"intent": matches[0]["intent"], "action": "AUTO_REPLY"}
    return {"intent": "INT-01", "action": "AUTO_REPLY"}

def proposed_pipeline(tweet: str):
    """Proposed System: Classifier + RAG + Guardrail Routing"""
    res = classify_tweet(tweet)
    return {"intent": res.intent_id, "action": res.suggested_action}

def run_evaluation():
    print("=" * 60)
    print("RUNNING AUTOMATED EVALUATION HARNESS & BASELINE COMPARISON")
    print("=" * 60)
    
    b1_correct = 0
    b2_correct = 0
    proposed_correct = 0
    
    total = len(GOLDEN_TEST_SET)
    
    for item in GOLDEN_TEST_SET:
        tweet = item["tweet"]
        true_intent = item["true_intent"]
        
        # Evaluate Baseline 1
        res_b1 = baseline_1_keyword_rules(tweet)
        if res_b1["intent"] == true_intent:
            b1_correct += 1
            
        # Evaluate Baseline 2
        res_b2 = baseline_2_retriever_only(tweet)
        if res_b2["intent"] == true_intent:
            b2_correct += 1
            
        # Evaluate Proposed Pipeline
        res_prop = proposed_pipeline(tweet)
        if res_prop["intent"] == true_intent:
            proposed_correct += 1

    print(f"\n--- EVALUATION RESULTS (Accuracy over {total} test samples) ---")
    print(f" Baseline 1 (Keyword Rules):      {(b1_correct / total) * 100:.1f}%")
    print(f" Baseline 2 (Retriever-Only):     {(b2_correct / total) * 100:.1f}%")
    print(f" Proposed System (Our Pipeline):  {(proposed_correct / total) * 100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()