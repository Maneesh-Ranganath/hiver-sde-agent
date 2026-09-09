from classifier import classify_tweet
from retriever import SupportRetriever

# Initialize local retriever database
retriever = SupportRetriever()

def run_agent_pipeline(tweet_text: str):
    print(f"\nIncoming Tweet: '{tweet_text}'")
    print("-" * 50)
    
    # Step 1: Classify intent locally
    classification = classify_tweet(tweet_text)
    print(f" [Intent Classified]: {classification.intent_id} - {classification.intent_name} (Confidence: {classification.confidence})")
    print(f" [Routing Decision]: {classification.suggested_action} (Reason: {classification.reasoning})")
    
    # Step 2: Retrieve similar historical context (RAG)
    similar_cases = retriever.retrieve_similar_context(tweet_text, n_results=1)
    
    # Step 3: Generate response based on local RAG match or escalation rule
    if classification.suggested_action == "ESCALATE":
        final_reply = "⚠️ [ESCALATED TO HUMAN AGENT]: High security/risk detected. Routed to human support queue."
    elif similar_cases:
        matched = similar_cases[0]
        final_reply = f"@AmazonHelp {matched['historical_reply']}"
    else:
        final_reply = "@AmazonHelp Thanks for reaching out! Please DM us your details so we can assist you further."

    print(f" [Generated Reply]: {final_reply}")
    print("=" * 50)
    return {
        "intent": classification.intent_id,
        "action": classification.suggested_action,
        "reply": final_reply
    }

if __name__ == "__main__":
    # Test sample run with two different tweets
    test_tweets = [
        "Where is my package? It was supposed to be delivered 2 days ago!",
        "Someone just accessed my account and changed my password without my permission!"
    ]
    
    for tweet in test_tweets:
        run_agent_pipeline(tweet)