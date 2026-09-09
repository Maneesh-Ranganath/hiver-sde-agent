import os
import chromadb
from chromadb.utils import embedding_functions

class SupportRetriever:
    def __init__(self, collection_name="amazon_support_history"):
        # Initialize local ChromaDB client
        self.client = chromadb.Client()
        
        # Use sentence-transformers embedding function (runs locally, no extra API cost)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
        
        # Seed with initial historical examples if empty
        if self.collection.count() == 0:
            self._seed_initial_data()

    def _seed_initial_data(self):
        """Seeds the vector database with representative historical customer-agent interactions."""
        sample_history = [
            {
                "id": "hist_1",
                "tweet": "My package was supposed to arrive yesterday and tracking hasn't updated!",
                "intent": "INT-01",
                "reply": "We apologize for the delay with your delivery! Please check your tracking link here: https://amazon.com/track or DM us your order ID so we can look into this."
            },
            {
                "id": "hist_2",
                "tweet": "I received a completely broken item in my delivery today, I want a refund.",
                "intent": "INT-02",
                "reply": "We're so sorry to hear your item arrived damaged. Please visit our Returns Center at https://amazon.com/returns to easily request a replacement or refund."
            },
            {
                "id": "hist_3",
                "tweet": "Someone hacked my account and changed my password! Help!",
                "intent": "INT-03",
                "reply": "We take account security very seriously. Please DM us your registered email immediately so our security team can help lock and secure your account."
            },
            {
                "id": "hist_4",
                "tweet": "Why am I being charged for Prime when I canceled it?",
                "intent": "INT-04",
                "reply": "We can certainly check your membership status. Please send us a DM with your account details so we can review any charges and assist with a refund if applicable."
            },
            {
                "id": "hist_5",
                "tweet": "Is the new Echo Dot currently in stock in black?",
                "intent": "INT-05",
                "reply": "Stock availability varies by location! You can check current availability and delivery estimates directly on the product page here: https://amazon.com/dp/example"
            }
        ]

        self.collection.add(
            documents=[item["tweet"] for item in sample_history],
            metadatas=[{"intent": item["intent"], "reply": item["reply"]} for item in sample_history],
            ids=[item["id"] for item in sample_history]
        )

    def retrieve_similar_context(self, query_text: str, n_results: int = 2) -> list:
        """Retrieves top-k similar historical interactions based on semantic proximity."""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            formatted_results = []
            if results and "documents" in results and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                for doc, meta in zip(docs, metas):
                    formatted_results.append({
                        "historical_tweet": doc,
                        "intent": meta["intent"],
                        "historical_reply": meta["reply"]
                    })
            return formatted_results
        except Exception as e:
            print(f"Retrieval warning: {e}")
            return []