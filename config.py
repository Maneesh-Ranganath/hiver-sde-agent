import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API Key setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# Define the 6 Core Intents for @AmazonHelp
INTENTS = {
    "INT-01": "Order & Delivery Tracking",
    "INT-02": "Refunds, Returns & Cancellations",
    "INT-03": "Account Access & Security",
    "INT-04": "Digital Services & Subscriptions",
    "INT-05": "Product Inquiries & Availability",
    "INT-06": "General Feedback & Dissatisfaction"
}

# Routing rules mapping intents to actions
DEFAULT_ROUTING = {
    "INT-01": "AUTO_REPLY",
    "INT-02": "AUTO_REPLY",
    "INT-03": "ESCALATE",  # Security issues always escalate
    "INT-04": "AUTO_REPLY",
    "INT-05": "AUTO_REPLY",
    "INT-06": "AUTO_REPLY"
}