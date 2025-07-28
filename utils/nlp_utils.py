import re
import torch
from nltk.tokenize import sent_tokenize
from indicnlp.tokenize import sentence_tokenize
from transformers import pipeline
import pandas as pd

# Sentiment model pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=0 if torch.cuda.is_available() else -1
)

def split_into_sentences(text, language="en"):
    try:
        if not text or not text.strip():
            print(f"No text provided for sentence splitting (language: {language})")
            return []

        if language == "en":
            try:
                sentences = sent_tokenize(text)
                if len(sentences) > 1:
                    print(f"Successfully split {len(sentences)} sentences using NLTK (English)")
                    return [s.strip() for s in sentences if s.strip()]
            except Exception as nltk_error:
                print(f"NLTK English tokenizer failed: {str(nltk_error)}")

        if language == "ml":
            try:
                sentences = sentence_tokenize.sentence_split(text, lang='mal')
                if len(sentences) > 1:
                    print(f"Successfully split {len(sentences)} sentences using Indic NLP (Malayalam)")
                    return [s.strip() for s in sentences if s.strip()]
            except Exception as indic_error:
                print(f"Indic NLP Malayalam tokenizer failed: {str(indic_error)}")

        print(f"Using fallback regex sentence splitting for language: {language}")
        sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)' r'(?<=\.|\?|\!)\s')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception as e:
        print(f"All sentence splitting methods failed: {str(e)}")
        return [text.strip()] if text.strip() else []

def analyze_sentiment_batch(texts):
    results = sentiment_pipeline(texts)
    outputs = []
    for result in results:
        label = result['label']
        if "1 star" in label or "2 stars" in label:
            sentiment = {"label": "negative", "score": 0.2}
        elif "3 stars" in label:
            sentiment = {"label": "neutral", "score": 0.5}
        elif "4 stars" in label:
            sentiment = {"label": "positive", "score": 0.7}
        elif "5 stars" in label:
            sentiment = {"label": "very positive", "score": 0.9}
        else:
            sentiment = {"label": "neutral", "score": 0.5}
        outputs.append(sentiment)
    return outputs

def detect_intent(text, language="en"):
    text_lower = text.lower().strip()
    intent_keywords = {
        "en": {
            "Strong_interest": ["yes", "definitely", "ready", "want to join", "interested", "share details", "send brochure", "i'll join", "let's proceed", "when can i start", "accept", "excited", "glad to", "eager", "share it", "i'm in"],
            "Moderate_interest": ["maybe", "consider", "think about", "let me think", "tell me more", "more details", "explain", "clarify", "not sure", "possibly", "might", "could be", "depends", "need to check", "will decide", "get back", "discuss", "review", "evaluate"],
            "No_interest": ["can't", "won't", "don't like", "not now", "later", "not suitable", "decline"],
            "Confirmation": ["ok", "interested", "send whatsapp", "got it", "acknowledge", "noted", "please send", "sent details", "agreed"],
            "company_query": ["tino software", "i am calling from tino software"],
            "Qualification_query": ["qualification", "education", "degree", "studying", "course", "background", "university", "college", "bsc", "graduate"],
            "Internship_details": ["internship", "placement", "program", "duration", "data science", "months", "structure"],
            "Location_query": ["online", "offline", "location", "place", "kozhikode", "kochi", "hybrid", "remote"],
            "Certificate_query": ["certificate", "proof", "experience certificate", "training certificate", "completion"],
            "Fee_query": ["fee", "payment", "cost", "6000", "stipend", "salary", "compensation"],
            "Project_details": ["live project", "work", "task", "project", "develop", "build", "implement"]
        },
        "ml": {
            "Strong_interest": ["തയ്യാറാണ്", "ആവശ്യമുണ്ട്", "ചെയ്യാം", "താൽപ്പര്യമുണ്ട്", "ഇഷ്ടമാണ്", "അറിയിച്ചോളൂ"],
            "Moderate_interest": ["ആലോചിക്കാം", "നോക്കാം", "പറയാം", "ചിന്തിക്കാം", "കൂടുതൽ വിവരങ്ങൾ"],
            "No_interest": ["ഇല്ല", "വേണ്ട", "സാധ്യമല്ല", "ഇഷ്ടമല്ല"],
            "Confirmation": ["ശരി", "താല്പര്യമുണ്ട്", "വാട്സാപ്പ്", "കിട്ടി", "അറിയിച്ചു", "സമ്മതം"],
            "company_query": ["ടിനോ സോഫ്റ്റ്വെയർ", "ടിനോ"],
            "Qualification_query": ["വിദ്യാഭ്യാസം", "ഡിഗ്രി", "ബിസി", "പഠിക്കുന്നു", "വിദ്യാർഥി"],
            "Internship_details": ["ഇന്റേൺഷിപ്പ്", "പരിശീലനം", "ഡാറ്റാ സയൻസിലെ"],
            "Location_query": ["ഓൺലൈൻ", "ഓഫ്ലൈൻ", "സ്ഥലം", "വിലാസം", "കൊഴിക്കോട്"],
            "Certificate_query": ["സർട്ടിഫിക്കറ്റ്", "സാക്ഷ്യപത്രം", "കമ്പ്ലീഷൻ"],
            "Fee_query": ["ഫീസ്", "പണം", "6000", "ശമ്പളം", "സ്റ്റൈപെൻഡ്"],
            "Project_details": ["പ്രോജക്ട്", "പ്രവൃത്തി", "ടാസ്‌ക്", "ഡവലപ്പുചെയ്യുക"]
        }
    }

    for intent, keywords in intent_keywords.get(language, {}).items():
        if any(k in text_lower for k in keywords):
            sentiment_score = {
                "Strong_interest": 0.7,
                "Moderate_interest": 0.5,
                "No_interest": 0.2,
                "Confirmation": 0.9
            }.get(intent, 0.5)
            return {"intent": intent, "sentiment": "neutral", "sentiment_score": sentiment_score}
    return {"intent": "Neutral_response", "sentiment": "neutral", "sentiment_score": 0.5}

def detect_keywords(text):
    trigger_words = ['price', 'features', 'complaint', 'satisfaction', 'fraud', 'angry', 'issue', 'problem', 'cost', 'payment', 'certificate', 'duration']
    text_lower = text.lower()
    found = [word for word in trigger_words if word in text_lower]
    return list(set(found))

def calculate_intent_score(segment_results):
    """
    Calculate intent score (0-100) based on segments with positive intent.
    Accepts list of segment analysis results from detect_intent.
    """
    positive_labels = ["Strong_interest", "Moderate_interest", "Fee_query", "Confirmation"]
    positive_count = sum(1 for s in segment_results if s['intent'] in positive_labels)
    total = len(segment_results)
    return int((positive_count / total) * 100) if total > 0 else 0

def calculate_lead_score(intent_score, sentiment, trigger_words):
    sentiment_weight = {
        "POSITIVE": 30,
        "NEGATIVE": -30,
        "NEUTRAL": 10
    }

    trigger_weight = 10 * len(trigger_words)

    sentiment_component = sentiment_weight.get(sentiment.upper(), 0)

    final_score = intent_score * 0.5 + sentiment_component * 0.3 + trigger_weight * 0.2
    return min(100, max(0, final_score))
