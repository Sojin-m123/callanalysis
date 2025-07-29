import re
import torch
from nltk.tokenize import sent_tokenize
from indicnlp.tokenize import sentence_tokenize
from transformers import pipeline
import pandas as pd


def split_into_sentences(text: str, language: str = "en") -> list[str]:
    
    try:
        if not text or not text.strip():
            print(f"No text provided for sentence splitting (language: {language})")
            return []

        print(f"Using regex sentence splitting as primary for language: {language}")
        sentence_endings = re.compile(
            r'(?<!\b[A-Z][a-z]\.)(?<!\b[A-Z]\.)(?<!\d\.\d)(?<=[.,?!।॥]|\n)(?=\s|$|[^\s.,?!])'
        )
        split_pattern = re.compile(r'\s*,\s*\.\s*\?\s*')
        common_abbreviations = {
            'en': r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|Inc|Co|Ltd|U\.S|yes)\.',
            'ml': r'\b(?:ഡോ|ശ്രീ|ശ്രീമതി|പ്രൊ|കോ|യെസ്)\.'
        }
        abbr_pattern = common_abbreviations.get(language, r'\b(?:Dr|Mr|Mrs|Ms)\.')

        sentences = sentence_endings.split(text)
        cleaned_sentences = []
        current_sentence = ""

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            split_sentences = split_pattern.split(sent)
            split_sentences = [s.strip() for s in split_sentences if s.strip()]

            for split_sent in split_sentences:
                if current_sentence:
                    if re.search(abbr_pattern + r'$', current_sentence):
                        current_sentence += " " + split_sent
                    else:
                        cleaned_sentences.append(current_sentence)
                        current_sentence = split_sent
                else:
                    current_sentence = split_sent
        if current_sentence:
            cleaned_sentences.append(current_sentence)

        
        final_sentences = []
        for sent in cleaned_sentences:
            subsentences = re.split(
                r'(?<!\b[A-Z][a-z]\.)(?<!\b[A-Z]\.)(?<!\d\.\d)(?<=[.,])(?=\s|$|[^\s.,?!])',
                sent
            )
            final_sentences.extend(subsent.strip() for subsent in subsentences if subsent.strip())

        if final_sentences:
            print(f"Regex split resulted in {len(final_sentences)} sentences")
            print(f"Sentences: {final_sentences}")
            return final_sentences
        else:
            print(f"Regex splitter returned no sentences for {language}, falling back to Indic NLP")

        lang_code = 'eng' if language == "en" else 'mal'
        try:
            sentences = sentence_tokenize.sentence_split(text, lang=lang_code)
            if sentences:
                print(f"Successfully split {len(sentences)} sentences using Indic NLP ({language})")
                cleaned_sentences = []
                current_sentence = ""

                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    split_sentences = split_pattern.split(sent)
                    split_sentences = [s.strip() for s in split_sentences if s.strip()]

                    for split_sent in split_sentences:
                        if current_sentence:
                            if re.search(abbr_pattern + r'$', current_sentence):
                                current_sentence += " " + split_sent
                            else:
                                cleaned_sentences.append(current_sentence)
                                current_sentence = split_sent
                        else:
                            current_sentence = split_sent
                if current_sentence:
                    cleaned_sentences.append(current_sentence)

                final_sentences = []
                for sent in cleaned_sentences:
                    subsentences = re.split(
                        r'(?<!\b[A-Z][a-z]\.)(?<!\b[A-Z]\.)(?<!\d\.\d)(?<=[.,])(?=\s|$|[^\s.,?!])',
                        sent
                    )
                    final_sentences.extend(subsent.strip() for subsent in subsentences if subsent.strip())

                print(f"Indic NLP post-processed into {len(final_sentences)} sentences")
                print(f"Sentences: {final_sentences}")
                return final_sentences
            else:
                print(f"Indic NLP returned no sentences for {language}")
        except Exception as indic_error:
            print(f"Indic NLP {language} tokenizer failed: {str(indic_error)}")

        print("All splitting methods failed, returning text as single sentence")
        return [text.strip()] if text.strip() else []

    except Exception as e:
        print(f"Sentence splitting error: {str(e)}")
        return [text.strip()] if text.strip() else []

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=0 if torch.cuda.is_available() else -1
)

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
    """Enhanced intent detection for internship interest analysis in English, Malayalam, and Tamil"""
    import logging
    logger = logging.getLogger(__name__)

    try:
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid input text for detect_intent: {text}. Returning Neutral_response.")
            return {"intent": "Neutral_response", "sentiment": "neutral", "sentiment_score": 0.5}

        text_lower = text.lower().strip()

        intent_keywords = {
            "en": {
                "Strong_interest": [
                    "definitely", "ready", "want to join", "interested",
                    "share details", "send brochure", "i'll join", "let's proceed",
                    "where do i sign", "share", "when can i start", "accept",
                    "looking forward", "excited", "happy to", "glad to", "eager",
                    "share it", "whatsapp", "i'm in"
                ],
                "Moderate_interest": [
                    "maybe", "consider", "think about", "let me think", "tell me more",
                    "more details", "explain", "clarify", "not sure", "possibly",
                    "might", "could be", "depends", "need to check", "will decide",
                    "get back", "discuss", "consult", "review", "evaluate"
                ],
                "No_interest": [
                    "can't", "won't", "don't like",
                    "not now", "later", "not suitable","not looking ", "decline"
                ],
                "company_query": [
                    "tino software and security solutions", "Tino software IT company", "Tino software",
                    "i am calling you from tino software and security solutions", "tinos software"
                ],
                "Qualification_query": [
                    "qualification", "education", "computer science", "degree", "studying", "course",
                    "background", "academics", "university", "college", "bsc",
                    "graduate", "year of study", "curriculum", "syllabus"
                ],
                "Internship_details": [
                    "internship", "placement", "program", "is looking for an internship", "duration",
                    "Data Science", "months", "period", "schedule", "timing", "timeframe",
                    "1 to 3", "three months", "structure", "plan", "framework",
                    "looking for an internship in data science"
                ],
                "Location_query": [
                    "online", "offline", "location", "place", "where",
                    "address", "relocate", "relocating", "from", "coming",
                    "kozhikode", "kochi", "palarivattam", "hybrid", "remote"
                ],
                "Certificate_query": [
                    "certificate", "certification", "document", "proof",
                    "experience certificate", "training certificate", "letter",
                    "completion", "award", "recognition"
                ],
                "Fee_query": [
                    "fee", "payment", "cost", "amount", "charge",
                    "6000", "six thousand", "money", "stipend", "salary",
                    "compensation", "paid", "free"
                ],
                "Project_details": [
                    "live project", "work", "assignment", "task", "project",
                    "trainee", "superiors", "team", "collaborate", "develop",
                    "build", "create", "implement", "hands-on", "practical"
                ],
                "Confirmation": [
                    "interested", "send whatsapp", "got it","I was looking for it Ok",
                    "acknowledge", "noted", "please send", "sent details", "agreed"
                ]
            },
            "ml": {
                "Strong_interest": [
                    "തയ്യാറാണ്", "ആവശ്യമുണ്ട്", "ചെയ്യാം", "ആഗ്രഹമുണ്ട്",
                    "ഇഷ്ടമാണ്", "അറിയിച്ചോളൂ", "താൽപ്പര്യമുണ്ട്.", "ബ്രോഷർ വേണം", "വിശദാംശങ്ങൾ വേണം",
                    "ശെയർ ചെയ്യുക", "ഞാൻ വരാം", "താൽപ്പര്യപ്പെടുന്നു", "ഉത്സാഹം", "താത്പര്യം",
                    "സമ്മതം", "അംഗീകരിക്കുന്നു", "ഹാപ്പിയാണ്", "ഞാൻ ചെയ്യാം",
                    "വാട്സാപ്പിൽ അയക്കൂ", "ആവശ്യമാണ്"
                ],
                "Moderate_interest": [
                    "ആലോചിക്കാം", "നോക്കാം", "താല്പര്യമുണ്ട്", "ഇന്റെറസ്റ്റഡ്",
                    "പറയാം", "ക്ഷണിക്കുക", "ചിന്തിക്കാം", "കാണാം", "ഉത്തരമില്ല",
                    "കൂടുതൽ വിവരങ്ങൾ", "വ്യാഖ്യാനിക്കുക", "അവലംബിക്കുക"
                ],
                "No_interest": [
                    "ഇല്ല", "വേണ്ട", "സാധ്യമല്ല", "ഇഷ്ടമല്ല","നോക്കുന്നില്ല."
                ],
                "company_query": [
                    "ടിനോ സോഫ്റ്റ്വെയറിൽ", "ടിനോ സോഫ്റ്റ്വെയർ", "ടിനോ"
                ],
                "Qualification_query": [
                    "വിദ്യാഭ്യാസം", "ഡിഗ്രി", "ബിസി", "പഠിക്കുന്നു",
                    "പഠനം", "അധ്യയനം", "ക്ലാസ്", "വർഷം",
                    "കോഴ്‌സ്", "സിലബസ്", "വിദ്യാർഥി", "ഗണിതം", "സയൻസ്"
                ],
                "Internship_details": [
                    "ഇന്റെണ്ഷിപ്", "പരിശീലനം", "ഡാറ്റാ സയൻസിലെ", "ഇന്റേൺഷിപ്പിനൊപ്പം", "പ്ലെയ്സ്മെന്റ്",
                    "മാസം", "സമയക്രമം", "ടൈമിംഗ്", "1 മുതൽ 3 വരെ",
                    "അവസാന വർഷം", "ലൈവ്", "ഫ്രെയിംവർക്ക്", "സ്ഥിരമായി",
                    "ഡാറ്റാ സയൻസിലെ", "ഇന്റേൺഷിപ്പ്", "ഡാറ്റാ സയൻസിലെ ഇന്റേൺഷിപ്പ്"
                ],
                "Location_query": [
                    "ഓൺലൈൻ", "ഓഫ്ലൈൻ", "സ്ഥലം", "വിലാസം", "കഴിഞ്ഞ്",
                    "എവിടെ", "കൊഴിക്കോട്", "പാലാരിവട്ടം", "മാറ്റം",
                    "റിലൊക്കേറ്റ്", "വരുന്നു", "എവിടെ നിന്നാണ്", "ഹൈബ്രിഡ്", "വിലാസം"
                ],
                "Certificate_query": [
                    "സർട്ടിഫിക്കറ്റ്", "ഡോക്യുമെന്റ്", "പ്രമാണം", "സാക്ഷ്യപത്രം", "കമ്പ്ലീഷൻ"
                ],
                "Fee_query": [
                    "ഫീസ്", "പണം", "6000", "ആറ് ആയിരം", "കാണിക്ക്",
                    "മാസതൊട്ടി", "ചാർജ്", "റുമണറേഷൻ", "ഫ്രീ",
                    "ശമ്പളം", "സ്റ്റൈപെൻഡ്"
                ],
                "Project_details": [
                    "പ്രോജക്ട്", "ലൈവ് പ്രോജക്ട്", "പ്രവൃത്തി", "ടാസ്‌ക്",
                    "ടീം", "മേധാവി", "ട്രെയിനി", "സഹപ്രവർത്തനം", "പ്രോജക്റ്റുകൾ",
                    "ഡവലപ്പുചെയ്യുക", "സൃഷ്ടിക്കുക", "ഇമ്പ്ലിമെന്റുചെയ്യുക",
                    "പ്രായോഗികം", "അഭ്യാസം"
                ],
                "Confirmation": [
                    "ശരി", "താല്പര്യമുണ്ട്", "തിരയുന്നു", "ഇഷ്ടമുണ്ട്", "വാട്സാപ്പിൽ അയക്കൂ", "ഷെയർ ചെയ്യുക",
                    "വാട്സാപ്പ്", "വാട്ട്സാപ്പ്", "കിട്ടി", "അറിയിച്ചു",
                    "നോട്ടു ചെയ്തു", "സമ്മതം", "അംഗീകരിച്ചു", "ഓക്കെ", "യെസ്",
                    "അക്ക്നലഡ്ജ്", "ക്ലിയർ", "തയാറാണ്", "അറിയിപ്പ് ലഭിച്ചു",
                    "വാട്ട്സ്ആപ്പിലേ", "ഞാൻ അതിനായി നോക്കിയിരുന്നു"
                ]
            },


        }

        # Check if language is supported
        if language not in intent_keywords:
            logger.warning(f"Unsupported language {language} in intent_keywords. Returning Neutral_response.")
            return {"intent": "Neutral_response", "sentiment": "neutral", "sentiment_score": 0.5}

        # Check for each intent type in order of priority
        if any(keyword in text_lower for keyword in intent_keywords[language]["Confirmation"]):
            logger.debug(f"Detected intent: Confirmation for text: {text_lower} in language: {language}")
            return {"intent": "Confirmation", "sentiment": "very positive", "sentiment_score": 0.9}

        if any(keyword in text_lower for keyword in intent_keywords[language]["Strong_interest"]):
            logger.debug(f"Detected intent: Strong_interest for text: {text_lower} in language: {language}")
            return {"intent": "Strong_interest", "sentiment": "positive", "sentiment_score": 0.7}

        if any(keyword in text_lower for keyword in intent_keywords[language]["company_query"]):
            logger.debug(f"Detected intent: company_query for text: {text_lower} in language: {language}")
            return {"intent": "company_query", "sentiment": "neutral", "sentiment_score": 0.5}

        if any(keyword in text_lower for keyword in intent_keywords[language]["No_interest"]):
            logger.debug(f"Detected intent: No_interest for text: {text_lower} in language: {language}")
            return {"intent": "No_interest", "sentiment": "negative", "sentiment_score": 0.2}

        if any(keyword in text_lower for keyword in intent_keywords[language]["Moderate_interest"]):
            logger.debug(f"Detected intent: Moderate_interest for text: {text_lower} in language: {language}")
            return {"intent": "Moderate_interest", "sentiment": "neutral", "sentiment_score": 0.5}

        for intent, keywords in intent_keywords[language].items():
            if intent not in ["Confirmation", "company_query", "Strong_interest", "No_interest", "Moderate_interest"]:
                if any(keyword in text_lower for keyword in keywords):
                    logger.debug(f"Detected intent: {intent} for text: {text_lower} in language: {language}")
                    return {"intent": intent, "sentiment": "neutral", "sentiment_score": 0.5}

        logger.debug(f"No specific intent detected for text: {text_lower} in language: {language}. Returning Neutral_response.")
        return {"intent": "Neutral_response", "sentiment": "neutral", "sentiment_score": 0.5}

    except Exception as e:
        logger.error(f"Error in detect_intent for language {language}: {str(e)}", exc_info=True)
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
    positive_labels = ["Strong_interest", "Moderate_interest", "Confirmation"]
    positive_count = sum(1 for s in segment_results if s['intent'] in positive_labels)
    total = len(segment_results)
    return int((positive_count / total) * 100) if total > 0 else 0

def calculate_lead_score(intent_score, sentiment, trigger_words, segment_results, language='en'):
    sentiment_weight = {
        "POSITIVE": 30,
        "NEGATIVE": -30,
        "NEUTRAL": 10
    }

    trigger_weight = 10 * len(trigger_words)

    sentiment_component = sentiment_weight.get(sentiment.upper(), 0)

    # Calculate extra points based on keyword matches in the last 5 sentences
    positive_keywords = {
        'en': ["share", "interested", "send whatsapp", "don't have any other", "got it", "acknowledge", "noted", "please send", "sent details", "agreed"],
        'ml': ["പങ്കിടുക", "താൽപ്പര്യം", "ശരി", "താല്പര്യമുണ്ട്", "തിരയുന്നു", "ഇഷ്ടമുണ്ട്", "വാട്സാപ്പിൽ അയക്കൂ", "വാട്സാപ്പ്", "വാട്ട്സാപ്പ്", "കിട്ടി", "അറിയിച്ചു", "നോട്ടു ചെയ്തു", "സമ്മതം", "അംഗീകരിച്ചു", "ഓക്കെ", "യെസ്", "അക്ക്നലഡ്ജ്", "ക്ലിയർ", "തയാറാണ്", "അറിയിപ്പ് ലഭിച്ചു", "വാട്ട്സാപ്പിലേ", "ഞാൻ അതിനായി നോക്കിയിരുന്നു"],
    }
    negative_keywords = {
        'en': ["not interested", "not looking", "can't", "don't have any other", "won't", "don't like", "not now", "later", "not suitable", "decline"],
        'ml': ["താല്പര്യമില്ല", "നോക്കുന്നില്ല", "ഇല്ല", "വേണ്ട", "മറ്റ് ജോലികൾ ചെയ്യാനില്ലേ?", "സാധ്യമല്ല", "ഇഷ്ടമല്ല"],
    }
    positive_extra_points = 10
    negative_extra_points = -10

    extra_points = 0
    if segment_results:
        last_five_segments = segment_results[-5:] if len(segment_results) >= 5 else segment_results
        for segment in last_five_segments:
            text = segment.get('text', '').lower()
            if any(keyword in text for keyword in positive_keywords.get(language, [])):
                extra_points += positive_extra_points
            if any(keyword in text for keyword in negative_keywords.get(language, [])):
                extra_points += negative_extra_points

    final_score = intent_score * 0.5 + sentiment_component * 0.3 + trigger_weight * 0.2 + extra_points
    return min(100, max(0, final_score))
