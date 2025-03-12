import neattext.functions as nfx

def clean_text(text):
    """
    Cleans input text by:
    - Converting to lowercase
    - Removing punctuation
    - Removing stopwords
    - Stripping extra spaces
    """
    if not isinstance(text, str):  # Ensure input is a string
        return ""

    text = text.lower().strip()  # Lowercase and remove extra spaces
    text = nfx.remove_puncts(text)  # Remove punctuation
    text = nfx.remove_stopwords(text)  # Remove stopwords

    return text
