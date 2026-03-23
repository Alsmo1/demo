import re

def clean_text(text):
    # Remove weird unicode characters
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove lines with only symbols
    text = re.sub(r"[^\w\s.,?!\-–—/]+", "", text)

    # Remove extra newlines
    text = re.sub(r"\n+", "\n", text)

    return text.strip()