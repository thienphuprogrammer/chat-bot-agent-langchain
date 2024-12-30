import spacy
from pyvi import ViTokenizer

# Load a SpaCy model (e.g., English model for pipeline)
nlp = spacy.blank("en")


# Custom tokenizer using Pyvi
def vietnamese_tokenizer(text):
    tokens = ViTokenizer.tokenize(text).split()
    return spacy.tokens.Doc(nlp.vocab, words=tokens)


# Add the custom tokenizer to the pipeline
nlp.tokenizer = vietnamese_tokenizer

# Process Vietnamese text
doc = nlp("Tôi là người Việt Nam.")
print([token.text for token in doc])  # Output: ['Tôi', 'là', 'người', 'Việt_Nam.']
