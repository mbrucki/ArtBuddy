import logging

logger = logging.getLogger(__name__)

QUESTION_WORDS = (
    "what", "who", "where", "when", "why", "how",
    "is", "are", "am", "was", "were",
    "do", "does", "did",
    "can", "could", "will", "would", "should", "may", "might",
    "tell me", "describe", "explain", "list"
)

def is_direct_question(text: str) -> bool:
    """Check if the text likely starts with a question word or ends with a question mark."""
    if not isinstance(text, str):
        return False # Handle non-string input gracefully
    cleaned_text = text.strip().lower()
    starts_with_question_word = cleaned_text.startswith(QUESTION_WORDS)
    ends_with_question_mark = cleaned_text.endswith('?')

    is_q = starts_with_question_word or ends_with_question_mark
    logger.debug(f"is_direct_question check for '{text[:50]}...': Result: {is_q}")
    return is_q 