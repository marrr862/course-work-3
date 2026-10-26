"""Word-level explanation by occlusion (leave one word out).

For a headline of n words the deployed model is run on the n variants that each drop one word.
effect(word) = P(answer | full text) - P(answer | text without the word): a positive effect means
the word pushed the model toward the answer it gave. This treats the whole ensemble as a black
box, so it explains the actual model in use rather than one of its parts.
"""
import numpy as np

MAX_WORDS = 60
MIN_EFFECT = 0.005


def occlusion(predict_proba, text, class_index, top=5):
    """predict_proba(list[str]) -> array (n, classes). Returns the most influential words or None."""
    words = text.split()
    if not 2 <= len(words) <= MAX_WORDS:
        return None
    variants = [' '.join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probs = np.asarray(predict_proba([text] + variants))[:, class_index]
    effects = probs[0] - probs[1:]
    order = np.argsort(-np.abs(effects))
    items = [{'word': words[i], 'position': int(i), 'effect': round(float(effects[i]), 4)}
             for i in order if abs(effects[i]) >= MIN_EFFECT][:top]
    return {'method': 'leave-one-word-out occlusion', 'base_score': round(float(probs[0]), 4), 'words': items}
