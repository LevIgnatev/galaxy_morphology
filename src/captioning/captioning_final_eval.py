import pandas as pd
from pathlib import Path
import tensorflow as tf
import os

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from caption_inference import generate_caption

PROJECT_ROOT = Path(__file__).parents[2]
captions_all = pd.read_csv(str(PROJECT_ROOT / "data" / "processed" / "captions_full" / "captions_full.csv"), dtype={"objid": str})
test_captions_fp = PROJECT_ROOT / "data" / "processed" / "captions_full" / "test_captions.txt"

test_captions_ids = pd.read_csv(
    test_captions_fp,
    header=None,
    names=["objid"],
    dtype=str,
    sep=r"\s+",
    engine="python",
)
df_test = captions_all.merge(test_captions_ids, on=["objid"], how="inner")
test_paths_list = df_test["filepath"].astype(str).apply(lambda x: str(PROJECT_ROOT / x)).tolist()
test_captions_list = df_test["caption"].astype(str).tolist()

predicted_captions_list = [generate_caption(image_fp) for image_fp in test_paths_list]

predicted_in_tokens = [caption.split(" ") for caption in predicted_captions_list]
reference_in_tokens = [[caption.split(" ")] for caption in test_captions_list]
smoothing_function = SmoothingFunction().method3

bleu1 = corpus_bleu(reference_in_tokens, predicted_in_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
bleu2 = corpus_bleu(reference_in_tokens, predicted_in_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
bleu3 = corpus_bleu(reference_in_tokens, predicted_in_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing_function)
bleu4 = corpus_bleu(reference_in_tokens, predicted_in_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)

print(f"BLEU-1 score: {bleu1}")
print(f"BLEU-2 score: {bleu2}")
print(f"BLEU-3 score: {bleu3}")
print(f"BLEU-4 score: {bleu4}")
