from pathlib import Path
import pandas as pd
import json
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "labels" / "captions"

special_tokens = {"<bos>", "<eos>", "<pad>", "<unk>"}

#vocab = ["<pad>", "<bos>", "<eos>", "<unk>", "with", "a", "of", "arm",
#         "arms", "galaxy", "spiral", "irregular", "tightly",
#         "moderately", "wound", "loosely", "1", "2", "3", "4",
#         "single", "pair", "three", "four", "more", "than", "5", "many",
#         "elliptical", "interacting/merging", "in", "merger", "edge-on",
#         "seen", "barred", "central", "bar", "rounded/cigar-shaped", "ring",
#         "structure", "dust", "lane", "lanes", "disturbed", "morphology",
#         "signs", "disturbance", "lens/arc", "feature", "small", "non-existent",
#         "faint", "not", "prominent", "rounded", "round", "bulge", "boxy", "rectangular/boxy",
#         "modest", "noticeable", "conspicuous", "dominant", "very", "large"
#]

#Path struggles
all_captions_path = DATA_PATH / "captions_sample.csv"
caption_train_path = DATA_PATH / "train_captions_sample.txt"
df = pd.read_csv(all_captions_path, dtype={"objid": str})
train_ids = pd.read_csv(
    caption_train_path,
    header=None,
    names=["objid"],
    dtype=str,
    sep=r"\s+",
    engine="python",
)
df_train = df.merge(train_ids, how="inner", on="objid").reset_index(drop=True)

captions = df_train["caption"].astype(str).tolist()

#Token inference
tokens = []
for caption in captions:
    caption = caption.lower().replace(",", "")
    tokens.extend([d for d in caption.split(" ")])
freq = [token for token, _ in Counter(tokens).most_common()]

inferred_vocab = [g for g in freq]
inferred_vocab = ["<pad>", "<bos>", "<eos>", "<unk>"] + inferred_vocab
vocab_path = DATA_PATH / "vocab.json"
json.dump(inferred_vocab, open(vocab_path, "w"))

config = {"pad_id" : 0, "bos_id" : 1, "eos_id" : 2, "unk_id" : 3, "max_len" : 24}
config_path = DATA_PATH / "config.json"
json.dump(config, open(config_path, "w"))

id2token = inferred_vocab
token2id = {token: inferred_vocab.index(token) for token in inferred_vocab}

#Token encoder/decoder
def encode(text):
    text = text.lower().replace(",", "")
    ids_list = [token2id.get(token, 3) for token in text.split()][:config["max_len"] - 2]

    cap_in = [1] + ids_list
    cap_out = ids_list + [2]
    cap_in = cap_in + [0] * (config["max_len"] - len(cap_in))
    cap_out = cap_out + [0] * (config["max_len"] - len(cap_out))

    pad_mask = [1] * (len(ids_list) + 1) + [0] * (config["max_len"] - len(ids_list) - 1)

    return cap_in, cap_out, pad_mask

def decode(id_list):
    tokens = []
    for i in id_list:
        token = id2token[i] if 0 <= i < len(id2token) else "<unk>"
        if token == "<eos>":
            break
        if token in ["<pad>", "<bos>", "<eos>"]:
            continue
        tokens.append(token)
    caption_out = " ".join(tokens)
    return caption_out
