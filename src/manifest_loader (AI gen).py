#!/usr/bin/env python3
# scripts/fast_build_manifest.py
"""
Fast manifest builder for Galaxy Zoo 2 (Hart 2016) + mapping -> sample.

Saves:
 - data/processed/manifest.csv
 - data/processed/manifest_with_labels.csv
 - data/labels/labels_manifest_1000.csv
 - data/labels/captions.csv
 - data/labels/images/ (copied sampled images)
 - data/labels/thumbs/ (224x224 thumbnails for sampled images)
"""
import os, sys, glob, shutil, math
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# -------- CONFIG --------
MAPPING_CSV = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\mapping.csv"
GZ_CSV = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\gz2_hart16.csv.gz"
IMAGES_DIR = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\images"
OUT_PROC_DIR = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed"
OUT_LABELS_DIR = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\labels"
OUT_MANIFEST = os.path.join(OUT_PROC_DIR, "manifest.csv")
OUT_MANIFEST_LABELS = os.path.join(OUT_PROC_DIR, "manifest_with_labels.csv")
SAMPLE_CSV = os.path.join(OUT_LABELS_DIR, "labels_manifest_1000.csv")
CAPTIONS_CSV = os.path.join(OUT_LABELS_DIR, "captions.csv")
SAMPLED_IMAGES_DIR = os.path.join(OUT_LABELS_DIR, "images")
THUMBS_DIR = os.path.join(OUT_LABELS_DIR, "thumbs")

SAMPLE_SIZE = 1000
LABEL_THRESHOLD = 0.60   # threshold to call a confident label
MAX_VOTE_COLS = 80      # limit how many vote columns to read (saves memory)
# ------------------------

os.makedirs(OUT_PROC_DIR, exist_ok=True)
os.makedirs(OUT_LABELS_DIR, exist_ok=True)
os.makedirs(SAMPLED_IMAGES_DIR, exist_ok=True)
os.makedirs(THUMBS_DIR, exist_ok=True)

def read_mapping(path=MAPPING_CSV):
    print("Reading mapping:", path)
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    # normalize column names
    if 'asset_id' not in df.columns:
        # try common alternatives
        for alt in ['assetid', 'asset id', 'asset-id']:
            if alt in df.columns:
                df = df.rename(columns={alt:'asset_id'})
                break
    if 'objid' not in df.columns:
        for alt in ['dr7objid','dr7_objid','obj_id','id','dr7_id']:
            if alt in df.columns:
                df = df.rename(columns={alt:'objid'})
                break
    if 'asset_id' not in df.columns or 'objid' not in df.columns:
        raise RuntimeError("mapping.csv must have columns 'objid' and 'asset_id' (found: {}).".format(df.columns.tolist()))
    df['asset_id'] = df['asset_id'].astype(str)
    df['objid'] = df['objid'].astype(str)
    print("Mapping rows:", len(df))
    return df

def inspect_gz_header(path=GZ_CSV):
    print("Inspecting gz CSV header:", path)
    head = pd.read_csv(path, compression='gzip', nrows=0)
    cols = list(head.columns)
    print("Total columns in gz CSV:", len(cols))
    # print first ~120 columns for quick inspection
    for i,c in enumerate(cols[:120], start=1):
        print(f"{i:03d}: {c}")
    return cols

def choose_vote_columns(all_cols, limit=MAX_VOTE_COLS):
    # prefer 'debiased' columns (Hart provides debiased columns), then weighted_fraction, then fraction
    candidates = []
    prioritized_suffixes = ['debiased', 'weighted_fraction', 'fraction', 'count', 'weight']
    for suf in prioritized_suffixes:
        for c in all_cols:
            if suf in c.lower() and c not in candidates:
                candidates.append(c)
    # also include gz2_class and key id columns
    extras = ['dr7objid', 'gz2_class', 'total_classifications', 'total_votes']
    # keep order and uniqueness, limit length
    chosen = [c for c in extras + candidates if c in all_cols]
    if len(chosen) > limit:
        chosen = chosen[:limit]
    print("Selected {} columns to read from gz CSV.".format(len(chosen)))
    return chosen

def read_gz_selected(path=GZ_CSV, cols_to_read=None):
    if cols_to_read is None:
        header_cols = inspect_gz_header(path)
        cols_to_read = choose_vote_columns(header_cols)
    print("Reading gz CSV (this may take a minute)...")
    df = pd.read_csv(path, compression='gzip', usecols=cols_to_read, low_memory=False)
    print("Rows in gz table:", len(df))
    # standardize id column name to dr7objid
    if 'dr7objid' not in df.columns:
        id_col = [c for c in df.columns if 'dr7' in c.lower() or 'objid' in c.lower()]
        if id_col:
            df = df.rename(columns={id_col[0]:'dr7objid'})
    df['dr7objid'] = df['dr7objid'].astype(str)
    return df

def build_image_map(images_dir=IMAGES_DIR):
    print("Building filename map (scanning images directory once):", images_dir)
    file_map = {}
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            name, ext = os.path.splitext(f)
            # simplify name by removing possible leading zeros (but keep original key too)
            file_map[name] = os.path.join(root, f)
    print("Total unique basenames found in images folder:", len(file_map))
    return file_map

def map_filepaths_vectorized(df_merge, file_map):
    # try mapping by asset_id first
    print("Mapping filepaths using exact asset_id -> filename match (fast)...")
    df_merge['asset_key'] = df_merge['asset_id'].astype(str)
    df_merge['filepath'] = df_merge['asset_key'].map(file_map).fillna("")
    missing_after_asset = df_merge['filepath'].eq("").sum()
    print("Missing paths after asset_id exact match:", missing_after_asset, "of", len(df_merge))
    # try mapping by objid if still missing
    if missing_after_asset > 0:
        print("Trying to map by objid for missing entries...")
        df_merge['objid_key'] = df_merge['objid'].astype(str)
        # for speed, only operate on missing rows
        mask = df_merge['filepath'] == ""
        df_merge.loc[mask, 'filepath'] = df_merge.loc[mask, 'objid_key'].map(file_map).fillna("")
        missing_after_obj = df_merge['filepath'].eq("").sum()
        print("Missing after objid mapping:", missing_after_obj)
    # final: try partial contains match for small remainder (slower, but only few rows)
    missing_final = df_merge['filepath'].eq("").sum()
    if missing_final > 0 and missing_final < 20000:
        print("Attempting partial match for remaining {} rows (this may take some seconds)...".format(missing_final))
        # build a list of keys found
        keys = list(file_map.keys())
        for i,row in df_merge.loc[df_merge['filepath']=="", ['asset_key','objid_key']].iterrows():
            a = row['asset_key']
            o = row.get('objid_key','')
            found = None
            for k in (a,o):
                if not k: continue
                if k in file_map:
                    found = file_map[k]; break
            if found:
                df_merge.at[i,'filepath'] = found
        missing_final = df_merge['filepath'].eq("").sum()
        print("Missing after partial attempt:", missing_final)
    return df_merge

def detect_label_columns(columns):
    # heuristics to pick best column for each question using suffix preference: debiased > weighted_fraction > fraction
    columns_lower = [c.lower() for c in columns]
    def pick(keywords):
        # return first column that contains all keywords and contains a preferred suffix
        for pref in ['debiased', 'weighted_fraction', 'fraction', 'count', 'weight']:
            for c in columns:
                low = c.lower()
                if all(kw in low for kw in keywords) and pref in low:
                    return c
        # fallback: any column with keywords
        for c in columns:
            low = c.lower()
            if all(kw in low for kw in keywords):
                return c
        return None
    spiral = pick(['spiral'])  # e.g. t04_spiral_a08_spiral_debiased
    smooth = pick(['smooth','features']) or pick(['smooth'])  # t01_smooth_or_features_*_debiased
    edge = pick(['edgeon','edge-on','edgeon']) or pick(['edgeon','yes']) or pick(['edgeon','no'])
    merger = pick(['merger','disturbed','odd']) or pick(['merger'])
    bar = pick(['bar','barred'])
    print("Detected label columns (may be None):")
    print(" spiral:", spiral)
    print(" smooth:", smooth)
    print(" edge-on:", edge)
    print(" merger:", merger)
    print(" bar:", bar)
    return {'spiral':spiral, 'smooth':smooth, 'edge':edge, 'merger':merger, 'bar':bar}

def derive_labels(df, label_cols, thr=LABEL_THRESHOLD):
    print("Deriving 'derived_label' using threshold", thr)
    # initialize scores to 0 if column missing
    for k,col in label_cols.items():
        if col and col in df.columns:
            df[f"score_{k}"] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[f"score_{k}"] = 0.0
    # priority order: spiral > smooth(elliptical) > edge > merger > barred
    conds = [
        df['score_spiral'] >= thr,
        df['score_smooth'] >= thr,
        df['score_edge'] >= thr,
        df['score_merger'] >= thr,
        df['score_bar'] >= thr
    ]
    choices = ['spiral','elliptical','edge-on','merger','barred']
    df['derived_label'] = np.select(conds, choices, default='ambiguous')
    # fallback: if gz2_class column exists and derived ambiguous, map some common strings
    if 'gz2_class' in df.columns:
        mask = df['derived_label'] == 'ambiguous'
        df.loc[mask, 'derived_label'] = df.loc[mask,'gz2_class'].astype(str).str.lower().map(
            lambda s: 'spiral' if 'spir' in s else ('elliptical' if ('ellip' in s or 'smooth' in s) else ('edge-on' if 'edge' in s else 'ambiguous'))
        )
    print("Label distribution (top):")
    print(df['derived_label'].value_counts().head(20))
    return df

def create_simple_captions(df):
    def caption_of(r):
        lab = r['derived_label']
        if lab == 'spiral':
            return "A spiral galaxy with visible arms."
        if lab == 'elliptical':
            return "An elliptical galaxy with a smooth light profile."
        if lab == 'edge-on':
            return "A galaxy seen edge-on; disk oriented side-on."
        if lab == 'merger':
            return "A galaxy showing disturbed or merger-like features."
        if lab == 'barred':
            return "A barred spiral galaxy with a central bar."
        return "Galaxy image."
    df['caption'] = df.apply(caption_of, axis=1)
    return df

def sample_and_write(df, sample_size=SAMPLE_SIZE):
    # filter confident labels
    conf = df[df['derived_label'] != 'ambiguous'].copy()
    if conf.empty:
        print("No confident labels available. Consider lowering threshold.")
        return conf
    top_classes = conf['derived_label'].value_counts().index.tolist()
    n_classes = min(len(top_classes), 6)
    chosen = top_classes[:n_classes]
    per_class = max(10, sample_size // max(1,n_classes))
    print("Sampling {} images, per class {} for classes: {}".format(sample_size, per_class, chosen))
    samples = []
    for c in chosen:
        sub = conf[conf['derived_label']==c]
        if len(sub) >= per_class:
            samples.append(sub.sample(n=per_class, random_state=42))
        else:
            samples.append(sub)
    sample_df = pd.concat(samples).reset_index(drop=True)
    # if fewer than requested, add more from conf
    if len(sample_df) < sample_size:
        need = sample_size - len(sample_df)
        pool = conf.loc[~conf.index.isin(sample_df.index)]
        if len(pool) > 0:
            sample_df = pd.concat([sample_df, pool.sample(n=min(need,len(pool)), random_state=42)])
    sample_df = sample_df.reset_index(drop=True)
    sample_df.to_csv(SAMPLE_CSV, index=False)
    print("Sample saved to", SAMPLE_CSV, "size:", len(sample_df))
    return sample_df

def copy_and_make_thumbs(sample_df):
    copied = 0
    thumb_err = 0
    for _, r in sample_df.iterrows():
        src = r.get('filepath','')
        if not src or not os.path.exists(src):
            continue
        fname = os.path.basename(src)
        dst = os.path.join(SAMPLED_IMAGES_DIR, fname)
        if not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as e:
                print("Copy error:", e, src)
                continue
        # thumbnail
        try:
            im = Image.open(src).convert("RGB")
            im.thumbnail((224,224))
            im.save(os.path.join(THUMBS_DIR, fname), quality=85)
        except Exception as e:
            thumb_err += 1
    print("Copied images:", copied, "thumbnail errors:", thumb_err)

def main():
    print("=== FAST BUILD MANIFEST START ===")
    mapping = read_mapping()
    gz_header_cols = inspect_gz_header(GZ_CSV)
    cols_to_read = choose_vote_columns(gz_header_cols)
    # ensure we at least read the id and class columns
    for must in ['dr7objid','gz2_class','total_classifications']:
        if must not in cols_to_read:
            cols_to_read.insert(0, must)
    gz_df = read_gz_selected(GZ_CSV, cols_to_read)
    # merge mapping (mapping.objid) with gz (dr7objid)
    print("Merging mapping and gz table (left merge) ...")
    merged = mapping.merge(gz_df, left_on='objid', right_on='dr7objid', how='left', suffixes=('','_gz'))
    print("Merged rows:", len(merged), "NaNs in gz key:", merged['dr7objid'].isna().sum())
    # Keep only rows that have gz labels (drop mapping rows without gz entry)
    merged = merged[~merged['dr7objid'].isna()].copy()
    print("Rows after keeping only label-matched entries:", len(merged))
    # Build filename map and map filepaths
    file_map = build_image_map(IMAGES_DIR)
    merged = map_filepaths_vectorized(merged, file_map)
    # Save intermediate manifest
    merged.to_csv(OUT_MANIFEST, index=False)
    print("Saved manifest (with paths) to", OUT_MANIFEST)
    # Derive labels
    label_cols = detect_label_columns(gz_header_cols)
    merged = derive_labels(merged, label_cols, thr=LABEL_THRESHOLD)
    merged.to_csv(OUT_MANIFEST_LABELS, index=False)
    print("Saved manifest with derived labels to", OUT_MANIFEST_LABELS)
    # Filter to rows that actually have a file
    have_file = merged['filepath'].astype(bool)
    print("Rows with non-empty filepath:", have_file.sum(), "of", len(merged))
    merged = merged[have_file].copy()
    # Sample and create thumbnails
    sample_df = sample_and_write(merged, SAMPLE_SIZE)
    if sample_df is not None and not sample_df.empty:
        sample_df = create_simple_captions(sample_df)
        sample_df[['asset_id','objid','filepath','derived_label','caption']].to_csv(CAPTIONS_CSV, index=False)
        print("Wrote captions to", CAPTIONS_CSV)
        copy_and_make_thumbs(sample_df)
    print("=== DONE ===\nOutputs:\n -", OUT_MANIFEST, "\n -", OUT_MANIFEST_LABELS, "\n -", SAMPLE_CSV, "\n -", CAPTIONS_CSV)
    print("Inspect data/processed/ and data/labels/")

if __name__ == "__main__":
    main()
