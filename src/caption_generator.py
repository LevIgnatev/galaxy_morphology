import pandas as pd

manifest_fp = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\manifest_train_and_val.csv"

df = pd.read_csv(manifest_fp)

features_column_names = {
    # General features
    "features": "t01_smooth_or_features_a02_features_or_disk_debiased",
    "smooth": "t01_smooth_or_features_a01_smooth_debiased",
    "spiral": "t04_spiral_a08_spiral_debiased",
    "no_spiral": "t04_spiral_a09_no_spiral_debiased",
    "edge_on": "t02_edgeon_a04_yes_debiased",
    "bar": "t03_bar_a06_bar_debiased",
    "rounded/cigar-shaped": "t07_rounded_a18_cigar_shaped_debiased",

    # Bulge features
    "no_bulge": "t05_bulge_prominence_a10_no_bulge_debiased",
    "bulge_just_noticeable": "t05_bulge_prominence_a11_just_noticeable_debiased",
    "bulge_obvious": "t05_bulge_prominence_a12_obvious_debiased",
    "bulge_dominant": "t05_bulge_prominence_a13_dominant_debiased",
    "bulge_rounded": "t09_bulge_shape_a25_rounded_debiased",
    "bulge_boxy": "t09_bulge_shape_a26_boxy_debiased",

    # Odd features
    "odd": "t06_odd_a14_yes_debiased",
    "ring": "t08_odd_feature_a19_ring_debiased",
    "lens_or_arc": "t08_odd_feature_a20_lens_or_arc_debiased",
    "disturbed": "t08_odd_feature_a21_disturbed_debiased",
    "irregular": "t08_odd_feature_a22_irregular_debiased",
    "other_odd": "t08_odd_feature_a23_other_debiased",
    "merger": "t08_odd_feature_a24_merger_debiased",
    "dust_lane": "t08_odd_feature_a38_dust_lane_debiased",

    # Arms winding features
    "winding_tight": "t10_arms_winding_a28_tight_debiased",
    "winding_medium": "t10_arms_winding_a29_medium_debiased",
    "winding_loose": "t10_arms_winding_a30_loose_debiased",

    # Number of arms
    "n_1": "t11_arms_number_a31_1_debiased",
    "n_2": "t11_arms_number_a32_2_debiased",
    "n_3": "t11_arms_number_a33_3_debiased",
    "n_4": "t11_arms_number_a34_4_debiased",
    "n_5+": "t11_arms_number_a36_more_than_4_debiased",
    "n_undefined": "t11_arms_number_a37_cant_tell_debiased",
}

def get_feature_value(row, feature_name):
    return row[feature_name]

captions = pd.DataFrame({"objid": [], "caption": []})

for i in range (df.shape[0]):
    caption = "galaxy"
    if df.iloc[i][features_column_names["irregular"]] >= 0.6:
        caption = "irregular " + caption
    elif df.iloc[i][features_column_names["spiral"]] >= 0.6 or df.iloc[i][features_column_names["features"]] >= df.iloc[i][features_column_names["smooth"]]:
        caption = "spiral " + caption
        if df.iloc[i][features_column_names["winding_tight"]] >= 0.6:
            caption += [" with tightly wound arms,", " with tightly wound spiral arms,"][i % 2]
        elif df.iloc[i][features_column_names["winding_medium"]] >= 0.6:
            caption += [" with moderately wound arms,", " with moderately wound spiral arms,"][i % 2]
        elif df.iloc[i][features_column_names["winding_loose"]] >= 0.6:
            caption += [" with loosely wound arms,", " with loosely wound spiral arms,"][i % 2]
        if df.iloc[i][features_column_names["n_1"]] >= 0.6:
            caption += [" with 1 spiral arm,", " with a single spiral arm,"][i % 2]
        elif df.iloc[i][features_column_names["n_2"]] >= 0.6:
            caption += [" with 2 spiral arms,", " with a pair of spiral arms,"][i % 2]
        elif df.iloc[i][features_column_names["n_3"]] >= 0.6:
            caption += [" with 3 spiral arms,", " with three spiral arms,"][i % 2]
        elif df.iloc[i][features_column_names["n_4"]] >= 0.6:
            caption += [" with 4 spiral arms,", " with four spiral arms,"][i % 2]
        elif df.iloc[i][features_column_names["n_5+"]] >= 0.6:
            caption += [" with more than 5 spiral arms,", " with many spiral arms,"][i % 2]
    elif df.iloc[i][features_column_names["smooth"]] >= 0.6 and df.iloc[i][features_column_names["smooth"]] >= df.iloc[i][features_column_names["features"]]:
        caption = "elliptical " + caption

    if df.iloc[i][features_column_names["merger"]] >= 0.6:
        if i % 2 == 0: caption = "interacting/merging " + caption
        else: caption += " in a merger,"
    if df.iloc[i][features_column_names["edge_on"]] >= 0.6:
        if i % 2 == 0: caption = "edge-on " + caption
        else: caption += " seen edge-on,"
    if df.iloc[i][features_column_names["bar"]] >= 0.6:
        if i % 2 == 0: caption = "barred " + caption
        else: caption += " with a central bar,"
    if df.iloc[i][features_column_names["rounded/cigar-shaped"]] >= 0.6:
        caption = "rounded/cigar-shaped " + caption

    if df.iloc[i][features_column_names["ring"]] >= 0.6:
        caption += [" with a ring,", " with a ring structure,"][i % 2]
    if df.iloc[i][features_column_names["dust_lane"]] >= 0.6:
        caption += [" with a dust lane,", " with dust lanes,"][i % 2]
    if df.iloc[i][features_column_names["disturbed"]] >= 0.6 and "interacting/merging" not in caption:
        caption += [" with disturbed morphology,", " with signs of disturbance,"][i % 2]
    if df.iloc[i][features_column_names["lens_or_arc"]] >= 0.6 and "elliptical" in caption:
        caption += " with a lens/arc feature,"

    if df.iloc[i][features_column_names["no_bulge"]] >= 0.6:
        caption += [" with a small or non-existent", " with a faint or not prominent"][i % 2]
        if df.iloc[i][features_column_names["bulge_rounded"]] >= 0.6:
            caption += [" rounded bulge,", " round bulge,"][i % 2]
        elif df.iloc[i][features_column_names["bulge_boxy"]] >= 0.6:
            caption += [" boxy bulge,", " rectangular/boxy bulge,"][i % 2]
        else:
            caption += " bulge,"
    elif df.iloc[i][features_column_names["bulge_just_noticeable"]] >= 0.6:
        caption += [" with a modest", " with a noticeable"][i % 2]
        if df.iloc[i][features_column_names["bulge_rounded"]] >= 0.6:
            caption += [" rounded bulge,", " round bulge,"][i % 2]
        elif df.iloc[i][features_column_names["bulge_boxy"]] >= 0.6:
            caption += [" boxy bulge,", " rectangular/boxy bulge,"][i % 2]
        else:
            caption += " bulge,"
    elif df.iloc[i][features_column_names["bulge_obvious"]] >= 0.6:
        caption += [" with a prominent", " with a conspicuous"][i % 2]
        if df.iloc[i][features_column_names["bulge_rounded"]] >= 0.6:
            caption += [" rounded bulge,", " round bulge,"][i % 2]
        elif df.iloc[i][features_column_names["bulge_boxy"]] >= 0.6:
            caption += [" boxy bulge,", " rectangular/boxy bulge,"][i % 2]
        else:
            caption += " bulge,"
    elif df.iloc[i][features_column_names["bulge_dominant"]] >= 0.6:
        caption += [" with a dominant", " with a very large"][i % 2]
        if df.iloc[i][features_column_names["bulge_rounded"]] >= 0.6:
            caption += [" rounded bulge,", " round bulge,"][i % 2]
        elif df.iloc[i][features_column_names["bulge_boxy"]] >= 0.6:
            caption += [" boxy bulge,", " rectangular/boxy bulge,"][i % 2]
        else:
            caption += " bulge,"

    if caption[-1] == ',': caption = caption[:-1]

    captions.loc[len(captions)] = [df.iloc[i]["objid"], caption]

out = captions

out.to_csv(r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\captions_full.csv", index=False, encoding="utf-8")