"""Frozen E4/E5 grouping names used by deck and config generation.

The labels come from ``label_ids.json`` and PLAN.md E4/E5.  Slugs are part of the
run/deck naming contract, so changing one requires a plan/registry migration.
"""

MODALITY_SLUGS = {
    "Clinical": "clinical",
    "Dermatology": "derm",
    "Mammography": "mammo",
    "Microscopy": "micro",
    "Retinography": "retina",
    "Ultrasound": "us",
    "Xray": "xray",
}

CLASSIFICATION_DATASET_SLUGS = {
    "BUS-UCLM": "busuclm",
    "BUSI": "busi",
    "CMMD": "cmmd",
    "bcn20000": "bcn20000",
    "bone_marrow": "bonemarrow",
    "dental": "dental",
    "fundus": "fundus",
    "iugc": "iugc",
    "neojaundice": "neojaundice",
    "retino": "retino",
}
