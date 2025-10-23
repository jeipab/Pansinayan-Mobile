"""
Label mapping utilities for sign language recognition.

This module provides functions to load and manage label mappings
for glosses and categories.
"""

import json
import os
import pandas as pd


def load_label_mappings(csv_path=None):
    """
    Load label mappings from CSV file or create default mappings.
    
    Args:
        csv_path: Path to CSV file with label mappings
        
    Returns:
        tuple: (gloss_mapping, category_mapping)
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Create gloss mapping: {id: label}
        gloss_mapping = {int(row['gloss_id']): str(row['label']) 
                        for _, row in df.iterrows()}
        
        # Create category mapping: {id: category}
        category_mapping = {}
        for _, row in df.iterrows():
            cat_id = int(row['cat_id'])
            if cat_id not in category_mapping:
                category_mapping[cat_id] = str(row['category'])
    else:
        # Create default mappings
        gloss_mapping = {i: f"sign_{i:03d}" for i in range(105)}
        category_mapping = {i: f"category_{i}" for i in range(10)}
    
    return gloss_mapping, category_mapping
