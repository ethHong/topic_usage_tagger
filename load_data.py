import json
import pandas as pd

with open("category_map.json", "r") as f:
    category = json.load(f)

candidate_labels = [terms for lists in list(category.values()) for terms in lists]
source = category["SOURCE"]
