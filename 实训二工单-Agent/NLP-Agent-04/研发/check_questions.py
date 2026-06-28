# -*- coding: utf-8 -*-
import json
with open('data/raw/bs_challenge_financial_14b_dataset/question.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"总题数: {len(data)}")
