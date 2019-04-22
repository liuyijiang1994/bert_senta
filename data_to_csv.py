import json
import pandas as pd
from collections import Counter

confidence_gate = 0.7
result_cont = 0
sentiment = []
review = []
with open('data/summary.txt', 'r') as f:
    for line in f:
        line = line.strip()
        ob = json.loads(line)
        for su in ob['summary']:
            if su['confidence'] > confidence_gate:
                sentiment.append(su['sentiment'])
                review.append(su['content'].replace(',', '，'))

save = pd.DataFrame({'label': sentiment, 'review': review})
save.to_csv('data/summary.csv', index=False, sep=',')
count = Counter(sentiment)
print(count)
