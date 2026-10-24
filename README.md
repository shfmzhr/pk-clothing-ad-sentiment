# Sentiment Analysis of Pakistani Clothing Ads (YouTube Comments)

An NLP pipeline that collects public YouTube comments on official Pakistani clothing-brand channels, classifies each comment as **positive / neutral / negative** using TF-IDF + Logistic Regression, and shows the results in a Streamlit dashboard.

Brands: **Khaadi**, **Gul Ahmed**, **Sana Safinaz** (a small third brand).

## Results

| Metric (5-fold cross-validation, 570 hand-labeled comments) | Majority-class baseline | Model |
|---|---|---|
| Accuracy | 58.2% | **76.8%** |
| Macro-F1 | 0.245 | **0.572** |

| Class | F1 | Labeled examples |
|---|---|---|
| positive | 0.833 | 332 |
| neutral | 0.723 | 215 |
| negative | 0.162 | 23 |

| Language (heuristic guess) | Accuracy |
|---|---|
| English (n=408) | 79.9% |
| Roman Urdu (n=134) | 65.7% |

Key findings:
- Comments focus on the ad as entertainment (song, actress) far more than on the clothes, price, quality, or delivery.
- Brand channels attract mostly praise and questions. Only about 4% of labeled comments were negative, so the negative class is poorly learned.
- Roman Urdu is harder than English (no standard spelling, code-mixing).
- Most model errors confuse positive with neutral (praise mixed with a question, slang, sarcasm).

Screenshots: `reports/dash_brand.png`, `reports/dash_language.png`, `reports/dash_topics.png`. Confusion matrix: `reports/confusion_matrix.png`. Full metrics: `reports/metrics.txt`.

## Pipeline

1. `src/collect.py`: collects top-level public comments with the YouTube Data API v3 (official brand channels only; commenter names are not stored).
2. `src/clean.py`: removes HTML, links, spam, and duplicates; guesses language; draws a 600-comment sample for hand labeling (all Roman Urdu and Urdu-script comments are included).
3. Manual labeling of 570 comments into positive, neutral, and negative.
4. `src/train.py`: TF-IDF (word and character n-grams) + Logistic Regression with balanced class weights, evaluated with stratified 5-fold cross-validation.
5. `app/dashboard.py`: Streamlit dashboard (sentiment by brand and language, topic keywords, comment browser).

## Run it yourself

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your YouTube Data API key in .env
python src/collect.py --brand Khaadi --videos 2 --comments 1000
python src/collect.py --brand "Gul Ahmed" --videos 4 --comments 500
python src/collect.py --brand "Sana Safinaz" --videos 1 --comments 200
python src/clean.py         # writes data/labeled/to_label.csv
# copy to_label.csv to labeled.csv and fill the sentiment column
python src/train.py
streamlit run app/dashboard.py
```

Comment text is not included in this repository. Collect it with your own API key.

## Limitations

- YouTube only. Instagram and Facebook comments are restricted.
- Comments are not guaranteed to come from Pakistan.
- Khaadi data comes from one music campaign (two videos), so brand differences partly reflect campaign differences.
- Sentiment is expressed opinion, not purchase intent. Public comments do not represent all customers.
- Language labels and topic counts come from simple word lists, not trained models.

## Ethics

Only public comments from official brand channels were used. Usernames were not stored. API keys are kept in `.env`, which is not committed.