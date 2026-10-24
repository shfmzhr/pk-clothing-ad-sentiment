from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "predictions.csv"
METRICS = ROOT / "reports" / "metrics.txt"

ORDER = ["positive", "neutral", "negative"]
COLORS = ["#2e8b57", "#9aa0a6", "#d6453d"]
LANG_NAMES = {
    "english": "English",
    "roman_urdu": "Roman Urdu",
    "urdu_script": "Urdu script",
    "emoji_only": "Emoji only",
    "turkish_likely": "Turkish (likely)",
}

# Keyword topics. These are simple word matches, NOT aspect-based sentiment.
TOPICS = {
    "Price": r"price|expensive|overpriced|costly|\bcost\b|\brate\b|\brs\b|mehng|mahng|sasta|sasti|discount|\bsale\b|paisa",
    "Design / clothes": r"dress|\bsuits?\b|lawn|print|design|colou?r|kapra|kapray|kapre|outfit|embroider|dupatta|shalwar|kurta|collection",
    "Quality": r"quality|stitch|material|fabric|fitting|\bwash",
    "Delivery / availability": r"delivery|shipping|courier|parcel|\border\b|branch|\bstore\b|available|\bstock\b|\bsizes?\b",
    "Song / actress / model": r"song|music|soundtrack|voice|singer|lyrics|\bost\b|gana|gaana|awaz|actress|\bmodel|esra",
}

st.set_page_config(page_title="Pakistani Clothing Ads: Sentiment", layout="wide")


def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def show_table(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


@st.cache_data
def load():
    d = pd.read_csv(DATA, encoding="utf-8-sig")
    d["language"] = d["lang_guess"].map(LANG_NAMES).fillna(d["lang_guess"])
    return d


st.title("Audience Sentiment on Pakistani Clothing Ads")
st.caption(
    "YouTube comments on official brand channels, classified with TF-IDF + "
    "Logistic Regression. Sentiment is the audience's expressed opinion, "
    "not purchase intent."
)

if not DATA.exists():
    st.error("data/processed/predictions.csv not found. "
             "Run: python src/clean.py, then python src/train.py")
    st.stop()

df = load()

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
brands = st.sidebar.multiselect("Brand", sorted(df["brand"].unique()),
                                default=sorted(df["brand"].unique()))
langs = st.sidebar.multiselect("Language (heuristic guess)",
                               sorted(df["language"].unique()),
                               default=sorted(df["language"].unique()))
unseen = st.sidebar.checkbox(
    "Exclude comments used to train the model",
    help="Hand-labeled comments were used for training, so their predictions "
         "are slightly optimistic. Tick this to see only unseen comments.")

view = df[df["brand"].isin(brands) & df["language"].isin(langs)]
if unseen:
    view = view[~view["hand_labeled"]]
if view.empty:
    st.warning("No comments match these filters.")
    st.stop()

# ---------- headline numbers ----------
share = view["sentiment_pred"].value_counts(normalize=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Comments", f"{len(view):,}")
c2.metric("Positive", f"{share.get('positive', 0):.0%}")
c3.metric("Neutral", f"{share.get('neutral', 0):.0%}")
c4.metric("Negative", f"{share.get('negative', 0):.0%}")

color = alt.Color("sentiment_pred:N", title="Sentiment",
                  scale=alt.Scale(domain=ORDER, range=COLORS))

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["By brand", "By language", "Topics", "Browse comments", "Model & limits"])

# ---------- tab 1: brand ----------
with tab1:
    st.subheader("Sentiment share per brand")
    show_chart(
        alt.Chart(view).mark_bar().encode(
            x=alt.X("brand:N", title=None),
            y=alt.Y("count():Q", stack="normalize", title="Share",
                    axis=alt.Axis(format="%")),
            color=color,
            tooltip=["brand", "sentiment_pred", alt.Tooltip("count():Q", title="comments")],
        ).properties(height=380)
    )
    table = pd.crosstab(view["brand"], view["sentiment_pred"]).reindex(
        columns=ORDER, fill_value=0).reset_index()
    show_table(table)
    st.caption("Khaadi comments are on two videos of one music campaign, "
               "so brand differences partly reflect campaign differences.")

# ---------- tab 2: language ----------
with tab2:
    st.subheader("Sentiment share per language")
    show_chart(
        alt.Chart(view).mark_bar().encode(
            x=alt.X("language:N", title=None),
            y=alt.Y("count():Q", stack="normalize", title="Share",
                    axis=alt.Axis(format="%")),
            color=color,
            tooltip=["language", "sentiment_pred", alt.Tooltip("count():Q", title="comments")],
        ).properties(height=380)
    )
    st.caption("Language is a word-list heuristic, so some comments are misassigned.")

# ---------- tab 3: topics ----------
with tab3:
    st.subheader("What do people mention?")
    rows = []
    for topic, pattern in TOPICS.items():
        sub = view[view["comment_clean"].str.contains(pattern, case=False, regex=True)]
        for s in ORDER:
            rows.append({"topic": topic, "sentiment_pred": s,
                         "comments": int((sub["sentiment_pred"] == s).sum())})
    tdf = pd.DataFrame(rows)
    show_chart(
        alt.Chart(tdf).mark_bar().encode(
            y=alt.Y("topic:N", sort="-x", title=None),
            x=alt.X("sum(comments):Q", title="Comments mentioning the topic"),
            color=color,
            tooltip=["topic", "sentiment_pred", "comments"],
        ).properties(height=320)
    )
    st.caption("Keyword matching only (not aspect-based sentiment). "
               "A comment can mention several topics. The sentiment shown is for "
               "the whole comment, not for the topic alone.")

# ---------- tab 4: browse ----------
with tab4:
    st.subheader("Browse comments")
    f1, f2, f3 = st.columns([2, 2, 2])
    sent = f1.multiselect("Sentiment", ORDER, default=ORDER)
    query = f2.text_input("Search text")
    min_conf = f3.slider("Minimum model confidence", 0.0, 1.0, 0.0, 0.05)

    b = view[view["sentiment_pred"].isin(sent) & (view["confidence"] >= min_conf)]
    if query:
        b = b[b["comment_clean"].str.contains(query, case=False, regex=False)]
    st.write(f"{len(b):,} comments")
    show_table(
        b.sort_values("like_count", ascending=False)[
            ["brand", "language", "comment_clean", "sentiment_pred",
             "confidence", "like_count"]
        ].head(500)
    )

# ---------- tab 5: model ----------
with tab5:
    st.subheader("Model evaluation (5-fold cross-validation on hand-labeled comments)")
    if METRICS.exists():
        st.code(METRICS.read_text(encoding="utf-8"), language=None)
    st.markdown(
        "**Limitations**\n"
        "- Brand YouTube channels attract mostly praise and questions; only "
        "about 4% of labeled comments were negative, so the negative class is "
        "poorly learned.\n"
        "- Only YouTube was used. Instagram and Facebook comments are restricted.\n"
        "- Comments are not guaranteed to come from Pakistan.\n"
        "- Roman Urdu has no standard spelling, and the model does worse on it.\n"
        "- Public comments do not represent all customers, and sentiment is not "
        "purchase intent."
    )