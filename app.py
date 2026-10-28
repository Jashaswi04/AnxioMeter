"""AnxioMeter Phase 2 dashboard. Run: streamlit run app.py"""

import html
import importlib.util
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/final_analytics.csv"
spec = importlib.util.spec_from_file_location("fluff_calculator", ROOT / "dev2_pipeline/06_fluff_calculator.py")
fluff_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fluff_module)


@st.cache_data
def load_data(path):
    return pd.read_csv(path, keep_default_na=False)


def playful_translation(text):
    lower = text.lower()
    if "excited to share" in lower or "thrilled to announce" in lower:
        return "Translation (just for fun): I have news and chose the professional announcement voice."
    if "learning journey" in lower:
        return "Translation (just for fun): Some of this was difficult, and I learned along the way."
    if "synergy" in lower or "unlock potential" in lower:
        return "Translation (just for fun): We are collaborating, and the details could use a concrete example."
    return "Translation (just for fun): This sounds like a polished career update. Only the writer knows the full story."


def highlight(text):
    import re
    pattern = re.compile("|".join(re.escape(x) for x in sorted(fluff_module.BUZZWORDS, key=len, reverse=True)), re.I)
    parts, start = [], 0
    for found in pattern.finditer(text):
        parts.append(html.escape(text[start:found.start()]))
        parts.append("<mark>" + html.escape(found.group()) + "</mark>")
        start = found.end()
    parts.append(html.escape(text[start:]))
    return "".join(parts).replace("\n", "<br>")


st.set_page_config(page_title="AnxioMeter", page_icon="📊", layout="wide")
st.title("AnxioMeter")
st.caption("Career anxiety and corporate language across platform and topic")

overview, translator = st.tabs(["Macro divergence", "LinkedIn-to-Reddit realism translator"])

with overview:
    if not DATA.exists():
        st.info("Run the Developer 2 pipeline first. Expected file: data/final_analytics.csv")
    else:
        df = load_data(str(DATA))
        required = {"platform", "topic_id", "topic_name", "cai", "cfs", "is_synthetic"}
        if missing := required - set(df.columns):
            st.error(f"Analytics file is missing: {sorted(missing)}")
            st.stop()
        synthetic = df.is_synthetic.astype(str).str.lower().isin(("true", "1", "yes"))
        if synthetic.any():
            st.warning("Synthetic prototype data: platform comparisons are demonstrations, not research findings.")
        include_outliers = st.checkbox("Include unassigned topic (ID -1)", value=True)
        if not include_outliers:
            df = df[df.topic_id.astype(str) != "-1"]
        if df.empty:
            st.info("No records in this selection.")
        else:
            summary = (df.assign(cai=pd.to_numeric(df.cai), cfs=pd.to_numeric(df.cfs))
                       .groupby(["topic_id", "topic_name", "platform"], as_index=False)
                       .agg(n=("cai", "size"), mean_cai=("cai", "mean"), mean_cfs=("cfs", "mean")))
            summary["topic"] = summary.topic_id.astype(str) + " · " + summary.topic_name
            st.caption(f"{len(df)} records · Scores are illustrative 0–100 indices · Group sizes shown on hover")
            left, right = st.columns(2)
            with left:
                fig = px.bar(summary, x="topic", y="mean_cai", color="platform", barmode="group",
                             hover_data=["n"], title="Mean Career Anxiety Index", range_y=[0, 100])
                st.plotly_chart(fig, width="stretch")
            with right:
                fig = px.bar(summary, x="topic", y="mean_cfs", color="platform", barmode="group",
                             hover_data=["n"], title="Mean Corporate Fluff Score", range_y=[0, 100])
                st.plotly_chart(fig, width="stretch")
            st.subheader("Topic × platform matrix")
            for metric, title in (("mean_cai", "CAI"), ("mean_cfs", "CFS")):
                matrix = summary.pivot(index="topic", columns="platform", values=metric)
                fig = px.imshow(matrix, zmin=0, zmax=100, text_auto=".1f", aspect="auto",
                                color_continuous_scale="Blues", title=title)
                st.plotly_chart(fig, width="stretch")
            st.dataframe(summary.round(2), width="stretch", hide_index=True)

with translator:
    st.write("Paste a post to inspect its buzzwords and get a playful paraphrase.")
    post = st.text_area("Post text", height=160, max_chars=5000)
    if post.strip():
        features = fluff_module.fluff_features(post)
        st.metric("Corporate Fluff Score", f"{features['cfs']:.1f} / 100")
        st.progress(min(features["cfs"] / 100, 1.0))
        st.caption(f"Buzzwords: {features['buzzword_count']} · Announcement phrases: {features['humblebrag_count']} · Type-token ratio: {features['type_token_ratio']:.2f}")
        st.markdown("**Marked buzzwords**")
        st.markdown(highlight(post), unsafe_allow_html=True)
        st.info(playful_translation(post))
        st.caption("This is a rule-based joke, not an inference about the writer's intent. No post is sent to a model by this tab.")
