import streamlit as st

from triage import LABELS, SAMPLE, Draft, analyze

st.set_page_config(page_title="IssueTriage AI", page_icon="📝")
st.title("IssueTriage AI")
st.write("Turn a developer report into an editable GitHub issue.")
engine = st.selectbox("Analysis mode", ["Local classifier", "Hugging Face"])
if engine == "Hugging Face":
    st.caption("Analyze sends this report to the configured Hugging Face inference provider.")
else:
    st.caption("Runs locally. A trained classifier suggests the label and a template drafts the follow-up.")
if st.button("Load sample report"):
    st.session_state.report = SAMPLE
report = st.text_area("Issue report", key="report", max_chars=4000, height=120)
if st.button("Analyze", type="primary"):
    st.session_state.pop("source", None)
    try:
        with st.spinner("Reading the report…"):
            draft = analyze(report, cloud=engine == "Hugging Face")
        for key, value in vars(draft).items():
            st.session_state[key] = value
        st.session_state.source = (report, engine)
    except ValueError as error:
        st.error(str(error))

if st.session_state.get("source") == (report, engine):
    st.subheader("Review your draft")
    title = st.text_input("Title", key="title")
    label = st.selectbox("Suggested label", LABELS, key="label")
    evidence = st.text_area("Evidence from report", key="evidence")
    question = st.text_input("Follow-up question", key="question")
    markdown = Draft(title, label, evidence, question).markdown()
    st.caption("Use the copy icon in the top-right corner of the Markdown box to paste this into GitHub.")
    st.code(markdown, language="markdown")
    st.download_button("Download issue.md", markdown, "issue.md", "text/markdown")
