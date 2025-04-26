import streamlit as st

st.set_page_config(
    page_title="GEMINISCRIBE",
    page_icon="🌟",
    layout="centered"
)

st.title("GEMINISCRIBE")
st.subheader("Choose what you'd like to do:")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/1_Chat_with_PDFs.py", label="📘 Chat with PDFs", icon="📄")

with col2:
    st.page_link("pages/2_Summarize_YouTube.py", label="🎥 Summarize YouTube Video", icon="🎬")

st.markdown("---")
st.caption("Built with 💙 Streamlit + Gemini AI")

