import streamlit as st
from backend import build_youtube, process_comments

st.set_page_config(page_title="AI Comment Moderation", layout="wide")

st.title("🛡 AI-Based Comment Moderation System")

api_key = st.text_input("Enter YouTube API Key", type="password")
url = st.text_input("Enter YouTube Video/Short URL")

if st.button("Analyze Comments"):

    if not api_key or not url:
        st.error("Please enter both API key and URL.")
    else:
        youtube = build_youtube(api_key)

        with st.spinner("Fetching and analyzing comments..."):
            df = process_comments(youtube, url)

        st.success("Analysis Complete!")

        st.subheader("Moderation Results")
        st.dataframe(df)

        st.subheader("Label Distribution")
        st.bar_chart(df["Label"].value_counts())