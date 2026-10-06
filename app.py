import streamlit as st
from pypdf import PdfReader
from docx import Document
from transformers import pipeline
import re

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="NoteGenie - AI Notes Summarization",
    page_icon="📝",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #4a4e69;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.box {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    '<div class="title">📝 NoteGenie</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI Notes Summarization System</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload your notes and NoteGenie will generate a short and meaningful summary."
)

# ---------------- TEXT EXTRACTION ----------------
def extract_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_docx(file):
    doc = Document(file)

    text = ""

    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"

    return text


def extract_txt(file):
    return file.read().decode("utf-8")


# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------- LOAD AI MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )


# ---------------- FILE UPLOAD ----------------
st.markdown("### 📂 Upload Your Notes")

uploaded_file = st.file_uploader(
    "Choose a PDF, DOCX or TXT file",
    type=["pdf", "docx", "txt"]
)

if uploaded_file:

    st.success(f"File uploaded: {uploaded_file.name}")

    # Extract text
    if uploaded_file.name.lower().endswith(".pdf"):
        text = extract_pdf(uploaded_file)

    elif uploaded_file.name.lower().endswith(".docx"):
        text = extract_docx(uploaded_file)

    else:
        text = extract_txt(uploaded_file)

    text = clean_text(text)

    if not text:
        st.error("❌ No readable text found in this file.")

    else:

        # ---------------- STATISTICS ----------------
        words = len(text.split())
        characters = len(text)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📖 Words", words)

        with col2:
            st.metric("🔤 Characters", characters)

        with col3:
            st.metric("📄 File Type", uploaded_file.name.split(".")[-1].upper())

        # ---------------- ORIGINAL NOTES ----------------
        with st.expander("👀 View Original Notes"):
            st.write(text)

        st.markdown("---")

        # ---------------- SUMMARY BUTTON ----------------
        if st.button("✨ Generate AI Summary", use_container_width=True):

            with st.spinner("🤖 AI is generating your summary..."):

                try:

                    summarizer = load_model()

                    # Limit very large input
                    input_text = text[:12000]

                    result = summarizer(
                        input_text,
                        max_length=180,
                        min_length=50,
                        do_sample=False
                    )

                    summary = result[0]["summary_text"]

                    # ---------------- SUMMARY ----------------
                    st.markdown("## 📌 AI Generated Summary")

                    st.success(summary)

                    # ---------------- IMPORTANT POINTS ----------------
                    st.markdown("## ⭐ Important Points")

                    sentences = re.split(r'(?<=[.!?])\s+', summary)

                    for sentence in sentences:
                        if sentence.strip():
                            st.write("• " + sentence.strip())

                    # ---------------- DOWNLOAD ----------------
                    st.markdown("---")

                    st.download_button(
                        label="⬇️ Download Summary",
                        data=summary,
                        file_name="NoteGenie_Summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                except Exception as e:

                    st.error("❌ Something went wrong while generating the summary.")

                    st.info(
                        "Make sure the required AI libraries and model are installed."
                    )

else:

    st.info("👆 Upload a PDF, DOCX or TXT file to get started.")

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
    "<center>📝 NoteGenie | AI Notes Summarization System</center>",
    unsafe_allow_html=True
)