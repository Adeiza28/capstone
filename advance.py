import os
import streamlit as st
import fitz  # PyMuPDF
import docx
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
import pickle

# ====================================
# Load Environment Variables
# ====================================
load_dotenv()

st.set_page_config(page_title="🇳🇬 AI Legal & Civic Assistant", layout="wide")
st.title("🇳🇬 AI Legal & Civic Assistant for Nigerian Citizens")

st.markdown("""
Upload Nigerian **law documents (PDF or DOCX)** and ask legal or civic questions.  
The AI will analyze context, provide answers, and cite sources.  
You can also find **job opportunities** from Jobberman and NGCareers.
""")

# ====================================
# Sidebar Settings
# ====================================
st.sidebar.header("⚙️ Settings")
api_key = st.sidebar.text_input("🔑 Google API Key", value=os.getenv("GOOGLE_API_KEY"))
uploaded_files = st.sidebar.file_uploader(
    "Upload Files", type=["pdf", "docx"], accept_multiple_files=True
)

# ====================================
# Gemini Setup
# ====================================
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
else:
    st.warning("⚠️ Provide a Google API Key in the sidebar to enable AI answers.")
    model = None

# ====================================
# ChromaDB Setup
# ====================================
client = chromadb.Client()
collection = client.get_or_create_collection("law_docs")

# ====================================
# Load Saved Documents
# ====================================
documents = []
titles = []

if os.path.exists("documents.pkl"):
    with open("documents.pkl", "rb") as f:
        saved = pickle.load(f)
        documents = saved["documents"]
        titles = saved["titles"]
        if documents:
            st.sidebar.success(f"✅ Loaded {len(documents)} previously saved documents.")

# ====================================
# Extract Text from Uploaded Files
# ====================================
for file in uploaded_files:
    text = ""

    # Handle PDF using PyMuPDF
    if file.type == "application/pdf":
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()
        pdf.close()

    # Handle DOCX
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])

    if text.strip() and file.name not in titles:
        documents.append(text)
        titles.append(file.name)

# Save persistently
with open("documents.pkl", "wb") as f:
    pickle.dump({"documents": documents, "titles": titles}, f)

# ====================================
# Generate & Store Embeddings
# ====================================
if documents:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = [embed_model.encode(doc) for doc in documents]

    existing_ids = collection.get().get("ids", [])
    new_ids = [i for i in range(len(documents)) if str(i) not in existing_ids]

    if new_ids:
        collection.add(
            documents=[documents[i] for i in new_ids],
            embeddings=[embeddings[i] for i in new_ids],
            metadatas=[{"title": titles[i]} for i in new_ids],
            ids=[str(i) for i in new_ids],
        )
        st.sidebar.success(f"📚 Added {len(new_ids)} new documents to ChromaDB.")

# ====================================
# Job Search Helper
# ====================================
def get_job_opportunities(keyword: str) -> str:
    link1 = f"https://www.jobberman.com/jobs?query={keyword.replace(' ','+')}"
    link2 = f"https://ngcareers.com/job_search?keywords={keyword.replace(' ','+')}"
    return f"**Job opportunities for '{keyword}':**\n- [Jobberman]({link1})\n- [NGCareers]({link2})"

# ====================================
# Chat Interface
# ====================================
query = st.text_input("💬 Ask your question (Legal or Jobs):")

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    elif not documents:
        st.warning("Upload documents first.")
    else:
        # Job-related query
        if any(w in query.lower() for w in ["job", "vacancy", "career", "employment"]):
            st.subheader("💼 Job Opportunities")
            keyword = query.replace("jobs", "").replace("find", "").strip()
            st.markdown(get_job_opportunities(keyword))

        else:
            # === Retrieve Top 3 Relevant Docs ===
            query_embedding = embed_model.encode(query)
            results = collection.query(query_embeddings=[query_embedding], n_results=3)

            retrieved_docs = []
            context_blocks = []

            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                retrieved_docs.append({"title": meta["title"], "text": doc})
                context_blocks.append(doc)

            context = "\n\n".join(context_blocks)

            # === Generate AI Answer ===
            if model:
                prompt = f"""
                You are a Nigerian Legal & Civic Assistant.
                Use ONLY the context provided to answer.

                --- Context ---
                {context}

                --- Question ---
                {query}

                Provide citations using the document titles.
                """

                response = model.generate_content(prompt)
                answer = response.text

                st.subheader("⚖️ AI Legal & Civic Answer")
                st.write(answer)

                # Show sources
                with st.expander("📚 Source Documents"):
                    for doc in retrieved_docs:
                        st.markdown(f"### {doc['title']}")
                        st.write(doc["text"][:1000] + "...")
                        st.markdown("---")
            else:
                st.error("Missing or invalid Google API Key.")
