import os
import streamlit as st
from pypdf import PdfReader
import docx
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
import pickle

# ===============================
# Load environment variables
# ===============================
load_dotenv()

# ===============================
# Streamlit App Setup
# ===============================
st.set_page_config(page_title="🇳🇬 AI Legal & Civic Assistant", layout="wide")
st.title("🇳🇬 AI Legal & Civic Assistant for Nigerian Citizens")

st.markdown("""
Upload Nigerian **law documents (PDF or DOCX)** and ask legal or civic questions.  
The AI will analyze context, provide answers, and cite sources.  
You can also find **job opportunities** from Jobberman and NGCareers.
""")

# ===============================
# Sidebar Settings
# ===============================
st.sidebar.header("⚙️ Settings")
api_key = st.sidebar.text_input("🔑 Google API Key", value=os.getenv("GOOGLE_API_KEY"))
uploaded_files = st.sidebar.file_uploader(
    "Upload Files", type=["pdf", "docx"], accept_multiple_files=True
)

# ===============================
# Configure Gemini AI
# ===============================
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
else:
    st.warning("⚠️ Provide a Google API Key in the sidebar to enable AI answers.")
    model = None

# ===============================
# Initialize ChromaDB
# ===============================
client = chromadb.Client()
collection = client.get_or_create_collection("law_docs")

# ===============================
# Load saved documents
# ===============================
documents = []
titles = []

if os.path.exists("documents.pkl"):
    with open("documents.pkl", "rb") as f:
        saved = pickle.load(f)
        documents = saved["documents"]
        titles = saved["titles"]
        if documents:
            st.sidebar.success(f"✅ Loaded {len(documents)} previously saved documents.")

# ===============================
# Extract Text from Uploaded Files
# ===============================
for file in uploaded_files:
    text = ""
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join(para.text for para in doc.paragraphs)

    if text.strip() and file.name not in titles:
        documents.append(text)
        titles.append(file.name)

# Save documents persistently
with open("documents.pkl", "wb") as f:
    pickle.dump({"documents": documents, "titles": titles}, f)

# ===============================
# Generate & store embeddings (once)
# ===============================
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
embeddings = []

if documents:
    embeddings = [embed_model.encode(doc) for doc in documents]
    existing_ids = collection.get().get("ids", [])
    new_ids = [idx for idx in range(len(documents)) if str(idx) not in existing_ids]

    if new_ids:
        collection.add(
            documents=[documents[i] for i in new_ids],
            embeddings=[embeddings[i] for i in new_ids],
            metadatas=[{"title": titles[i]} for i in new_ids],
            ids=[str(i) for i in new_ids],
        )
        st.sidebar.success(f"📚 Added {len(new_ids)} new documents to ChromaDB.")

# ===============================
# Job Search Helper
# ===============================
def get_job_opportunities(keyword: str) -> str:
    jobberman = f"https://www.jobberman.com/jobs?query={keyword.replace(' ','+')}"
    ngcareers = f"https://ngcareers.com/job_search?keywords={keyword.replace(' ','+')}"
    return f"**Job opportunities for '{keyword}':**\n- [Jobberman]({jobberman})\n- [NGCareers]({ngcareers})"

# ===============================
# Conversation History
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# Main Chat Interface
# ===============================
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
            job_links = get_job_opportunities(keyword)
            st.markdown(job_links)
            st.session_state.history.append({"query": query, "answer": job_links})
        else:
            # Retrieve Top 3 Relevant Documents
            query_embedding = embed_model.encode(query)
            results = collection.query(query_embeddings=[query_embedding], n_results=3)

            retrieved_docs = []
            context_blocks = []

            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                retrieved_docs.append({"title": meta["title"], "text": doc})
                context_blocks.append(doc)

            context = "\n\n".join(context_blocks)

            # Generate AI Answer with Gemini
            if model:
                prompt = f"""
You are a Nigerian Legal & Civic Assistant.
Use ONLY the context provided to answer the question clearly.

--- Context ---
{context}

--- Question ---
{query}

Provide citations using document titles.
"""
                response = model.generate_content(prompt)
                answer_text = response.text

                st.subheader("⚖️ AI Legal & Civic Answer")
                st.write(answer_text)

                # Show sources
                with st.expander("📚 Source Documents"):
                    for doc in retrieved_docs:
                        st.markdown(f"### {doc['title']}")
                        st.write(doc["text"][:1000] + "...")
                        st.markdown("---")
            else:
                st.error("Missing or invalid Google API Key.")

# ===============================
# Optional: Show conversation history
# ===============================
if st.session_state.history:
    st.subheader("📝 Previous Queries & Answers")
    for entry in reversed(st.session_state.history):
        st.markdown(f"**Q:** {entry['query']}")
        st.markdown(f"**A:** {entry['answer']}")
        st.markdown("---")
