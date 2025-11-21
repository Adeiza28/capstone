import os
import streamlit as st
import pdfplumber
import docx
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
import pickle

# ============== Load Environment Variables ==============
load_dotenv()

st.set_page_config(page_title="🇳🇬 AI Legal & Civic Assistant", layout="wide")
st.title("🇳🇬 AI Legal & Civic Assistant for Nigerian Citizens")
st.markdown(
    """
Upload Nigerian **law documents (PDF or DOCX)** and ask legal or civic questions.  
The AI will analyze context, provide answers, and cite sources.  
You can also find **job opportunities** from Jobberman and NGCareers.
"""
)

# ================= Sidebar Settings =================
st.sidebar.header("⚙️ Settings")
api_key = st.sidebar.text_input("🔑 Google API Key", value=os.getenv("GOOGLE_API_KEY"))
st.sidebar.markdown("📄 Upload Nigerian law documents:")
uploaded_files = st.sidebar.file_uploader(
    "Upload Files", type=["pdf", "docx"], accept_multiple_files=True
)

# ================= Configure Gemini =================
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
else:
    st.warning("⚠️ Provide a Google API Key in the sidebar to enable AI answers.")
    model = None

# ================= Initialize ChromaDB =================
client = chromadb.Client()
collection_name = "law_docs"
collection = client.get_or_create_collection(collection_name)

# ================= Load Saved Documents =================
documents = []
titles = []

if os.path.exists("documents.pkl"):
    with open("documents.pkl", "rb") as f:
        saved = pickle.load(f)
        documents = saved["documents"]
        titles = saved["titles"]
        if documents:
            st.sidebar.success(f"✅ Loaded {len(documents)} previously saved documents.")

# ================= Extract Text from Uploaded Docs =================
for file in uploaded_files:
    text = ""
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])

    if text.strip() and file.name not in titles:
        documents.append(text)
        titles.append(file.name)

# Save persistently
with open("documents.pkl", "wb") as f:
    pickle.dump({"documents": documents, "titles": titles}, f)

# ================= Generate & Store Embeddings =================
if documents:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = [embed_model.encode(doc) for doc in documents]

    existing_ids = collection.get()["ids"]  # list of string IDs
    new_ids = []
    for idx, _ in enumerate(documents):
        if str(idx) not in existing_ids:
            new_ids.append(idx)

    if new_ids:
        collection.add(
            documents=[documents[i] for i in new_ids],
            embeddings=[embeddings[i] for i in new_ids],
            metadatas=[{"title": titles[i]} for i in new_ids],
            ids=[str(i) for i in new_ids],
        )
        st.sidebar.success(f"📚 Added {len(new_ids)} new documents to ChromaDB.")

# ================= Job Search Helper =================
def get_job_opportunities(keyword: str) -> str:
    jobberman = f"https://www.jobberman.com/jobs?query={keyword.replace(' ','+')}"
    ngcareers = f"https://ngcareers.com/job_search?keywords={keyword.replace(' ','+')}"
    return f"**Job opportunities for '{keyword}':**\n- [Jobberman]({jobberman})\n- [NGCareers]({ngcareers})"

# ================= Conversation History =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= Main Chat Interface =================
query = st.text_input("💬 Ask your question (Legal or Jobs):")

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    elif not documents:
        st.warning("Upload Nigerian law documents first.")
    else:
        # If query is about jobs
        if any(w in query.lower() for w in ["job", "career", "vacancy", "employment", "work"]):
            st.subheader("💼 Job Opportunities")
            keyword = query.replace("find", "").replace("show", "").replace("jobs", "").strip()
            job_links = get_job_opportunities(keyword)
            st.markdown(job_links)
            st.session_state.history.append({"query": query, "answer": job_links})
        else:
            # ===== Retrieve Top 3 Relevant Docs =====
            embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            query_embedding = embed_model.encode(query)
            results = collection.query(query_embeddings=[query_embedding], n_results=3)

            retrieved_docs = []
            context_texts = []
            for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                if doc:
                    retrieved_docs.append({"title": metadata["title"], "text": doc})
                    context_texts.append(doc)

            context = "\n".join(context_texts)

            # ===== Generate Answer Using Gemini =====
            if model:
                prompt = f"""
                You are a Nigerian Legal & Civic Assistant.
                Answer the user's question clearly, based on the Nigerian law context below.
                Cite document names where relevant.

                Context:
                {context}

                Question:
                {query}
                """
                response = model.generate_content(prompt)
                answer_text = response.text

                st.subheader("⚖️ AI Legal & Civic Answer")
                st.write(answer_text)
                st.session_state.history.append({"query": query, "answer": answer_text})

                # ===== Show Sources =====
                with st.expander("📚 Source Documents"):
                    for doc in retrieved_docs:
                        st.markdown(f"**{doc['title']}**")
                        st.write(doc["text"][:1000] + "...")
                        st.markdown("---")
            else:
                st.info("Please provide a valid Google API Key to use Gemini.")

# ================= Conversation History =================
# if st.session_state.history:
#     st.subheader("📝 Previous Queries & Answers")
#     for entry in reversed(st.session_state.history):
#         st.markdown(f"**Q:** {entry['query']}")
#         st.markdown(f"**A:** {entry['answer']}")
#         st.markdown("---")
