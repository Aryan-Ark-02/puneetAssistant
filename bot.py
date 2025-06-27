import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.voyageai import VoyageEmbedding
import asyncio

# Load environment variables from .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY environment variable not set. Please check your .env file.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please check your .env file.")

RESUME_PATH = os.path.join(os.path.dirname(__file__), '../PUNEET SINHA_15_CC.pdf')
STORAGE_DIR = os.path.join(os.path.dirname(__file__), 'storage')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global objects for efficiency
embed_model = VoyageEmbedding(
    voyage_api_key=VOYAGE_API_KEY,
    model_name="voyage-3.5",
)
llm = Gemini(api_key=GEMINI_API_KEY)

# Helper to build or load the index
async def get_index():
    if os.path.exists(STORAGE_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context, embed_model=embed_model)
    else:
        document = LlamaParse(
            api_key=os.getenv("LLAMA_PARSE_API_KEY"),
            result_type="markdown",
            content_guideline_instruction="Extract the main points of the document"
        ).load_data(RESUME_PATH)
        index = VectorStoreIndex.from_documents(document, embed_model=embed_model)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
    return index

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "")
    if not question:
        return {"answer": "No question provided."}
    index = await get_index()
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)
    response = query_engine.query(question)
    return {"answer": str(response)}
