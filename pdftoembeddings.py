import asyncio
import os
import traceback
import ollama
import chromadb
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

# -----------------------------
# Configure Ollama server
# -----------------------------
os.environ["OLLAMA_HOST"] = "http://135.125.143.36:11435"

# -----------------------------
# Ollama embeddings helper
# -----------------------------
def ollama_embed_sync(texts):
    if isinstance(texts, str):
        texts = [texts]
    response = ollama.embed(model="nomic-embed-text", input=texts)
    return response["embeddings"]

async def ollama_embed_func(texts):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: ollama_embed_sync(texts))

# -----------------------------
# Dummy LLM function
# -----------------------------
async def dummy_llm(prompt, **kwargs):
    return "dummy response"

# -----------------------------
# Main workflow
# -----------------------------
async def main():
    pdf_path = r"C:\Users\Administrator\Downloads\test.pdf"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Configure RAGAnything
    # -----------------------------
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=dummy_llm,
        vision_model_func=dummy_llm,
        embedding_func=EmbeddingFunc(768, 8192, ollama_embed_func),
    )

    all_chunks = []

    # -----------------------------
    # Step 1: Check if already processed
    # -----------------------------
    doc_processed = False
    try:
        status = await rag.get_document_processing_status(pdf_path)
        if status.get("is_fully_processed"):
            print("📦 Document already processed — extracting chunks from cache...")
            doc_processed = True
    except Exception:
        pass

    # -----------------------------
    # Step 2: Process PDF if not already processed
    # -----------------------------
    if not doc_processed:
        print("📄 Starting fresh document processing...")
        try:
            await rag.process_document_complete(
                file_path=pdf_path,
                output_dir=output_dir,
                parse_method="auto"
            )
            print(f"✅ PDF parsed and stored in {output_dir}")
        except Exception:
            print("❌ Error during PDF processing:")
            traceback.print_exc()

    # -----------------------------
    # Step 3: Extract chunks safely
    # -----------------------------
    try:
        # 1️⃣ Try LightRAG graph first
        lightrag_obj = getattr(rag, "_lightrag", None) or getattr(rag, "lightrag", None)
        if lightrag_obj:
            graph = getattr(lightrag_obj, "_graph", None) or getattr(lightrag_obj, "graph", None)
            if graph:
                for node_id, node_data in graph.nodes(data=True):
                    if isinstance(node_data, dict) and node_data.get("type") == "text":
                        text = node_data.get("text") or node_data.get("content")
                        if text:
                            all_chunks.append(text)

        # 2️⃣ Fall back to parse_cache (JsonKVStorage) using only allowed attributes
        kv = getattr(rag, "parse_cache", None)
        if not all_chunks and kv is not None:
            all_items = await kv.get_all()  # ✅ await the coroutine
            for key, value in all_items.items():
                if isinstance(value, dict):
                    content_list = value.get("content_list") or value.get("content") or []
                    for block in content_list:
                        if isinstance(block, dict):
                            t = block.get("text") or block.get("content")
                            if t:
                                all_chunks.append(t)

        print(f"🧩 Extracted {len(all_chunks)} text chunks.")
    except Exception:
        print("❌ Error while extracting chunks:")
        traceback.print_exc()

    if not all_chunks:
        print("⚠️ No text chunks found — cannot embed or store.")
        return

    # -----------------------------
    # Step 4: Generate embeddings
    # -----------------------------
    try:
        print("🧠 Generating embeddings...")
        embeddings = await ollama_embed_func(all_chunks)
        print(f"✅ Generated {len(embeddings)} embeddings.")
    except Exception:
        print("❌ Error generating embeddings:")
        traceback.print_exc()
        return

    # -----------------------------
    # Step 5: Store embeddings in ChromaDB
    # -----------------------------
    try:
        print("💾 Storing embeddings in ChromaDB...")
        chroma_client = chromadb.PersistentClient(path="./chroma_storage")
        collection = chroma_client.get_or_create_collection(name="pdf_knowledge_base")

        ids = [f"chunk_{i}" for i in range(len(all_chunks))]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=all_chunks
        )
        print("✅ Embeddings stored in ChromaDB.")
    except Exception:
        print("❌ Error storing embeddings in ChromaDB:")
        traceback.print_exc()

    # -----------------------------
    # Step 6: Finalize RAG storages safely
    # -----------------------------
    try:
        await rag.finalize_storages()
    except Exception as e:
        print(f"⚠️ Warning while finalizing storages: {e}")

    print("🏁 Workflow complete.")

if __name__ == "__main__":
    asyncio.run(main())
