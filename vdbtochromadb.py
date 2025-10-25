import asyncio
import os
import traceback
import ollama
import chromadb
import json

# -----------------------------
# Configure Ollama server
# -----------------------------
os.environ["OLLAMA_HOST"] = "http://ollama:11434"

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
# Main workflow
# -----------------------------
async def main():
    vdb_path = "./rag_storage/vdb_chunks.json"
    
    # -----------------------------
    # Step 1: Load all chunks from vdb_chunks.json
    # -----------------------------
    try:
        with open(vdb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_chunks = [chunk["content"] for chunk in data.get("data", []) if "content" in chunk]
        print(f"🧩 Total chunks loaded from vdb_chunks.json: {len(all_chunks)}")
    except Exception:
        print("❌ Failed to load vdb_chunks.json:")
        traceback.print_exc()
        return

    if not all_chunks:
        print("⚠️ No chunks found in vdb_chunks.json — cannot embed or store.")
        return

    # -----------------------------
    # Step 2: Generate embeddings
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
    # Step 3: Store embeddings in ChromaDB
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
        print("✅ All embeddings stored in ChromaDB.")
    except Exception:
        print("❌ Error storing embeddings in ChromaDB:")
        traceback.print_exc()

    print("🏁 Workflow complete.")

if __name__ == "__main__":
    asyncio.run(main())
