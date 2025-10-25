import asyncio
import json
import os
import ollama
import chromadb

# -----------------------------
# Configure Ollama server
# -----------------------------
os.environ["OLLAMA_HOST"] = "http://ollama:11434"

# -----------------------------
# Ollama helpers
# -----------------------------
def ollama_model_complete_sync(model, prompt, system_prompt=None, history_messages=[]):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]

async def ollama_model_complete(model, prompt, system_prompt=None, history_messages=[]):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: ollama_model_complete_sync(model, prompt, system_prompt, history_messages)
    )

def ollama_embed_sync(texts):
    if isinstance(texts, str):
        texts = [texts]
    return ollama.embed(model="nomic-embed-text", input=texts)["embeddings"]

async def ollama_embed_func(texts):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: ollama_embed_sync(texts))

# -----------------------------
# Chat memory helpers
# -----------------------------
chat_history_file = "chat_memory.json"

def load_memory():
    if os.path.exists(chat_history_file):
        with open(chat_history_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def update_memory(user_msg, assistant_msg):
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": assistant_msg})
    with open(chat_history_file, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

# -----------------------------
# Main chat loop
# -----------------------------
async def main():
    global chat_history
    chat_history = load_memory()

    # Load ChromaDB collection
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    collection = chroma_client.get_collection("pdf_knowledge_base")

    print("💬 Chatbot ready. Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Embed user query
        query_embedding = await ollama_embed_func(user_input)

        # Retrieve top relevant chunks
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=500,            # retrieve more chunks
            include=["documents"]     # ensure documents are returned
        )

        # Flatten all retrieved documents
        retrieved_texts = []
        for docs_per_query in results.get("documents", []):
            retrieved_texts.extend(docs_per_query)

        context = "\n".join(retrieved_texts)

        final_prompt = f"""
You are a helpful assistant. Use the following document context and chat history to answer.

Chat History:
{[m['content'] for m in chat_history]}

Document Context:
{context}

User Question: {user_input}
"""
        response = await ollama_model_complete("llama3.1", final_prompt)
        print("Assistant:", response)
        update_memory(user_input, response)

if __name__ == "__main__":
    asyncio.run(main())
