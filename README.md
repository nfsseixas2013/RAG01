# RAG of book Mobile Big Data: A roadmap from models to technologies

**Hi there! Yet another LLM enthusiast joining the scene.**

This project showcases the power of **Retrieval-Augmented Generation (RAG)** in action. I developed a **FastAPI-based API** that processes user queries and, with the help of **DeepSeek**, delivers contextualized responses based on the content of this book.

First, I’ll provide a high-level overview of the project architecture, followed by step-by-step instructions on how to use it.

# System Architecture

![RAG01 (1).jpg](RAG/RAG01_(1).jpg)

The image illustrates the system's design. First, we extract text from the PDF and split it into manageable chunks. Next, we use the **"all-mpnet-base-v2"** embedding model to generate vector representations of these chunks. These embeddings are then stored as **PyTorch tensors**.

When a user submits a query, it is also embedded and compared against the stored tensor using a **dot product** to determine the most relevant chunks—a process known as **semantic search**. The retrieved chunks serve as contextual information in a dynamically generated prompt, which is then passed to **DeepSeek** for inference. Finally, the system returns a well-contextualized answer to the user.

For other embedding models, please see [https://sbert.net/docs/sentence_transformer/pretrained_models.html](https://sbert.net/docs/sentence_transformer/pretrained_models.html)

# How to use it

Python used in this project: 3.9.19

```bash
poetry install
```

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

To access the docs, digit: [http://localhost:8000/rag-api/docs](http://10.27.0.18:8000/rag-api/docs)
