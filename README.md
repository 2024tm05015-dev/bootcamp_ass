# 🚗 Multimodal RAG System for Vehicle Owner’s Manual

An end-to-end **Multimodal Retrieval-Augmented Generation (RAG)** system that enables intelligent querying of vehicle manuals using text, tables, and images.

---

## 📌 Problem Statement

### 1. Domain Identification
This project is situated in the domain of **Automotive Engineering and Customer Experience**, with a focus on post-sales vehicle usage and support systems. Modern vehicles are increasingly complex, incorporating advanced safety systems, electronic controls, and feature-rich interfaces. As a result, vehicle owners rely heavily on owner’s manuals to understand vehicle functionality, troubleshooting procedures, and safety guidelines.

---

### 2. Problem Description
Vehicle owner’s manuals, such as the Tata Punch Owner’s Manual, are comprehensive documents containing critical information about vehicle operation, maintenance, safety systems, and troubleshooting procedures. These manuals are inherently **multimodal**, consisting of textual descriptions, structured tables (e.g., child restraint system compatibility), warning notes, and visual diagrams (e.g., airbag deployment zones, dashboard indicators).

Despite their importance, these documents present significant usability challenges:

- Information is **highly fragmented** across multiple sections, making retrieval slow
- Users struggle to map **real-world problems to manual sections**
- Traditional search fails due to **technical terminology**
- Critical safety instructions are buried in dense text

---

### 3. Why This Problem Is Unique

- Safety-critical instructions require **high accuracy**
- Presence of **tables and structured data**
- Use of **technical automotive terminology**
- Inclusion of **visual diagrams and indicators**
- Context-dependent answers based on vehicle conditions

---

### 4. Why RAG Is the Right Approach

- Avoids expensive fine-tuning
- Enables **semantic search**
- Supports **multimodal retrieval**
- Reduces hallucination using grounded context
- Scales easily to multiple manuals

---

### 5. Expected Outcomes

The system enables:

- Troubleshooting queries:
  - *“Why is my car not starting?”*
- Feature explanations:
  - *“What is ABS?”*
- Safety understanding:
  - *“When do airbags deploy?”*
- Procedural guidance:
  - *“How to change a tyre?”*

---

### 6. Future Scope

- Expand to multiple vehicle models
- Add voice assistant integration
- Improve image understanding
- Integrate real-time diagnostics

---

## 🏗️ Architecture Overview

### 🔄 Pipeline Flow

```mermaid
flowchart LR

A[PDF Upload /ingest] --> B[Docling Parser]
B --> C[Text Extraction]
B --> D[Table Extraction]
B --> E[Image Extraction]

E --> F[VLM Image Summary]

C --> G[Chunking]
D --> G
F --> G

G --> H[Embeddings - BGE Model]
H --> I[ChromaDB Vector Store]

J[User Query] --> K[Embedding]
K --> I
I --> L[Top-K Retrieval]
L --> M[LLM - OpenRouter / HuggingFace]
M --> N[Final Answer + Sources]