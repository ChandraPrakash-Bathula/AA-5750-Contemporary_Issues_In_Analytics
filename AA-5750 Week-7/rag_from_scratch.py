# -*- coding: utf-8 -*-
"""
RAG From Scratch – End-to-End Local Pipeline (Single File)

This script builds a complete Retrieval-Augmented Generation (RAG) system:

1. Download & parse a PDF (Human Nutrition textbook).
2. Extract text and perform sentence-based chunking.
3. Generate dense embeddings (SentenceTransformers: all-mpnet-base-v2).
4. Save & reload embeddings + metadata safely.
5. Perform semantic search (vector search) for a query.
6. Augment a local LLM (Gemma-7B-IT or similar) with retrieved context.
7. Generate grounded answers.
8. Optionally evaluate the RAG pipeline using RAGAS + OpenAI (LLM-as-judge).

References:
- Sentence-Transformers (embeddings)  [https://www.sbert.net/]
- all-mpnet-base-v2 model card        [https://huggingface.co/sentence-transformers/all-mpnet-base-v2]
- Gemma-7B-IT (local LLM)             [https://huggingface.co/google/gemma-7b-it]
- RAGAS (RAG evaluation)              [https://github.com/explodinggradients/ragas]
- LangChain OpenAI wrappers           [https://python.langchain.com/docs/integrations/llms/openai]

NOTE:
- No API keys are hard-coded. Set OPENAI_API_KEY / HF_TOKEN externally.
- This is written as a single Python file for clarity and teaching.
"""

from __future__ import annotations

import os
import re
import textwrap
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import requests
import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm

import fitz  # PyMuPDF  (PDF parsing)

# spaCy for sentence segmentation
import spacy

# Sentence-Transformers for embeddings
from sentence_transformers import SentenceTransformer, util  # [sbert.net]

# Hugging Face Transformers for local LLM
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.utils import is_flash_attn_2_available

# For optional evaluation (RAGAS + OpenAI + LangChain)
# These imports are guarded later to avoid hard failure if not installed.
# - ragas: https://github.com/explodinggradients/ragas
# - langchain-openai: https://python.langchain.com/docs/integrations/llms/openai


# =============================================================================
# 0. Global Config
# =============================================================================

PDF_URL = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
PDF_PATH = "./Human_Nutrition.pdf"

CHUNK_METADATA_PATH = "./chunk_metadata.parquet"
CHUNK_EMBEDDINGS_PATH = "./chunk_embeddings.npy"

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # [HuggingFace model card]
LLM_MODEL_ID = "google/gemma-7b-it"  # instruction-tuned, good for RAG-style QA

MIN_TOKENS_PER_CHUNK = 30           # prune ultra-short fragments
SENTENCES_PER_CHUNK = 10            # sentence-based chunking size
TOP_K_RETRIEVAL = 5                 # default K for retrieval


# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")


# =============================================================================
# 1. Utilities
# =============================================================================

def print_wrapped(text: str, width: int = 90) -> None:
    """Pretty-print long text with wrapping for readability."""
    print(textwrap.fill(text, width))


# =============================================================================
# 2. PDF Download & Text Extraction
# =============================================================================

def download_pdf(pdf_path: str = PDF_PATH, url: str = PDF_URL) -> None:
    """Download the PDF if it does not exist locally."""
    if os.path.exists(pdf_path):
        print(f"[INFO] PDF already exists at: {pdf_path}")
        return

    print("[INFO] PDF not found locally. Downloading...")
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download PDF; status code = {resp.status_code}")

    with open(pdf_path, "wb") as f:
        f.write(resp.content)

    print(f"[INFO] Download complete: {pdf_path}")


def preprocess_page_text(raw_text: str) -> str:
    """Light page-level cleaning. Customize as needed."""
    return raw_text.replace("\n", " ").strip()


@dataclass
class PageRecord:
    page_index: int
    document_page_number: int
    char_count: int
    word_count: int
    token_count_approx: float
    text: str


def extract_pdf_text_with_stats(
    pdf_path: str,
    page_offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF page-by-page and computes basic text statistics.

    page_offset is used if the printed page number differs from PDF index.
    """
    doc = fitz.open(pdf_path)
    pages: List[Dict[str, Any]] = []

    for raw_page_index, page in tqdm(
        enumerate(doc),
        total=len(doc),
        desc="Extracting PDF pages"
    ):
        raw_text = page.get_text("text")
        cleaned = preprocess_page_text(raw_text)

        rec = {
            "page_index": raw_page_index,
            "document_page_number": raw_page_index + page_offset,
            "char_count": len(cleaned),
            "word_count": len(cleaned.split()),
            "token_count_approx": len(cleaned) / 4.0,  # rough heuristic
            "text": cleaned,
        }
        pages.append(rec)

    doc.close()
    return pages


# =============================================================================
# 3. Sentence Segmentation & Chunking
# =============================================================================

def build_spacy_sentencizer() -> Any:
    """
    Build a lightweight spaCy pipeline for English sentence segmentation.

    Uses 'en_core_web_sm' model which must be installed separately:
    - python -m spacy download en_core_web_sm
    """
    nlp = spacy.load("en_core_web_sm")
    # Sentencizer is already built-in; no extra pipeline needed if model loaded.
    return nlp


def chunk_sentences(sentences: List[str], size: int) -> List[List[str]]:
    """Split a list of sentences into fixed-size chunks."""
    return [sentences[i:i + size] for i in range(0, len(sentences), size)]


def create_sentence_chunks(
    pdf_pages: List[Dict[str, Any]],
    nlp: Any,
    sentences_per_chunk: int = SENTENCES_PER_CHUNK
) -> List[Dict[str, Any]]:
    """
    Add sentence and chunk info to each page record.
    Returns a flat list of chunk-level records.
    """
    # Add sentences to each page
    for page in tqdm(pdf_pages, desc="Segmenting sentences"):
        doc = nlp(page["text"])
        sentences = [str(s) for s in doc.sents]
        page["sentences"] = sentences
        page["sentence_count_spacy"] = len(sentences)

    # Flatten into chunk-level records
    chunk_records: List[Dict[str, Any]] = []
    for page in tqdm(pdf_pages, desc="Creating sentence chunks"):
        sentence_chunks = chunk_sentences(
            page["sentences"],
            sentences_per_chunk
        )
        for chunk_idx, sentence_group in enumerate(sentence_chunks):
            chunk_text = " ".join(sentence_group).strip()
            # Fix missing space after periods like "word.Another"
            chunk_text = re.sub(r"\.([A-Z])", r". \1", chunk_text)

            chunk_record = {
                "page_index": page["page_index"],
                "document_page_number": page["document_page_number"],
                "chunk_index": chunk_idx,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "token_count_approx": len(chunk_text) / 4.0,
            }
            chunk_records.append(chunk_record)

    return chunk_records


def filter_chunks_by_min_tokens(
    chunk_records: List[Dict[str, Any]],
    min_tokens: int = MIN_TOKENS_PER_CHUNK
) -> List[Dict[str, Any]]:
    """Remove low-information fragments with very few tokens."""
    df = pd.DataFrame(chunk_records)
    filtered_df = df[df["token_count_approx"] > min_tokens]
    print(f"[INFO] Filtered chunks: {len(df)} -> {len(filtered_df)} (min_tokens={min_tokens})")
    return filtered_df.to_dict(orient="records")


# =============================================================================
# 4. Embeddings – Build, Save, Load
# =============================================================================

def load_embedding_model(model_name: str = EMBED_MODEL_NAME, device: str = DEVICE) -> SentenceTransformer:
    """Load a Sentence-Transformers embedding model."""
    print(f"[INFO] Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name_or_path=model_name, device=device)
    return model


def build_and_save_embeddings(
    chunk_records: List[Dict[str, Any]],
    model: SentenceTransformer,
    metadata_path: str = CHUNK_METADATA_PATH,
    embeddings_path: str = CHUNK_EMBEDDINGS_PATH,
    batch_size: int = 32
) -> None:
    """
    Generate embeddings for all chunks, and persist:
    - metadata → parquet
    - embeddings → .npy
    """
    texts = [rec["text"] for rec in chunk_records]

    print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Save metadata
    df = pd.DataFrame(chunk_records)
    df.to_parquet(metadata_path, index=False)
    print(f"[INFO] Saved chunk metadata to: {metadata_path}")

    # Save embeddings
    np.save(embeddings_path, embeddings)
    print(f"[INFO] Saved embeddings to: {embeddings_path}")


def load_embeddings_and_metadata(
    metadata_path: str = CHUNK_METADATA_PATH,
    embeddings_path: str = CHUNK_EMBEDDINGS_PATH,
    device: str = DEVICE
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    """
    Load chunk metadata and embedding matrix from disk.
    Returns:
        chunk_records: list[dict] (metadata)
        embedding_matrix: torch.Tensor on the specified device
    """
    if not os.path.exists(metadata_path) or not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"Missing {metadata_path} or {embeddings_path}. "
            f"Run embedding generation first."
        )

    df = pd.read_parquet(metadata_path)
    chunk_records = df.to_dict(orient="records")

    embeddings_np = np.load(embeddings_path)
    embedding_matrix = torch.tensor(
        embeddings_np,
        dtype=torch.float32,
        device=device
    )

    print(f"[INFO] Loaded {len(chunk_records)} chunk records and "
          f"embedding matrix of shape {embedding_matrix.shape}")

    return chunk_records, embedding_matrix


# =============================================================================
# 5. Semantic Search (Retrieve Top-k)
# =============================================================================

def retrieve_top_k(
    query: str,
    embedding_matrix: torch.Tensor,
    model: SentenceTransformer,
    k: int = TOP_K_RETRIEVAL,
    device: str = DEVICE,
    print_time: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Embed a query and retrieve top-k most similar embeddings.

    Returns:
        scores  : similarity scores (Tensor[k])
        indices : indices into embedding_matrix (Tensor[k])
    """
    # Ensure model is on correct device
    model.to(device)

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        device=device
    )

    # dot_score uses batched matrix multiplication under the hood.
    start = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

    if device == "cuda":
        start.record()

    scores = util.dot_score(query_embedding, embedding_matrix)[0]

    if device == "cuda":
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        if print_time:
            print(f"[INFO] Scored {embedding_matrix.shape[0]} embeddings "
                  f"in {elapsed_ms:.4f} ms")
    else:
        if print_time:
            print(f"[INFO] Scored {embedding_matrix.shape[0]} embeddings (CPU)")

    top_scores, top_indices = torch.topk(scores, k=k)
    return top_scores, top_indices


def print_search_results(
    query: str,
    embedding_matrix: torch.Tensor,
    chunk_records: List[Dict[str, Any]],
    model: SentenceTransformer,
    k: int = TOP_K_RETRIEVAL,
    device: str = DEVICE
) -> None:
    """Debug helper: run semantic search and print human-readable results."""
    scores, indices = retrieve_top_k(
        query=query,
        embedding_matrix=embedding_matrix,
        model=model,
        k=k,
        device=device
    )

    print(f"\nQuery: {query!r}\n")
    print("Top Results:\n")

    for score, idx in zip(scores, indices):
        chunk = chunk_records[idx.item()]
        print(f"Score: {float(score):.4f}")
        print("Text:")
        print_wrapped(chunk["text"])
        print(f"Document page: {chunk['document_page_number']}")
        print("-" * 80)


# =============================================================================
# 6. RAG Prompt Formatter
# =============================================================================

def prompt_formatter(
    query: str,
    context_items: List[Dict[str, Any]],
) -> str:
    """
    Format prompt for RAG: query + retrieved context.

    This is the "A" (Augmentation) in RAG. We:
    - Serialize context chunks with indices.
    - Provide clear instructions to the LLM to stay grounded.

    Citations:
    - Prompt engineering best practices are still emerging; see:
      Prompting Guide [https://www.promptingguide.ai]
      Brex Prompt Engineering [https://github.com/brexhq/prompt-engineering]
    """
    context_blocks = []
    for i, item in enumerate(context_items, start=1):
        text_content = item["text"]
        page = item["document_page_number"]
        context_blocks.append(f"[{i}] (page {page}) {text_content}")

    context = "\n".join(context_blocks)

    base_prompt = f"""
You are a careful, factual teaching assistant answering questions about human nutrition.

You are given several context passages from a textbook. Your job is to:

1. Answer the user query using ONLY the information in the context.
2. Do NOT use outside knowledge, even if you think you know the answer.
3. If the answer cannot be found in the context, say:
   "I don't know based on the provided document."
4. When relevant, mention citations like [1], [2] referring to the context items.

---
Context:
{context}

---
User Query:
{query}

Now provide a clear, detailed answer grounded ONLY in the context above.
Include inline citations like [1], [2] pointing to the context items you used.

Answer:
""".strip()

    return base_prompt


# =============================================================================
# 7. Local LLM Loader (Gemma-7B-IT or similar)
# =============================================================================

def select_attention_impl(device: str = DEVICE) -> str:
    """Choose best attention implementation available."""
    if (
        device == "cuda"
        and is_flash_attn_2_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
    ):
        print("[INFO] Using flash_attention_2")
        return "flash_attention_2"
    else:
        print("[INFO] Using SDPA attention")
        return "sdpa"


def load_local_llm(
    model_id: str = LLM_MODEL_ID,
    device: str = DEVICE,
    use_quantization: bool | None = None
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load a local instruction-tuned LLM for RAG generation.

    Default: google/gemma-7b-it (good trade-off for quality vs speed).

    Citations:
    - Gemma model card: https://huggingface.co/google/gemma-7b-it
    """
    # Simple heuristic: quantize if VRAM < 20GB
    if use_quantization is None:
        if not torch.cuda.is_available():
            use_quantization = True
        else:
            props = torch.cuda.get_device_properties(0)
            gpu_mem_gb = round(props.total_memory / (2**30))
            use_quantization = gpu_mem_gb < 20

    print(f"[INFO] Loading LLM: {model_id}")
    print(f"[INFO] Quantization: {use_quantization}")

    quantization_config = None
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("[INFO] 4-bit NF4 quantization enabled (bitsandbytes).")

    attn_impl = select_attention_impl(device=device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True
    )

    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
        device_map="auto" if quantization_config else None
    )

    if device == "cuda" and not quantization_config:
        llm.to(device)

    llm.eval()
    print("[INFO] LLM loaded and ready.")
    return tokenizer, llm


# =============================================================================
# 8. RAG Generation – High Level `ask()`
# =============================================================================

def ask(
    query: str,
    retrieval_model: SentenceTransformer,
    embedding_matrix: torch.Tensor,
    chunk_records: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    llm_model: AutoModelForCausalLM,
    device: str = DEVICE,
    k: int = TOP_K_RETRIEVAL,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    return_answer_only: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    End-to-end RAG function:
    - Retrieve top-k chunks
    - Build RAG prompt with citations
    - Run local LLM
    - Return answer (+ context if requested)
    """

    # 1) RETRIEVE
    scores, indices = retrieve_top_k(
        query=query,
        embedding_matrix=embedding_matrix,
        model=retrieval_model,
        device=device,
        k=k
    )

    # 2) Map to records; attach scores WITHOUT mutating global state
    context_items = []
    for score, idx in zip(scores, indices):
        rec = dict(chunk_records[idx.item()])
        rec["score"] = float(score.cpu())
        context_items.append(rec)

    # 3) AUGMENT – build base prompt and apply chat template
    raw_prompt = prompt_formatter(query=query, context_items=context_items)

    full_model_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

    # 4) GENERATE
    inputs = tokenizer(
        full_model_prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            temperature=temperature,
            do_sample=True,
            max_new_tokens=max_new_tokens
        )

    decoded = tokenizer.decode(outputs[0])

    # 5) Strip prompt + special tokens for clean answer
    answer = (
        decoded
        .replace(full_model_prompt, "")
        .replace("<bos>", "")
        .replace("<eos>", "")
        .strip()
    )

    if return_answer_only:
        return answer, context_items

    return answer, context_items


# =============================================================================
# 9. RAGAS Evaluation (Optional)
# =============================================================================

def run_ragas_evaluation(
    evaluation_data: List[Dict[str, Any]]
) -> None:
    """
    Run RAGAS evaluation on the RAG system, using:
    - OpenAI GPT-4o-mini as judge LLM (or compatible)
    - OpenAI text-embedding-3-large for internal embeddings

    Requires:
    - OPENAI_API_KEY in environment
    - ragas, datasets, langchain-openai installed

    Citations:
    - RAGAS: https://github.com/explodinggradients/ragas
      Krantz et al., "RAGAS: Automated Evaluation of Retrieval-Augmented Generation"
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            context_precision,
            context_recall,
            answer_relevancy,
            faithfulness,
        )
        try:
            from ragas.metrics import context_entity_recall
        except ImportError:
            context_entity_recall = None

        try:
            from ragas.metrics import noise_robustness
        except ImportError:
            noise_robustness = None

        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as e:
        print(f"[WARN] RAGAS/related deps not installed: {e}")
        print("[WARN] Skipping RAGAS evaluation.")
        return

    if "OPENAI_API_KEY" not in os.environ:
        print("[WARN] OPENAI_API_KEY not set; skipping RAGAS evaluation.")
        return

    eval_df = pd.DataFrame(evaluation_data)
    eval_dataset = Dataset.from_pandas(eval_df)

    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
    )

    metrics = [
        context_precision,
        context_recall,
        answer_relevancy,
        faithfulness
    ]
    if context_entity_recall is not None:
        metrics.append(context_entity_recall)
    if noise_robustness is not None:
        metrics.append(noise_robustness)

    print("[INFO] Running RAGAS evaluation...")
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    results_df = results.to_pandas()

    # Identify numeric metric columns
    metric_cols: List[str] = []
    for col in results_df.columns:
        if col in ["user_input", "retrieved_contexts", "response", "reference"]:
            continue
        values = pd.to_numeric(results_df[col], errors="coerce")
        if not values.isna().all():
            metric_cols.append(col)

    print("\n" + "=" * 80)
    print("                        RAG EVALUATION RESULTS")
    print("=" * 80)
    print(results_df[metric_cols].round(3))

    # Summary
    summary_df = (
        results_df[metric_cols]
        .astype(float)
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )

    print("\n📈 OVERALL RAG PIPELINE PERFORMANCE")
    print("-" * 80)
    print(summary_df)

    avg_scores = results_df[metric_cols].astype(float).mean().round(3)
    print("\n📈 OVERALL AVERAGE SCORES")
    print("-" * 50)
    for metric, value in avg_scores.items():
        print(f"{metric.replace('_', ' ').title():<30}: {value:.3f}")

    overall_avg = avg_scores.mean()
    if overall_avg >= 0.8:
        rating = "🌟 Excellent"
    elif overall_avg >= 0.6:
        rating = "👍 Good"
    elif overall_avg >= 0.4:
        rating = "😐 Fair"
    else:
        rating = "⚠️ Poor"

    print("\n📌 PERFORMANCE SUMMARY")
    print("-" * 50)
    print(f"Overall RAG Quality Score: {overall_avg:.3f}")
    print(f"Overall Rating: {rating}")

    # Save detailed results
    detailed_results = pd.DataFrame(evaluation_data)
    for col in metric_cols:
        detailed_results[col] = results_df[col].values

    detailed_results.to_csv("rag_evaluation_results.csv", index=False)
    print("\n[INFO] Detailed RAGAS results saved to 'rag_evaluation_results.csv'")


# =============================================================================
# 10. Main Orchestration
# =============================================================================

def main() -> None:
    # -----------------------------
    # 1. Download & Extract PDF
    # -----------------------------
    download_pdf(PDF_PATH, PDF_URL)

    print("[INFO] Extracting PDF text...")
    pdf_pages = extract_pdf_text_with_stats(
        pdf_path=PDF_PATH,
        page_offset=-41  # printed numbering starts later (e.g., page 42)
    )

    # -----------------------------
    # 2. Sentence Segmentation + Chunking
    # -----------------------------
    print("[INFO] Loading spaCy sentencizer...")
    nlp = build_spacy_sentencizer()

    print("[INFO] Creating sentence chunks...")
    chunk_records = create_sentence_chunks(pdf_pages, nlp)

    print("[INFO] Filtering low-information chunks...")
    filtered_chunk_records = filter_chunks_by_min_tokens(
        chunk_records,
        min_tokens=MIN_TOKENS_PER_CHUNK
    )

    # -----------------------------
    # 3. Embeddings: Build or Load
    # -----------------------------
    embedding_model = load_embedding_model(EMBED_MODEL_NAME, device=DEVICE)

    if not (os.path.exists(CHUNK_METADATA_PATH) and os.path.exists(CHUNK_EMBEDDINGS_PATH)):
        print("[INFO] Embeddings not found on disk; generating...")
        build_and_save_embeddings(
            filtered_chunk_records,
            model=embedding_model,
            metadata_path=CHUNK_METADATA_PATH,
            embeddings_path=CHUNK_EMBEDDINGS_PATH
        )
    else:
        print("[INFO] Found existing embeddings; skipping generation.")

    chunk_records_loaded, embedding_matrix = load_embeddings_and_metadata(
        metadata_path=CHUNK_METADATA_PATH,
        embeddings_path=CHUNK_EMBEDDINGS_PATH,
        device=DEVICE
    )

    # -----------------------------
    # 4. Test Semantic Search
    # -----------------------------
    test_query = "What are macronutrients and how does each contribute to energy production?"
    print("\n" + "=" * 80)
    print("SEMANTIC SEARCH DEMO")
    print("=" * 80)
    print_search_results(
        query=test_query,
        embedding_matrix=embedding_matrix,
        chunk_records=chunk_records_loaded,
        model=embedding_model,
        k=3,
        device=DEVICE
    )

    # -----------------------------
    # 5. Load Local LLM
    # -----------------------------
    print("\n" + "=" * 80)
    print("LOADING LOCAL LLM")
    print("=" * 80)
    tokenizer, llm_model = load_local_llm(
        model_id=LLM_MODEL_ID,
        device=DEVICE
    )

    # -----------------------------
    # 6. End-to-end RAG Demo
    # -----------------------------
    print("\n" + "=" * 80)
    print("RAG DEMO – FULL PIPELINE")
    print("=" * 80)

    query_list = [
        "What are macronutrients and how does each contribute to energy production?",
        "Describe the process of digestion from the mouth to the small intestine.",
        "How do vitamins differ from minerals in biological function?",
        "What role does fiber play in digestion? Name five fiber-rich foods.",
        "How often should infants be breastfed?",
        "What are the symptoms of pellagra?",
        "How does saliva help with digestion?",
        "What is the recommended daily intake of protein?",
        "What are micronutrients?"
    ]

    demo_query = random.choice(query_list)
    print(f"Demo Query: {demo_query}\n")

    answer, ctx_items = ask(
        query=demo_query,
        retrieval_model=embedding_model,
        embedding_matrix=embedding_matrix,
        chunk_records=chunk_records_loaded,
        tokenizer=tokenizer,
        llm_model=llm_model,
        device=DEVICE,
        k=TOP_K_RETRIEVAL,
        temperature=0.7,
        max_new_tokens=512,
        return_answer_only=True
    )

    print("RAG Answer:\n")
    print_wrapped(answer)
    print("\n---\nContext used (top 2 chunks):\n")
    for i, item in enumerate(ctx_items[:2], start=1):
        print(f"[{i}] (score={item['score']:.4f}, page={item['document_page_number']})")
        print_wrapped(item["text"])
        print("-" * 80)

    # -----------------------------
    # 7. Optional RAGAS Evaluation
    # -----------------------------
    print("\n" + "=" * 80)
    print("OPTIONAL: RAGAS EVALUATION")
    print("=" * 80)

    eval_questions = [
        "How often should infants be breastfed?",
        "What are symptoms of pellagra?",
        "How does saliva help with digestion?",
        "What is the recommended protein intake per day, based on your weight?",
        "What are micronutrients?"
    ]

    ground_truth_answers = [
        ("A newborn infant (birth to 28 days) requires feedings eight to twelve times a day or more. "
         "Between 1 and 3 months of age, the breastfed infant becomes more efficient, and the number "
         "of feedings per day often becomes fewer even though the amount of milk consumed stays the same."),
        ("Niacin deficiency is commonly known as pellagra and the symptoms include fatigue, decreased "
         "appetite, and indigestion. These symptoms are then commonly followed by the four D's: "
         "diarrhea, dermatitis, dementia, and sometimes death."),
        ("The mechanical and chemical digestion of carbohydrates begins in the mouth. Chewing (mastication) "
         "breaks food into smaller pieces. Saliva secreted by the salivary glands contains the enzyme "
         "salivary amylase, which begins the breakdown of starches into smaller glucose chains such as "
         "dextrins and maltose."),
        ("The recommended protein intake can be estimated using the equation: "
         "(body weight in kilograms × 0.8 grams per kilogram). "
         "If a person is overweight, this calculation may overestimate protein needs."),
        ("Micronutrients are nutrients required in small amounts but are essential for normal body "
         "functions. They include vitamins and minerals.")
    ]

    evaluation_data: List[Dict[str, Any]] = []

    for q, gt in zip(eval_questions, ground_truth_answers):
        print(f"[EVAL] Generating answer for: {q[:50]}...")
        ans, ctx = ask(
            query=q,
            retrieval_model=embedding_model,
            embedding_matrix=embedding_matrix,
            chunk_records=chunk_records_loaded,
            tokenizer=tokenizer,
            llm_model=llm_model,
            device=DEVICE,
            k=TOP_K_RETRIEVAL,
            temperature=0.7,
            max_new_tokens=512,
            return_answer_only=True
        )
        evaluation_data.append({
            "question": q,
            "answer": ans,
            "contexts": [c["text"] for c in ctx],
            "ground_truth": gt
        })

    run_ragas_evaluation(evaluation_data)


if __name__ == "__main__":
    main()