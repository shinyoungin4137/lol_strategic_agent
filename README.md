# 🎮 LoL Strategic Expert Agent (RAG + RAFT LoRA)

This project provides an AI-driven strategy agent for League of Legends that combines up-to-date game statistics with deep domain knowledge. It overcomes the knowledge cutoff of standard LLMs by using **RAG** (Retrieval-Augmented Generation) and **RAFT-based LoRA** fine-tuning.

> **Term Project 2 (upgraded) version.** See "What's New" below for changes over the original system.

---

## 📺 Project Demo

* **Demo Video:** [Watch on YouTube](https://www.youtube.com/watch?v=tbED2T05VFA)
* **Direct Execution:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pr4LHnfpyk6pcyFDwd-QN7UQ3u3vFMFV)
    * *Note: Run all cells in Colab (T4 GPU required) to generate the public Gradio link. The model and RAG data download automatically from Hugging Face.*

---

## 🚀 Key Features

* **RAFT Fine-Tuning:** Training data reconstructed with oracle + distractor documents so the model grounds its answers in retrieved context (fixing the prior version's context-ignoring hallucinations).
* **Up-to-Date RAG:** Fetches the latest champion stats (win rate, tier, core items, counters) using the OP.GG MCP API.
* **Stronger Retrieval:** Multilingual **BGE-M3** embedder + **bge-reranker-v2-m3** cross-encoder re-ranking (top-5); retrieval context enriched with counters and lane.
* **Domain Knowledge:** Fine-tuned **Mistral-7B-v0.3** on League of Legends Wiki strategy data.
* **Efficient Inference:** 4-bit quantization enables high-speed execution on consumer-grade cloud GPUs; the LoRA adapter is hosted on Hugging Face.

---

## 🆕 What's New (vs Term Project 1)

* Unified base model (Mistral-7B-v0.3) across training and inference
* RAFT-format training data (1,268 examples) vs context-free QA (519)
* Full-epoch training vs a 60-step smoke test
* BGE-M3 + reranker vs a Korean-only embedder with top-1 retrieval
* Counter-matchup queries now supported
* Base / RAG / RAG+LoRA ablation evaluation

---

## 📂 Repository Structure

* `data/`: Raw scraped text and processed CSV/JSONL files (incl. `lora_training_data_v2_raft.jsonl`).
* `src/`: Python scripts for data collection, scraping, and RAFT conversion.
* `train_LORA.ipynb`: RAFT LoRA fine-tuning (Mistral-7B-v0.3, Unsloth, 4-bit).
* `LOL_Agent_Execution_V2.ipynb`: RAG + agent inference + Gradio demo.

---

## 🛠️ How to Run

1.  Open [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pr4LHnfpyk6pcyFDwd-QN7UQ3u3vFMFV)
2.  Set the runtime to **T4 GPU** (Runtime > Change runtime type).
3.  Execute all cells (the model and RAG data download automatically from Hugging Face).
4.  Once the Gradio public URL is generated (`https://xxxx.gradio.live`), click to chat with the agent.

---

## 👨‍💻 Author

* **Name:** Shin Young-in (shinyoungin4137)
* **Affiliation:** Soongsil University, Computer Science & Engineering
