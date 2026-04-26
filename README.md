# 🎮 LoL Strategic Expert Agent (RAG + LoRA)

This project provides an AI-driven strategy agent for League of Legends that combines real-time game statistics with deep domain knowledge. It overcomes the knowledge cutoff of standard LLMs by using **RAG** (Retrieval-Augmented Generation) and **LoRA** fine-tuning.

---

## 📺 Project Demo
* **Demo Video:** [[](https://www.youtube.com/watch?v=LSHDjbMsRf0)]](https://www.youtube.com/watch?v=LSHDjbMsRf0)
**Direct Execution:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZBZOqok2DTJPPOxE4FD9aDGs8xciT7d1)
    * *Note: Run all cells in Colab (T4 GPU required) to generate the public Gradio link.*

---

## 🚀 Key Features
* **Real-time RAG:** Fetches latest champion stats (Win rate, Tiers) using OP.GG MCP API.
* **Domain-Specific Knowledge:** Fine-tuned Mistral-7B model on Inven community strategy data.
* **Efficient Inference:** 4-bit quantization allows high-speed execution on consumer-grade cloud GPUs.

---

## 📂 Repository Structure
* `data/`: Raw scraped text and processed CSV/JSONL files.
* `src/`: Python scripts for data collection, scraping, and processing.
* `notebooks/`: Jupyter notebooks for model training and final execution.
* `requirements.txt`: List of dependencies for environment setup.

---

## 🛠️ How to Run
1.  Open `notebooks/lol_agent_execution.ipynb` in Google Colab.
2.  Set the runtime to **T4 GPU** (Runtime > Change runtime type).
3.  Execute all cells. 
4.  Once the Gradio public URL is generated (`https://xxxx.gradio.live`), click to chat with the agent.

---

## 👨‍💻 Author
* **Name:** Shin Young-in (shinyoungin4137)
* **Affiliation:** Soongsil University, Computer Science & Engineering
