# EdgeMind

**A private, on-device AI workspace copilot optimised for Snapdragon-powered HP PCs.**

EdgeMind indexes your documents and notes locally, then lets you ask questions, summarise long material and search across everything, with cited sources. Models run on the Snapdragon X **Hexagon NPU** through ONNX Runtime's QNN execution provider, so it is fast, battery-friendly and private. No document text ever leaves the device, and it works offline.

## Documentation

| Document | Description |
|---|---|
| [Project proposal (PDF)](docs/EdgeMind_Proposal.pdf) | Full proposal: problem, solution, architecture, optimisation strategy, roadmap, risks |
| [Presentation (PPTX)](docs/EdgeMind_Presentation.pptx) | 14-slide deck |
| [One-page pitch (PDF)](docs/EdgeMind_Pitch.pdf) | Short summary |

## Features

- **Private document Q&A** across text, Markdown and PDF files, with source citations
- **Summarisation** of long documents at a chosen length
- **Local semantic search** using a persistent on-disk vector index
- **Power-aware execution** that adapts NPU performance mode and batch size to battery state
- **Ordered hardware fallback:** NPU (QNN) -> GPU (DirectML) -> CPU, so it runs on any machine
- **Compiled-graph caching** via ONNX Runtime EP context files for faster launches
- **Offline by design:** no network calls, no telemetry

## Why Snapdragon-powered HP PCs

| Platform strength | How EdgeMind uses it |
|---|---|
| Hexagon NPU (up to 45 TOPS on Snapdragon X series) | Embedding and speech models run on the NPU, keeping the CPU free and power low |
| Oryon ARM64 CPU | ARM64-native code avoids x64 emulation overhead |
| Long battery life | Enables background indexing without heavy battery drain |
| HP business PCs | Enterprise manageability and hardware-backed security (BitLocker/TPM) |

## Architecture

```
+-----------------------------------------------------------+
|  Presentation layer (CLI today, ARM64-native app planned)  |
+-----------------------------------------------------------+
|  Orchestrator + power-aware policy engine                  |
+---------------------+--------------------+-----------------+
|  Speech service     |  Retrieval service |  Language       |
|  (planned)          |  chunk, embed,     |  service        |
|                     |  vector search     |  (optional LLM) |
+---------------------+--------------------+-----------------+
|  ONNX Runtime: QNN EP (NPU) | DirectML (GPU) | CPU fallback |
+-----------------------------------------------------------+
|  Snapdragon X: Hexagon NPU | Adreno GPU | Oryon CPU        |
+-----------------------------------------------------------+
```

## Quick start

### Any machine (CPU fallback, no model download needed)

```bash
pip install -r requirements.txt onnxruntime tokenizers
python edgemind.py info
python edgemind.py ingest sample_data
python edgemind.py ask "How does EdgeMind protect privacy?"
python edgemind.py summarize sample_data/overview.md --sentences 3
python edgemind.py bench --chunks 200
```

### Snapdragon X HP PC (ARM64 Python, NPU path)

```bash
pip install -r requirements.txt onnxruntime-qnn tokenizers
python edgemind.py info
python edgemind.py --model models/embedder.onnx --tokenizer models/tokenizer.json ingest <folder>
python edgemind.py --model models/embedder.onnx --tokenizer models/tokenizer.json bench
```

`info` reports whether `QNNExecutionProvider` is available. You supply a BERT-style ONNX embedding model (for example a quantised MiniLM or BGE-small) and its `tokenizer.json`. Models are not included in this repository.

## Commands

| Command | Purpose |
|---|---|
| `info` | Show hardware, ONNX Runtime providers, battery state and active policy |
| `ingest <path>` | Index a file or folder (`.txt`, `.md`, `.pdf`) |
| `ask "<question>"` | Answer a question from indexed documents, with sources |
| `summarize <file>` | Summarise a document (`--sentences N`) |
| `bench` | Measure embedding throughput (`--chunks N`) |

Global options: `--index DIR`, `--policy performance|balanced|efficiency`, `--model`, `--tokenizer`. Use `--llm <dir>` with `ask` to enable an optional local language model.

## Power policy

| Device state | NPU performance mode | Batch size |
|---|---|---|
| Plugged in or no battery | `burst` | 32 |
| On battery, above 50% | `balanced` | 16 |
| On battery, 20% to 50% | `high_power_saver` | 8 |
| On battery, below 20% | `power_saver` | 4 |

Override with `--policy` to compare modes when benchmarking.

## Project structure

```
edgemind-snapdragon/
├── edgemind.py            # Prototype: hardware detection, policy, ORT session,
│                          # embedders, vector store, Q&A, summariser, CLI
├── tests/
│   └── test_edgemind.py   # pytest suite
├── sample_data/           # Demo documents
├── docs/                  # Proposal, presentation, pitch
├── requirements.txt
├── LICENSE
└── README.md
```

## Testing

```bash
pip install pytest
python -m pytest -q
```

The 7 tests cover policy tiers, chunking, embeddings, the index save/load round trip and summarisation.

## Project status

**Working and tested (on x86-64 CPU):** hardware and provider detection, power policy engine, chunking, hashing embedder, local vector store, extractive Q&A, summariser, CLI.

**Not yet validated:**
- NPU latency, throughput and battery impact on Snapdragon hardware. The proposal lists these as targets to measure, not as results.
- The optional local LLM path (`onnxruntime-genai`), which is experimental because that API changes between releases.

**Note on retrieval quality:** the default `HashingEmbedder` exists so the pipeline runs anywhere without a model. Use the ONNX embedder for semantic-quality retrieval.

## Roadmap

1. Validate the NPU path and record CPU vs NPU benchmarks on a Snapdragon X HP PC
2. Whisper-class speech-to-text on the NPU for meeting transcripts
3. Quantised local LLM for grounded answers with citations
4. ARM64-native Windows desktop app with background indexing
5. Enterprise packaging (MSIX, Intune) and admin policy controls

## Privacy

All processing happens on the device. Documents, embeddings and prompts are never transmitted. The index is stored in a local folder (`.edgemind_index/` by default) and can be placed on a BitLocker-protected drive.

## License

MIT. See [LICENSE](LICENSE).

## Author

Ayush ([@ayush2459](https://github.com/ayush2459))
