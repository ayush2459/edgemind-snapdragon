# EdgeMind

**A private, on-device AI workspace copilot optimised for Snapdragon-powered HP PCs.**

EdgeMind indexes your documents, notes and transcripts locally, then lets you ask questions,
summarise and search across them. Models run on the Snapdragon X **Hexagon NPU** through
ONNX Runtime's QNN execution provider, so it is fast, battery-friendly and fully private.
No document text leaves the device.

## Why Snapdragon + HP

| Need | How EdgeMind uses the platform |
|---|---|
| Sustained AI at low power | Embedding and speech models run on the Hexagon NPU (QNN EP, HTP backend) |
| Long battery life | Power-aware policy lowers NPU performance mode and batch size on battery |
| Fast start-up | EP context caching stores compiled NPU graphs between launches |
| Native speed | ARM64-native Python and libraries, no x64 emulation |
| Privacy and compliance | Local-only index, compatible with BitLocker/TPM and HP business security |

## Quick start

```bash
# 1. Any machine (CPU fallback, no model needed)
pip install -r requirements.txt onnxruntime tokenizers
python edgemind.py info
python edgemind.py ingest sample_data
python edgemind.py ask "How does EdgeMind protect privacy?"
python edgemind.py summarize sample_data/overview.md --sentences 3
python edgemind.py bench --chunks 200

# 2. Snapdragon X HP PC (ARM64 Python) - NPU path
pip install -r requirements.txt onnxruntime-qnn tokenizers
python edgemind.py --model models/embedder.onnx --tokenizer models/tokenizer.json ingest <folder>
python edgemind.py --model models/embedder.onnx --tokenizer models/tokenizer.json bench
```

`info` shows whether `QNNExecutionProvider` is detected. `bench` prints throughput so you can
compare CPU vs NPU and each power policy (`--policy performance|balanced|efficiency`).

## Project layout

```
edgemind.py            # the whole prototype: hardware detection, policy, ORT session,
                       # embedders, vector store, Q&A, summariser, CLI
sample_data/           # demo documents
docs/                  # proposal (PDF), presentation (PPTX), one-page pitch (PDF)
tests/                 # pytest suite (policy, chunking, embedder, ingest round-trip)
requirements.txt
```

## Optional local LLM

Point `--llm` at an `onnxruntime-genai` model folder (for example a 4-bit small language model
built for Snapdragon X) to generate answers. Without it, EdgeMind returns extractive answers.
The genai API changes between releases, so treat this path as experimental.

## Status and honesty notes

- Verified: policy engine, chunking, hashing embedder, vector store, CLI, tests (on x86-64 CPU).
- **Not yet measured on hardware:** NPU latency, throughput and battery figures. The proposal lists
  these as targets to validate on a Snapdragon X HP PC, not as results.
- The default `HashingEmbedder` exists so the pipeline runs anywhere; use the ONNX embedder for
  semantic-quality retrieval.

## Roadmap

1. Validate NPU path and record benchmarks on a Snapdragon X HP PC
2. Whisper-class speech-to-text on NPU for meeting transcripts
3. Windows desktop UI (WinUI 3 / Tauri, ARM64 native)
4. Quantised local LLM for grounded answers with citations
5. Enterprise packaging (MSIX, Intune) and admin policy controls

## Run tests

```bash
pip install pytest
python -m pytest -q
```

## License

MIT
