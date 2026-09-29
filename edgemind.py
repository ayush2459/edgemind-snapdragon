#!/usr/bin/env python3
"""
EdgeMind - private, on-device AI workspace copilot for Snapdragon-powered HP PCs.

What this file contains (everything in one place):
  1. Hardware + power detection (ARM64, ONNX Runtime providers, battery state)
  2. Power-aware policy engine that tunes the Qualcomm NPU (HTP) performance mode
  3. ONNX Runtime session builder: QNN (NPU) -> DirectML (GPU) -> CPU fallback,
     with EP context caching to cut model start-up time
  4. Embedders: OnnxEmbedder (NPU-accelerated) and HashingEmbedder (no model needed)
  5. Local vector store (NumPy, persisted to disk) and document chunking
  6. Question answering + summarisation (extractive by default, optional local LLM)
  7. CLI: info | ingest | ask | summarize | bench

Everything runs locally. No document text ever leaves the device.

Usage:
  python edgemind.py info
  python edgemind.py ingest sample_data
  python edgemind.py ask "How does EdgeMind protect privacy?"
  python edgemind.py summarize sample_data/overview.md --sentences 3
  python edgemind.py bench --chunks 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__version__ = "0.1.0"

DEFAULT_INDEX_DIR = Path(".edgemind_index")
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}

STOPWORDS = set(
    "a an and are as at be but by for from has have how i in into is it its of on or "
    "that the their this to was were what when where which who why will with you your "
    "can do does not so if then than they them we our about also more most such".split()
)


# --------------------------------------------------------------------------- #
# 1. Hardware and power detection
# --------------------------------------------------------------------------- #
def is_arm64() -> bool:
    """True on Windows-on-ARM / Snapdragon X class machines."""
    return platform.machine().lower() in ("arm64", "aarch64")


def available_providers() -> List[str]:
    """ONNX Runtime execution providers installed on this machine."""
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def battery_state() -> Optional[Dict[str, float]]:
    """Battery percentage and plug state, or None on desktops / when psutil is missing."""
    try:
        import psutil

        b = psutil.sensors_battery()
        if b is None:
            return None
        return {"percent": float(b.percent), "plugged": bool(b.power_plugged)}
    except Exception:
        return None


def system_report() -> Dict[str, object]:
    providers = available_providers()
    return {
        "edgemind_version": __version__,
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "arm64_native": is_arm64(),
        "python": platform.python_version(),
        "onnxruntime_providers": providers,
        "npu_available": "QNNExecutionProvider" in providers,
        "gpu_directml_available": "DmlExecutionProvider" in providers,
        "battery": battery_state(),
    }


# --------------------------------------------------------------------------- #
# 2. Power-aware policy engine
# --------------------------------------------------------------------------- #
@dataclass
class PowerPolicy:
    """Maps device power state to NPU tuning. Values follow the QNN EP option
    `htp_performance_mode` (burst, balanced, high_power_saver, power_saver, ...)."""

    name: str
    htp_performance_mode: str
    batch_size: int
    reason: str


def choose_policy(battery: Optional[Dict[str, float]], override: Optional[str] = None) -> PowerPolicy:
    """Pick an execution policy from battery state. `override` may force a mode."""
    table = {
        "performance": PowerPolicy("performance", "burst", 32, "forced: performance"),
        "balanced": PowerPolicy("balanced", "balanced", 16, "forced: balanced"),
        "efficiency": PowerPolicy("efficiency", "power_saver", 4, "forced: efficiency"),
    }
    if override in table:
        return table[override]
    if battery is None or battery["plugged"]:
        return PowerPolicy("performance", "burst", 32, "plugged in or no battery: favour speed")
    pct = battery["percent"]
    if pct > 50:
        return PowerPolicy("balanced", "balanced", 16, f"on battery ({pct:.0f}%): balanced")
    if pct > 20:
        return PowerPolicy("efficiency", "high_power_saver", 8, f"on battery ({pct:.0f}%): save power")
    return PowerPolicy("efficiency", "power_saver", 4, f"low battery ({pct:.0f}%): minimum power")


# --------------------------------------------------------------------------- #
# 3. ONNX Runtime session builder (NPU -> GPU -> CPU)
# --------------------------------------------------------------------------- #
def build_session(model_path: str, policy: PowerPolicy, qnn_backend: str = "QnnHtp.dll",
                  cache_dir: Optional[str] = None):
    """Create an InferenceSession that prefers the Snapdragon NPU.

    On a Snapdragon X machine with `onnxruntime-qnn` installed, the QNN execution
    provider (HTP backend) runs the model on the Hexagon NPU. Elsewhere the same
    code falls back to DirectML or CPU, so development works on any machine.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    if cache_dir:
        # Cache compiled NPU graph so later launches skip re-compilation.
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        so.add_session_config_entry("ep.context_enable", "1")
        so.add_session_config_entry(
            "ep.context_file_path", str(Path(cache_dir) / (Path(model_path).stem + "_ctx.onnx"))
        )

    avail = ort.get_available_providers()
    providers: List = []
    if "QNNExecutionProvider" in avail:
        providers.append(("QNNExecutionProvider", {
            "backend_path": qnn_backend,
            "htp_performance_mode": policy.htp_performance_mode,
            "enable_htp_fp16_precision": "1",
        }))
    if "DmlExecutionProvider" in avail:
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    return session, [p if isinstance(p, str) else p[0] for p in providers]


# --------------------------------------------------------------------------- #
# 4. Embedders
# --------------------------------------------------------------------------- #
def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class HashingEmbedder:
    """Dependency-free embedder (hashing trick, unigrams + bigrams).

    Lets the whole pipeline run and be tested with no model download. Swap for
    OnnxEmbedder to get semantic quality and NPU acceleration."""

    name = "hashing"

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.provider = "CPU (numpy)"

    def _bucket(self, token: str) -> Tuple[int, float]:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        return h % self.dim, 1.0 if (h >> 64) & 1 else -1.0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            words = [w for w in tokenize_words(t) if w not in STOPWORDS]
            grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
            for g in grams:
                idx, sign = self._bucket(g)
                out[i, idx] += sign
        return _l2(out)


class OnnxEmbedder:
    """Sentence embedder running through ONNX Runtime (QNN/NPU when available).

    Expects a BERT-style ONNX model (e.g. a quantised MiniLM / BGE-small) and its
    `tokenizer.json`. Mean-pools token states and L2-normalises."""

    name = "onnx"

    def __init__(self, model_path: str, tokenizer_path: str, policy: PowerPolicy,
                 max_len: int = 256, cache_dir: Optional[str] = ".edgemind_cache"):
        from tokenizers import Tokenizer

        self.policy = policy
        self.session, self.providers = build_session(model_path, policy, cache_dir=cache_dir)
        self.provider = self.providers[0]
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(length=None)
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        chunks = []
        bs = self.policy.batch_size
        for s in range(0, len(texts), bs):
            enc = self.tok.encode_batch(list(texts[s:s + bs]))
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            chunks.append(pooled.astype(np.float32))
        return _l2(np.vstack(chunks)) if chunks else np.zeros((0, 1), dtype=np.float32)


# --------------------------------------------------------------------------- #
# 5. Documents, chunking and the local vector store
# --------------------------------------------------------------------------- #
def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, max_words: int = 120, overlap: int = 20) -> List[str]:
    """Paragraph-aware chunking with a small word overlap between windows."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    for p in paras:
        words = p.split()
        if len(words) <= max_words:
            chunks.append(p)
            continue
        step = max_words - overlap
        for s in range(0, len(words), step):
            chunks.append(" ".join(words[s:s + max_words]))
            if s + max_words >= len(words):
                break
    return chunks


class VectorStore:
    """Tiny persistent cosine-similarity store. Fine for tens of thousands of chunks."""

    def __init__(self, matrix: Optional[np.ndarray] = None, items: Optional[List[Dict]] = None):
        self.matrix = matrix if matrix is not None else np.zeros((0, 0), dtype=np.float32)
        self.items: List[Dict] = items or []

    def add(self, vectors: np.ndarray, items: List[Dict]) -> None:
        if not len(items):
            return
        self.matrix = vectors if self.matrix.size == 0 else np.vstack([self.matrix, vectors])
        self.items.extend(items)

    def search(self, qvec: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        if self.matrix.size == 0:
            return []
        scores = self.matrix @ qvec.reshape(-1)
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.items[i]) for i in top]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.matrix)
        (directory / "chunks.json").write_text(json.dumps(self.items), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        vp, cp = directory / "vectors.npy", directory / "chunks.json"
        if not vp.exists() or not cp.exists():
            return cls()
        return cls(np.load(vp), json.loads(cp.read_text(encoding="utf-8")))


def ingest_path(path: Path, embedder, store: VectorStore) -> int:
    files = [path] if path.is_file() else sorted(
        f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED_SUFFIXES
    )
    added = 0
    for f in files:
        chunks = chunk_text(read_document(f))
        if not chunks:
            continue
        store.add(embedder.encode(chunks), [{"source": str(f), "text": c} for c in chunks])
        added += len(chunks)
    return added


# --------------------------------------------------------------------------- #
# 6. Summarisation and question answering
# --------------------------------------------------------------------------- #
def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.split()) >= 4]


def extractive_summary(text: str, n: int = 3, focus: Optional[str] = None) -> str:
    """Frequency-based extractive summary; `focus` biases scoring toward a query."""
    sents = split_sentences(text)
    if len(sents) <= n:
        return " ".join(sents)
    freq: Dict[str, int] = {}
    for w in tokenize_words(text):
        if w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    focus_words = {w for w in tokenize_words(focus or "") if w not in STOPWORDS}
    scored = []
    for i, s in enumerate(sents):
        ws = [w for w in tokenize_words(s) if w not in STOPWORDS]
        score = sum(freq.get(w, 0) for w in ws) / (len(ws) ** 0.5 + 1e-9)
        score += 5.0 * len(focus_words & set(ws))
        scored.append((score, i))
    keep = sorted(i for _, i in sorted(scored, reverse=True)[:n])
    return " ".join(sents[i] for i in keep)


class LocalLLM:
    """Optional generative model via onnxruntime-genai (e.g. a 4-bit Phi-3.5-mini or
    Llama-3.2-3B build for Snapdragon X). Experimental: the genai API changes between
    releases, so any failure falls back to extractive answers."""

    def __init__(self, model_dir: str):
        import onnxruntime_genai as og

        self.og = og
        self.model = og.Model(model_dir)
        self.tokenizer = og.Tokenizer(self.model)

    def generate(self, prompt: str, max_length: int = 512) -> str:
        og = self.og
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=max_length)
        gen = og.Generator(self.model, params)
        gen.append_tokens(self.tokenizer.encode(prompt))
        out = []
        while not gen.is_done():
            gen.generate_next_token()
            out.append(self.tokenizer.decode(gen.get_next_tokens()[0]))
        return "".join(out).strip()


def answer_question(question: str, embedder, store: VectorStore, k: int = 4,
                    llm: Optional[LocalLLM] = None) -> Dict[str, object]:
    hits = store.search(embedder.encode([question])[0], k=k)
    if not hits:
        return {"answer": "The index is empty. Run `ingest` first.", "sources": []}
    context = "\n\n".join(h["text"] for _, h in hits)
    sources = sorted({h["source"] for _, h in hits})
    if llm is not None:
        try:
            prompt = (f"Answer using only the context.\n\nContext:\n{context}\n\n"
                      f"Question: {question}\nAnswer:")
            return {"answer": llm.generate(prompt), "sources": sources, "mode": "llm"}
        except Exception as exc:  # fall back rather than fail
            print(f"[warn] LLM failed ({exc}); using extractive answer", file=sys.stderr)
    return {"answer": extractive_summary(context, n=3, focus=question),
            "sources": sources, "mode": "extractive"}


# --------------------------------------------------------------------------- #
# 7. CLI
# --------------------------------------------------------------------------- #
def make_embedder(args, policy: PowerPolicy):
    if getattr(args, "model", None) and getattr(args, "tokenizer", None):
        return OnnxEmbedder(args.model, args.tokenizer, policy)
    return HashingEmbedder()


def cmd_info(args) -> int:
    rep = system_report()
    pol = choose_policy(rep["battery"], args.policy)
    rep["active_policy"] = asdict(pol)
    print(json.dumps(rep, indent=2))
    if not rep["arm64_native"]:
        print("\n[note] Not running on ARM64. Use ARM64 Python + onnxruntime-qnn on a "
              "Snapdragon X HP PC to enable NPU execution.", file=sys.stderr)
    return 0


def cmd_ingest(args) -> int:
    policy = choose_policy(battery_state(), args.policy)
    emb = make_embedder(args, policy)
    store = VectorStore.load(Path(args.index))
    t0 = time.perf_counter()
    n = ingest_path(Path(args.path), emb, store)
    store.save(Path(args.index))
    print(f"Indexed {n} chunks in {time.perf_counter() - t0:.2f}s "
          f"(embedder={emb.name}, provider={emb.provider}, policy={policy.name})")
    return 0


def cmd_ask(args) -> int:
    policy = choose_policy(battery_state(), args.policy)
    emb = make_embedder(args, policy)
    llm = LocalLLM(args.llm) if args.llm else None
    res = answer_question(args.question, emb, VectorStore.load(Path(args.index)), args.k, llm)
    print(res["answer"])
    if res["sources"]:
        print("\nSources:\n  " + "\n  ".join(res["sources"]))
    return 0


def cmd_summarize(args) -> int:
    print(extractive_summary(read_document(Path(args.file)), n=args.sentences))
    return 0


def cmd_bench(args) -> int:
    policy = choose_policy(battery_state(), args.policy)
    emb = make_embedder(args, policy)
    texts = [("Snapdragon NPU on-device inference battery efficiency privacy " * 12) + str(i)
             for i in range(args.chunks)]
    emb.encode(texts[:4])  # warm-up
    t0 = time.perf_counter()
    emb.encode(texts)
    dt = time.perf_counter() - t0
    print(json.dumps({
        "embedder": emb.name, "provider": emb.provider, "policy": policy.name,
        "chunks": args.chunks, "total_s": round(dt, 4),
        "chunks_per_s": round(args.chunks / dt, 1),
        "avg_ms_per_chunk": round(1000 * dt / args.chunks, 3),
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="edgemind", description=__doc__.split("\n")[1])
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--index", default=str(DEFAULT_INDEX_DIR), help="index directory")
    p.add_argument("--policy", choices=["performance", "balanced", "efficiency"],
                   help="force a power policy instead of auto-detecting")
    p.add_argument("--model", help="ONNX embedding model (enables NPU path)")
    p.add_argument("--tokenizer", help="tokenizer.json for the embedding model")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="show hardware, providers and active policy").set_defaults(fn=cmd_info)
    s = sub.add_parser("ingest", help="index a file or folder")
    s.add_argument("path")
    s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("ask", help="ask a question over indexed documents")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=4)
    s.add_argument("--llm", help="onnxruntime-genai model directory (optional)")
    s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("summarize", help="summarise a document")
    s.add_argument("file")
    s.add_argument("--sentences", type=int, default=3)
    s.set_defaults(fn=cmd_summarize)
    s = sub.add_parser("bench", help="measure embedding throughput")
    s.add_argument("--chunks", type=int, default=200)
    s.set_defaults(fn=cmd_bench)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
