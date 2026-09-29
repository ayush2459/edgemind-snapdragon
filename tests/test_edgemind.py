import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import edgemind as em


def test_policy_plugged_prefers_performance():
    p = em.choose_policy({"percent": 40, "plugged": True})
    assert p.name == "performance" and p.htp_performance_mode == "burst"


def test_policy_battery_tiers():
    assert em.choose_policy({"percent": 80, "plugged": False}).name == "balanced"
    assert em.choose_policy({"percent": 35, "plugged": False}).htp_performance_mode == "high_power_saver"
    assert em.choose_policy({"percent": 10, "plugged": False}).htp_performance_mode == "power_saver"


def test_policy_override_and_no_battery():
    assert em.choose_policy(None).name == "performance"
    assert em.choose_policy({"percent": 90, "plugged": True}, "efficiency").name == "efficiency"


def test_chunking_overlap_and_paragraphs():
    text = "Short paragraph.\n\n" + " ".join(f"w{i}" for i in range(300))
    chunks = em.chunk_text(text, max_words=100, overlap=10)
    assert chunks[0] == "Short paragraph." and len(chunks) >= 4


def test_hashing_embedder_normalised_and_similar():
    e = em.HashingEmbedder()
    v = e.encode(["npu battery efficiency", "npu battery efficiency", "cooking pasta recipe"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert v[0] @ v[1] > v[0] @ v[2]


def test_ingest_search_roundtrip(tmp_path):
    e, store = em.HashingEmbedder(), em.VectorStore()
    sample = Path(__file__).resolve().parent.parent / "sample_data"
    assert em.ingest_path(sample, e, store) > 0
    store.save(tmp_path)
    loaded = em.VectorStore.load(tmp_path)
    res = em.answer_question("Does any document text leave the device?", e, loaded)
    assert res["sources"] and res["mode"] == "extractive"


def test_extractive_summary_length():
    text = ("EdgeMind runs models on the NPU. The NPU saves battery. "
            "Privacy is preserved locally. Users can search documents. Cats are nice animals.")
    assert len(em.split_sentences(em.extractive_summary(text, n=2))) == 2
