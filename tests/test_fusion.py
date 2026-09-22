from app.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_rewards_agreement():
    # 'b' appears in BOTH lists; 'a' and 'c' each appear in only one.
    lists = {
        "bm25": [("a", 9.0), ("b", 8.0)],
        "dense": [("c", 0.9), ("b", 0.8)],
    }
    fused = reciprocal_rank_fusion(lists, k=60)
    ids = [f.chunk_id for f in fused]
    assert ids[0] == "b"  # presence in both lists wins
    top = fused[0]
    assert top.ranks == {"bm25": 2, "dense": 2}


def test_rrf_uses_rank_not_score_scale():
    # Wildly different score scales must not matter — only ranks.
    a = {"x": [("p", 1e6), ("q", 1.0)]}
    b = {"y": [("p", 0.001), ("q", 0.0009)]}
    fused = reciprocal_rank_fusion({**a, **b}, k=60)
    assert fused[0].chunk_id == "p"


def test_rrf_single_list():
    fused = reciprocal_rank_fusion({"only": [("a", 1.0), ("b", 0.5)]}, k=60)
    assert [f.chunk_id for f in fused] == ["a", "b"]
