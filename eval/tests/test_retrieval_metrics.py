"""Seam S10: recall/precision/MRR/nDCG metric functions vs hand-worked values.

Every expected number below was computed by hand from the formulas in
docs/02-eval-framework.md §5 — none are derived from the code under test.
"""

import pytest

from eval.metrics import (
    dcg_from_grades,
    mean_reciprocal_rank_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

RANKED_5 = ["d1", "d2", "d3", "d4", "d5"]
RELEVANT_2 = {"d2", "d4"}


class TestRecallAtK:
    def test_both_relevant_within_top_5(self) -> None:
        assert recall_at_k(RANKED_5, RELEVANT_2, 5) == 1.0

    def test_one_of_two_relevant_within_top_3(self) -> None:
        assert recall_at_k(RANKED_5, RELEVANT_2, 3) == 0.5

    def test_no_relevant_hits_in_top_1(self) -> None:
        assert recall_at_k(RANKED_5, RELEVANT_2, 1) == 0.0

    def test_empty_relevant_set_is_zero(self) -> None:
        assert recall_at_k(RANKED_5, set(), 5) == 0.0

    def test_results_beyond_k_are_ignored(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"c"}, 2) == 0.0


class TestPrecisionAtK:
    def test_two_hits_in_top_5(self) -> None:
        assert precision_at_k(RANKED_5, RELEVANT_2, 5) == pytest.approx(0.4)

    def test_one_hit_in_top_3_divides_by_k_not_by_results(self) -> None:
        assert precision_at_k(RANKED_5, RELEVANT_2, 3) == pytest.approx(1 / 3)


class TestReciprocalRankAtK:
    def test_first_relevant_at_rank_4(self) -> None:
        ranked = ["a", "b", "c", "d", "e"]
        assert reciprocal_rank_at_k(ranked, {"d"}, 5) == pytest.approx(0.25)

    def test_relevant_hit_beyond_k_scores_zero(self) -> None:
        ranked = ["a", "b", "c", "d", "e"]
        assert reciprocal_rank_at_k(ranked, {"d"}, 3) == 0.0

    def test_first_position_scores_one(self) -> None:
        assert reciprocal_rank_at_k(["a", "b"], {"a"}, 2) == 1.0


class TestMeanReciprocalRankAtK:
    def test_mean_of_half_one_and_zero(self) -> None:
        rankings = [["a", "b", "c"], ["x", "y", "z"], ["p", "q", "r"]]
        relevance = [{"b"}, {"z"}, {"m"}]  # rrs: 1/2, 1/3, 0
        assert mean_reciprocal_rank_at_k(rankings, relevance, 3) == pytest.approx(
            (0.5 + 1 / 3 + 0) / 3
        )

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            mean_reciprocal_rank_at_k([["a"]], [{"a"}, {"b"}], 3)

    def test_empty_query_set_is_zero(self) -> None:
        assert mean_reciprocal_rank_at_k([], [], 3) == 0.0


class TestDcgFromGrades:
    def test_hand_worked_sequence(self) -> None:
        # 3/1 + 0/log2(3) + 1/log2(4) = 3 + 0 + 0.5
        assert dcg_from_grades([3.0, 0.0, 1.0]) == pytest.approx(3.5)


class TestNdcgAtK:
    def test_graded_relevance_hand_worked(self) -> None:
        # DCG   = 3/log2(2) + 0 + 1/log2(4)            = 3.5
        # IDCG  = 3/log2(2) + 1/log2(3) (ideal [3, 1])  = 3.63093...
        # nDCG  = 3.5 / 3.63093...                      = 0.96394...
        ranked = ["d1", "d2", "d3"]
        grades = {"d1": 3.0, "d3": 1.0}
        assert ndcg_at_k(ranked, grades, 3) == pytest.approx(0.963940, abs=1e-5)

    def test_single_binary_hit_at_rank_3_is_exactly_half(self) -> None:
        # DCG = 1/log2(4) = 0.5; IDCG = 1/log2(2) = 1.0
        assert ndcg_at_k(["a", "b", "c"], {"c": 1.0}, 3) == 0.5

    def test_perfect_ranking_is_one(self) -> None:
        ranked = ["d1", "d2", "d3"]
        grades = {"d1": 3.0, "d2": 2.0}
        assert ndcg_at_k(ranked, grades, 3) == pytest.approx(1.0)

    def test_unretrieved_relevant_doc_lowers_score(self) -> None:
        # gold has two relevant docs; only the lesser one retrieved at rank 2.
        # DCG  = 0 + 1/log2(3) + 0                     = 0.63093...
        # IDCG = 3/log2(2) + 1/log2(3) (ideal [3, 1])  = 3.63093...
        # nDCG = 0.63093... / 3.63093...               = 0.17377...
        ranked = ["a", "b", "c"]
        grades = {"b": 1.0, "gold": 3.0}
        assert ndcg_at_k(ranked, grades, 3) == pytest.approx(0.173765, abs=1e-5)

    def test_empty_grade_map_is_zero(self) -> None:
        assert ndcg_at_k(["a", "b"], {}, 2) == 0.0

    def test_truncation_to_k_ignores_tail(self) -> None:
        assert ndcg_at_k(["a", "b", "c"], {"c": 1.0}, 2) == 0.0


class TestValidation:
    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_raises_everywhere(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            recall_at_k(["a"], {"a"}, k)
        with pytest.raises(ValueError, match="k must be >= 1"):
            precision_at_k(["a"], {"a"}, k)
        with pytest.raises(ValueError, match="k must be >= 1"):
            reciprocal_rank_at_k(["a"], {"a"}, k)
        with pytest.raises(ValueError, match="k must be >= 1"):
            mean_reciprocal_rank_at_k([["a"]], [{"a"}], k)
        with pytest.raises(ValueError, match="k must be >= 1"):
            ndcg_at_k(["a"], {"a": 1.0}, k)
