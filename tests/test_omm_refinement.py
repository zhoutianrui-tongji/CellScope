import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

from cellscope.omm_refinement import refine_omm_metaspots


POSITIVE_MARKERS = {"OMM_A": 1.0, "OMM_B": 0.8}
ALTERNATIVE_MARKERS = {"ER_A": 1.0, "CYTO_A": 0.8}
GENES = ["OMM_A", "OMM_B", "ER_A", "CYTO_A", "NOISE"]


def _make_bimodal_inputs(
    *,
    n_spots=240,
    positive_fraction=0.5,
    shuffle_final=False,
):
    n_positive = int(round(n_spots * positive_fraction))
    n_negative = n_spots - n_positive
    rows = []
    domain_ids = []
    cell_ids = []
    for index in range(n_spots):
        if index < n_positive:
            # Small deterministic variation avoids an artificial point mass.
            rows.append(
                [
                    18 + index % 4,
                    13 + index % 3,
                    1 + index % 2,
                    1,
                    2,
                ]
            )
            domain_ids.append(f"positive_{index // 20}")
        else:
            rows.append(
                [
                    1 + index % 2,
                    1,
                    18 + index % 4,
                    13 + index % 3,
                    2,
                ]
            )
            domain_ids.append(f"negative_{index // 20}")
        cell_ids.append(f"cell_{index % 12}")

    spot_ids = pd.Index([f"spot_{index}" for index in range(n_spots)])
    obs = pd.DataFrame(
        {
            "metaspot": spot_ids,
            "meta-domain": domain_ids,
            "cell_id": cell_ids,
            "r_mean": np.full(n_spots, 0.5),
        },
        index=spot_ids,
    )
    adata1 = ad.AnnData(
        X=sparse.csr_matrix(np.asarray(rows, dtype=np.int32)),
        obs=obs,
        var=pd.DataFrame(index=GENES),
    )
    domain_states = {
        domain: (
            "OMM"
            if str(domain).startswith("positive_")
            else "Cytosol"
        )
        for domain in pd.unique(np.asarray(domain_ids, dtype=object))
    }

    final_df = pd.DataFrame(
        {
            "row_id": np.arange(n_spots),
            "metaspot": spot_ids,
            "meta-domain": domain_ids,
            "subcellular_annotation_domain": [
                domain_states[domain] for domain in domain_ids
            ],
        }
    )
    if shuffle_final:
        final_df = final_df.sample(frac=1.0, random_state=17)
    return adata1, final_df, domain_states


def _run_refinement(adata1, final_df, domain_states):
    return refine_omm_metaspots(
        adata1,
        final_df,
        POSITIVE_MARKERS,
        ALTERNATIVE_MARKERS,
        domain_states=domain_states,
        min_measured_markers=2,
        min_informative_transcripts=2,
        min_eligible_spots=40,
        min_cells=2,
        min_component_spots=15,
        min_component_separation=1.0,
        min_bic_improvement=5.0,
        min_anchor_spots=15,
        min_anchor_cells=2,
        random_state=7,
        copy_adata=True,
    )


def test_bimodal_soft_prevalence_changes_with_signal_fraction():
    low_inputs = _make_bimodal_inputs(positive_fraction=0.25)
    high_inputs = _make_bimodal_inputs(positive_fraction=0.50)

    low_adata, _, low_meta = _run_refinement(*low_inputs)
    high_adata, _, high_meta = _run_refinement(*high_inputs)

    assert low_meta["identifiable"] is True
    assert high_meta["identifiable"] is True

    low_confirmed = (
        low_adata.obs["omm_domain_gate"].astype(str) == "confirmed"
    )
    high_confirmed = (
        high_adata.obs["omm_domain_gate"].astype(str) == "confirmed"
    )
    assert (
        np.nanmedian(
            low_adata.obs.loc[
                low_confirmed,
                "omm_reference_posterior",
            ]
        )
        > 0.95
    )
    assert (
        np.nanmedian(
            high_adata.obs.loc[
                ~high_confirmed,
                "omm_reference_posterior",
            ]
        )
        < 0.05
    )

    low_prevalence = low_meta["soft_prevalence"][
        "metaspot_soft_confirmed"
    ]
    high_prevalence = high_meta["soft_prevalence"][
        "metaspot_soft_confirmed"
    ]
    assert low_prevalence < 0.35
    assert high_prevalence > 0.45
    assert high_prevalence - low_prevalence > 0.15


def test_single_component_data_fails_closed():
    adata1, final_df, domain_states = _make_bimodal_inputs()
    # Every MetaSpot has exactly the same OMM/alternative ratio.
    identical = np.tile(
        np.asarray([[10, 10, 10, 10, 2]], dtype=np.int32),
        (adata1.n_obs, 1),
    )
    adata1.X = sparse.csr_matrix(identical)

    refined, refined_final, metadata = _run_refinement(
        adata1,
        final_df,
        domain_states,
    )

    assert metadata["identifiable"] is False
    assert "single_valued_contrast" in metadata["failure_reasons"]
    assert refined.obs["omm_reference_posterior"].isna().all()
    assert refined_final["omm_reference_posterior"].isna().all()
    assert set(refined.obs["omm_refined_state"]) == {"Unidentifiable"}
    assert (
        metadata["soft_prevalence"]["metaspot_soft_confirmed"] is None
    )


def test_shuffled_final_table_maps_by_metaspot_and_preserves_domain_label():
    adata1, final_df, domain_states = _make_bimodal_inputs(
        positive_fraction=0.4,
        shuffle_final=True,
    )
    original_index = final_df.index.copy()
    original_row_ids = final_df["row_id"].copy()
    original_domain_labels = final_df[
        "subcellular_annotation_domain"
    ].copy()

    refined, refined_final, metadata = _run_refinement(
        adata1,
        final_df,
        domain_states,
    )

    assert metadata["identifiable"] is True
    pd.testing.assert_index_equal(refined_final.index, original_index)
    pd.testing.assert_series_equal(refined_final["row_id"], original_row_ids)
    pd.testing.assert_series_equal(
        refined_final["subcellular_annotation_domain"],
        original_domain_labels,
    )

    posterior_by_spot = pd.Series(
        refined.obs["omm_reference_posterior"].to_numpy(),
        index=refined.obs["metaspot"].astype(str),
    )
    expected = refined_final["metaspot"].astype(str).map(posterior_by_spot)
    np.testing.assert_allclose(
        refined_final["omm_reference_posterior"],
        expected,
        equal_nan=True,
    )

    counts = np.asarray(refined.X.sum(axis=1)).ravel()
    soft_weight = np.nan_to_num(
        refined.obs["omm_confirmed_soft_weight"].to_numpy(dtype=float),
        nan=0.0,
    )
    expected_prevalence = float(np.dot(counts, soft_weight) / counts.sum())
    actual_prevalence = metadata["soft_prevalence"][
        "transcript_weighted_soft_confirmed"
    ]
    np.testing.assert_allclose(actual_prevalence, expected_prevalence)
