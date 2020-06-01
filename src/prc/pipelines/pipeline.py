"""General pipelines."""
from kedro.pipeline import Pipeline, node
from prc.nodes.de import prm
from prc.nodes.ds import fea

de_pipeline = Pipeline(
    [
        node(
            prm.merge_detailed_info,
            inputs=["raw_calendar", "raw_detail_listing"],
            outputs="mem_raw_spine",
            name="merge_detailed_info",
        ),
        node(
            prm.process_date_cols,
            inputs="mem_raw_spine",
            outputs="mem_prm_date_processed",
            name="process_date_cols",
        ),
        node(
            prm.process_price_cols,
            inputs="mem_prm_date_processed",
            outputs="mem_prm_price_processed",
            name="process_price_cols",
        ),
        node(
            prm.drop_outlier,
            inputs="mem_prm_price_processed",
            outputs="mem_prm_output",
            name="drop_outlier",
        ),
    ],
    tags=["de"],
)


fea_pipeline = Pipeline(
    [
        node(
            fea.process_log_cols,
            inputs=["mem_prm_output", "params:log_cols"],
            outputs="mem_fea_logged",
            name="process_log_cols",
        ),
        node(
            fea.drop_useless_cols,
            inputs=["mem_fea_logged", "params:fea_drop_columns"],
            outputs="fea_output",
            name="drop_useless_cols",
        ),
    ],
    tags=["fea"],
)
