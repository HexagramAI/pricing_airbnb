"""General pipelines."""
from kedro.pipeline import Pipeline, node
from prc.nodes.de import prm
from prc.nodes.ds import fea

de_pipeline = Pipeline(
    [
        node(
            prm.process_date_cols,
            inputs="raw_calender",
            outputs="mem_prm_date_processed",
            name="process_date_cols",
        ),
        node(
            prm.process_price_cols,
            inputs="mem_prm_date_processed",
            outputs="mem_prm_price_processed",
            name="process_price_cols",
        ),
    ],
    tags=["de"],
)


fea_pipeline = Pipeline(
    [
        node(
            fea.process_log_cols,
            inputs=["mem_prm_price_processed", "params:log_cols"],
            outputs="fea_output",
            name="process_log_cols",
        )
    ],
    tags=["fea"],
)
