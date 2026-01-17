from evaluation.result_normalizer import normalize_results
from pipelines.auto_corrector import auto_correct_pipeline
from pipelines.mongo_executor import execute_mongo_query


def run_with_correction(
    sql,
    sql_result,
    mongo_db,
    collection,
    initial_pipeline,
    fix_prompt_path
):
    """
    Execute Mongo pipeline and auto-correct if results mismatch.
    """
    mongo_result = execute_mongo_query(
        mongo_db_name=mongo_db,
        collection_name=collection,
        pipeline=initial_pipeline
    )

    mongo_result = normalize_results(mongo_result)

    if mongo_result == sql_result:
        return initial_pipeline, True

    corrected_pipeline = auto_correct_pipeline(
        sql=sql,
        sql_result=sql_result,
        mongo_pipeline=initial_pipeline,
        mongo_result=mongo_result,
        prompt_path=fix_prompt_path
    )

    mongo_result = execute_mongo_query(
        mongo_db_name=mongo_db,
        collection_name=collection,
        pipeline=corrected_pipeline
    )

    mongo_result = normalize_results(mongo_result)

    return corrected_pipeline, mongo_result == sql_result
