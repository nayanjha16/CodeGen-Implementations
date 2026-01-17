import json
from llm.llm_client import call_llm


def auto_correct_pipeline(
    sql: str,
    sql_result,
    mongo_pipeline,
    mongo_result,
    prompt_path: str,
    max_retries: int = 3
):
    """
    Retry MongoDB pipeline generation using LLM feedback.
    """
    with open(prompt_path) as f:
        template = f.read()

    current_pipeline = mongo_pipeline

    for attempt in range(max_retries):
        prompt = template.format(
            sql=sql,
            sql_result=sql_result,
            mongo_pipeline=current_pipeline,
            mongo_result=mongo_result
        )

        raw = call_llm(prompt)

        try:
            corrected_pipeline = json.loads(raw)
            return corrected_pipeline
        except json.JSONDecodeError:
            continue

    raise RuntimeError("Failed to auto-correct Mongo pipeline")
