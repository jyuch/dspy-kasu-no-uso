import csv

import dspy
import mlflow

from generate_lie import GenerateLie, create_lie_metric
from dotenv import load_dotenv
from dspy.evaluate.evaluate import Evaluate


LLMS = [
    ("claude-haiku-4-5-20251001", "haiku-4-5"),
    ("databricks/databricks-gpt-oss-120b", "gpt-oss-120b"),
    ("databricks/databricks-gpt-oss-20b", "gpt-oss-20b"),
]


def main():
    load_dotenv()
    mlflow.dspy.autolog()

    reflection_lm = dspy.LM("claude-sonnet-4-5-20250929", cache=False, temperature=0.8)
    metric = create_lie_metric(reflection_lm=reflection_lm)

    with open("./datasets/kasu_no_uso.csv", encoding="utf_8_sig") as f:
        reader = csv.DictReader(f)
        train_examples = [
            dspy.Example(keyword=row["keyword"], lie=row["lie"]).with_inputs("keyword")
            for row in reader
        ]

    evaluate = Evaluate(
        devset=train_examples, num_threads=1, display_progress=True, display_table=0
    )

    with open("evaluate.txt", "w", encoding="utf_8") as f:
        for model, _ in LLMS:
            llm = dspy.LM(model=model, cache=False, temperature=0.8)
            with dspy.context(lm=llm, adapter=dspy.JSONAdapter()):
                for _, symbol in LLMS:
                    p = GenerateLie()
                    p.load(f"./program/generate_lie_{symbol}.json")
                    result = evaluate(p, metric=metric)
                    print(f"{model},{symbol},{result.score}", file=f)


if __name__ == "__main__":
    main()
