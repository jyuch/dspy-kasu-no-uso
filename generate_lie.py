import csv
import random
from typing import List
import dspy
import mlflow

from lie_quality_judge import LieQualityJudge


class LieGenerate(dspy.Signature):
    """Generate kasu no Uso (A stupid lie) from keyword"""

    keyword: str = dspy.InputField(desc="Keyword of Kasu no Uso")
    lie: str = dspy.OutputField(desc="Generated Kasu no Uso")


class GenerateLie(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(LieGenerate)

    def forward(self, keyword):
        return self.extractor(keyword=keyword)


def create_lie_metric():
    evaluator = LieQualityJudge()
    evaluator.load("./program/lie_quality_judge.json")

    def lie_metric(expected, actual, trace=None):
        result = evaluator(actual.lie)
        return result.judgment

    return lie_metric


def run_prompt_optimizer(train_examples: List[dspy.Example]):
    student_program = GenerateLie()
    metric = create_lie_metric()
    optimizer = dspy.MIPROv2(
        metric=metric,
        num_threads=1,
        auto="heavy",
    )
    compiled_program: GenerateLie = optimizer.compile(
        student_program, trainset=train_examples
    )
    compiled_program.save("./program/generate_lie.json", save_program=False)


def main():
    mlflow.dspy.autolog(
        log_compiles=True,
        log_evals=True,
        log_traces_from_compile=True,
    )

    llm = dspy.LM("databricks/databricks-gpt-oss-120b", cache=False)

    dspy.configure(lm=llm)

    train_examples: List[dspy.Example]

    with open("./datasets/kasu_no_uso.csv", encoding="utf_8_sig") as f:
        reader = csv.DictReader(f)
        train_examples = [
            dspy.Example(keyword=row["keyword"], lie=row["lie"]).with_inputs("keyword")
            for row in reader
            if row["judgment"] == "TRUE"
        ]

    random.shuffle(train_examples)
    run_prompt_optimizer(train_examples)


if __name__ == "__main__":
    main()
