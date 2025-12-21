import csv
import random
from typing import List
import dspy
import mlflow


class JudgmentLieQuality(dspy.Signature):
    """Assess the quality of Kasu no Uso (A stupid lie)"""

    lie: str = dspy.InputField(desc="Kasu no Uso")
    judgment: bool = dspy.OutputField(desc="Quality of Kasu no Uso")


class LieQualityJudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(JudgmentLieQuality)

    def forward(self, lie):
        return self.extractor(lie=lie)


def jadge_metric(expected, actual, trace=None):
    return expected.judgment == actual.judgment


def run_prompt_optimizer(train_examples: List[dspy.Example]):
    student_program = LieQualityJudge()
    optimizer = dspy.MIPROv2(
        metric=jadge_metric,
        num_threads=1,
        auto="heavy",
    )
    compiled_program: LieQualityJudge = optimizer.compile(
        student_program, trainset=train_examples
    )
    compiled_program.save("./program/lie_quality_judge.json", save_program=False)


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
            dspy.Example(
                lie=row["lie"], judgment=bool(row["judgment"] == "TRUE")
            ).with_inputs("lie")
            for row in reader
        ]

    random.shuffle(train_examples)
    run_prompt_optimizer(train_examples)


if __name__ == "__main__":
    main()
