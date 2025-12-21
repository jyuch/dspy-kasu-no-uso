import csv
import random
from typing import List
import dspy
import mlflow


class LieTransform(dspy.Signature):
    """Transform Kasu no Uso (A stupid lie) tone into that of a downer older sister"""

    lie: str = dspy.InputField(desc="Kasu no Uso")
    downer_onesan_lie: str = dspy.OutputField(desc="Transformed Kasu no Uso")


class TransformLie(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(LieTransform)

    def forward(self, lie):
        return self.extractor(lie=lie)


def create_lie_metric():
    class StyleEvaluation(dspy.Signature):
        """応答のスタイルを評価"""

        response = dspy.InputField(desc="評価対象の応答")
        criteria = dspy.InputField(desc="評価基準")
        score = dspy.OutputField(desc="スコア（0-10）", format=int)
        explanation = dspy.OutputField(desc="評価理由")

    evaluator = dspy.ChainOfThought(StyleEvaluation)

    def lie_metric(expected, actual, trace=None):
        criteria = """
        以下の基準で0-10点で評価してください:
        1. 語頭に「ね、知ってる？」や「実は」を適切に使っているか（2点）
        2. 語尾に「なんだよ」「なんだって」を適切に使っているか（2点）
        3. 穏やかで落ち着いた口調か（2点）
        4. 日本語として自然で読みやすいか（2点）
        5. 与えられた嘘のトーンを変えるだけで余計な言葉を付け足していないか（2点）
            - トーンを変えるだけで、余計な言葉を付け足さない
        """

        eval_result = evaluator(response=actual.downer_onesan_lie, criteria=criteria)
        score = min(10, max(0, float(eval_result.score))) / 10.0
        return score

    return lie_metric


def run_prompt_optimizer(train_examples: List[dspy.Example]):
    student_program = TransformLie()
    metric = create_lie_metric()
    optimizer = dspy.MIPROv2(
        metric=metric,
        num_threads=1,
        auto="light",
    )
    compiled_program: TransformLie = optimizer.compile(
        student_program, trainset=train_examples
    )
    compiled_program.save(
        "./program/generate_downer_onesan_lie.json", save_program=False
    )


def main():
    mlflow.dspy.autolog(
        log_compiles=True,
        log_evals=True,
        log_traces_from_compile=True,
    )

    llm = dspy.LM("databricks/databricks-gpt-oss-120b", cache=False)

    dspy.configure(lm=llm, adapter=dspy.JSONAdapter())

    train_examples: List[dspy.Example]

    with open("./datasets/kasu_no_uso.csv", encoding="utf_8_sig") as f:
        reader = csv.DictReader(f)
        train_examples = [
            dspy.Example(
                lie=row["lie"], downer_onesan_lie=row["downer_onesan_lie"]
            ).with_inputs("lie")
            for row in reader
            if row["judgment"] == "TRUE"
        ]

    random.shuffle(train_examples)
    run_prompt_optimizer(train_examples)


if __name__ == "__main__":
    main()
