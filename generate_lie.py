import csv
import random
from typing import List
import dspy
import mlflow


class LieGenerate(dspy.Signature):
    """与えられたキーワードからカスの嘘を生成します"""

    keyword: str = dspy.InputField(desc="キーワード")
    lie: str = dspy.OutputField(desc="生成されたカスの嘘")


class GenerateLie(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(LieGenerate)

    def forward(self, keyword):
        return self.extractor(keyword=keyword)


def create_lie_metric():
    class StyleEvaluation(dspy.Signature):
        """応答のスタイルを評価"""

        response = dspy.InputField(desc="評価対象の応答")
        criteria = dspy.InputField(desc="評価基準")
        score = dspy.OutputField(desc="スコア（0-20）", format=int)
        explanation = dspy.OutputField(desc="詳細な項目毎のスコアとその理由")

    evaluator = dspy.ChainOfThought(StyleEvaluation)

    def lie_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        criteria = """
        以下の基準で0-20点で評価してください:
        1. キーワードに沿った嘘となっているか（2点）
        2. 20-60文字で、「〇〇は、〇〇」や「〇〇はね、〇〇」という形式となっているか（2点）
        3. あからさまな嘘ではなく、一見すると嘘とわからないようなもっともらしい嘘となっているか（6点）
            - 魔法やドラゴン・妖精といった空想的存在に基づいた嘘ではないか
            - 宇宙人や火星といった存在が確認されていないものや現在の科学水準で達成できない事柄を含んでいないか
            - 食品が必修科目として学ばれるなど、キーワードと嘘の内容が乖離していないか
            - 存在しない税が免除されるといった一つの文の中に複数の嘘が入っている場合は減点
        4. 余計な解説や文が入っておらず、嘘のみの出力となっているか（2点）
        5. 語頭に「ね、知ってる？」や「実は」を適切に使っているか（2点）
        6. 語尾に「なんだよ」や「なんだって」を自然に接続して適切に使っているか（2点）
            - 「らしいなんだよ」のような、不自然な接続となっている場合は減点
            - 語尾が厳密に「なんだよ」「なんだって」になっていなくても、自然に接続されている場合は得点とする
        7. 日本語として自然で読みやすいか（2点）
        8. 穏やかで落ち着いた口調か（2点）
        """

        eval_result = evaluator(response=pred.lie, criteria=criteria)
        score = min(20, max(0, float(eval_result.score))) / 20.0
        explanation = eval_result.explanation
        return dspy.Prediction(score=score, feedback=explanation)

    return lie_metric


def run_prompt_optimizer(
    trainset: List[dspy.Example], valset: List[dspy.Example], reflection_lm: dspy.LM
):
    student_program = GenerateLie()
    metric = create_lie_metric()
    optimizer = dspy.GEPA(
        metric=metric,
        auto="light",
        reflection_lm=reflection_lm,
        track_stats=True,
        track_best_outputs=True,
        num_threads=1,
    )

    compiled_program: GenerateLie = optimizer.compile(
        student_program, trainset=trainset, valset=valset
    )
    compiled_program.save("./program/generate_lie.json", save_program=False)


def main():
    mlflow.dspy.autolog(
        log_compiles=True,
        log_evals=True,
        log_traces_from_compile=True,
    )

    llm = dspy.LM("databricks/databricks-gpt-oss-120b", cache=False, temperature=1.0)
    # llm = dspy.LM(
    #    "databricks/databricks-qwen3-next-80b-a3b-instruct",
    #    cache=False,
    #    temperature=1.0,
    # )

    dspy.configure(lm=llm, adapter=dspy.JSONAdapter())

    train_examples: List[dspy.Example]

    with open("./datasets/kasu_no_uso.csv", encoding="utf_8_sig") as f:
        reader = csv.DictReader(f)
        train_examples = [
            dspy.Example(keyword=row["keyword"], lie=row["lie"]).with_inputs("keyword")
            for row in reader
        ]

    random.shuffle(train_examples)
    trainset = train_examples[: int(len(train_examples) * 0.7)]
    valset = train_examples[int(len(train_examples) * 0.3) :]

    run_prompt_optimizer(trainset, valset, llm)


if __name__ == "__main__":
    main()
