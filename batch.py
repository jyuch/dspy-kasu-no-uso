import csv

import dspy
import mlflow

from generate_lie import GenerateLie

LLMS = [
    ("claude-haiku-4-5-20251001", "haiku-4-5"),
    ("databricks/databricks-gpt-oss-120b", "gpt-oss-120b"),
    ("databricks/databricks-gpt-oss-20b", "gpt-oss-20b"),
]


KEYWORDS = [
    "チョコレート",
    "ちくわ",
    "お餅",
    "初詣",
    "おみくじ",
    "東京タワー",
    "軽トラ",
    "懐中電灯",
    "乾電池",
    "生成AI",
]


def main():
    mlflow.dspy.autolog()

    with open("result.csv", "w", newline="", encoding="utf_8_sig") as csvfile:
        fieldnames = ["symbol", "keyword", "lie"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model, symbol in LLMS:
            llm = dspy.LM(model=model, cache=False, temperature=0.8)
            with dspy.context(lm=llm, adapter=dspy.JSONAdapter()):
                p = GenerateLie()
                p.load(f"./program/generate_lie_{symbol}.json")

                for keyword in KEYWORDS:
                    try:
                        response = p(keyword=keyword)
                        writer.writerow(
                            {"symbol": symbol, "keyword": keyword, "lie": response.lie}
                        )
                        print(f"{symbol} {keyword} {response.lie}")
                    except dspy.utils.exceptions.AdapterParseError:
                        print(f"Error in {symbol} {keyword}")
                        writer.writerow(
                            {"symbol": symbol, "keyword": keyword, "lie": "error"}
                        )


if __name__ == "__main__":
    main()
