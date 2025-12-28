import os

import dspy
import mlflow

from generate_lie import GenerateLie


def main():
    mlflow.dspy.autolog()

    lm_model = os.environ["LM_MODEL"]
    llm_symbol = os.environ.get("LM_SYMBOL")

    llm = dspy.LM(lm_model, cache=False, temperature=0.8)
    dspy.configure(lm=llm, adapter=dspy.JSONAdapter())

    p = GenerateLie()
    if llm_symbol:
        p.load(f"./program/generate_lie_{llm_symbol}.json")
    else:
        p.load("./program/generate_lie.json")

    while True:
        print("> ", end="")
        keyword = input().strip()

        if not keyword:
            break

        response = p(keyword=keyword)
        print(response.lie)


if __name__ == "__main__":
    main()
