import dspy
import mlflow

from generate_lie import GenerateLie


def main():
    mlflow.dspy.autolog()

    llm = dspy.LM("databricks/databricks-gpt-oss-120b", cache=False, temperature=1.0)
    # llm = dspy.LM(
    #    "databricks/databricks-qwen3-next-80b-a3b-instruct",
    #    cache=False,
    #    temperature=1.0,
    # )
    dspy.configure(lm=llm, adapter=dspy.JSONAdapter())

    p = GenerateLie()
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
