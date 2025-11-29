
from text2sql_model import Text2SQLModel
from utils.rag import RAGRetriever
from utils.dataset_loader import load_mini_dataset
from evaluation.mini_eval import run_mini_eval

FEWSHOT = """Example fewshot here"""

def main():
    model = Text2SQLModel()
    mini = load_mini_dataset()
    retriever = RAGRetriever(mini)
    acc, _ = run_mini_eval(model, retriever, mini, FEWSHOT)
    print("Mini accuracy:", acc)

if __name__ == "__main__":
    main()
