from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
from extract_info import get_movie_titles, get_movie_plots
import random
from llm import get_llm_output
from rouge import Rouge
random.seed(42)

movies_path = "data/wiki_movie_plots_deduped.csv"
df = pd.read_csv(movies_path)
movie_names = list(df['Title'])

dataset = load_dataset("wiki_movies")
# print(dataset)

def sample_test_set(dataset) -> list:
    '''
    filter out questions about director, release, year, casts, etc. and only keep quesions that ask movie title based on movie content.
    :param dataset: test dataset loaded from huggingface
    :return: sampled test set in format: [(question, answer), (question, answer)]
    '''
    test_queries = []
    for example in dataset["train"]:
        answer = example["answer"].rstrip("\n")
        answer = answer.split(",")
        answer = [a.strip() for a in answer]
        is_in_dataset = 0
        for name in answer:
            if name in movie_names:
                is_in_dataset = 1
                break
        if is_in_dataset:
            question = example['question']
            skip_words = [
                " act", " direct", " write", " written",
                " release", "genre", "appear", "star in",
                "sort of", "kind of", "who", "type of", "language"
            ]
            if any(skip_word in question for skip_word in skip_words):
                continue
            if "about" in question:
                test_queries.append((example['question'], answer))

    # test_set = random.sample(test_queries, 2000)
    return test_queries


def score(y_pred: list, y_true: list) -> int:
    '''
    check if any movie in gold is in predicted result
    :param y_pred: a list of predicted movie names
    :param y_true: a list of gold movie names
    :return: 0 or 1
    '''
    y_pred = [n.lower() for n in y_pred]
    for name in y_true:
        if name.lower() in y_pred:
            return 1
    return 0


def accuracy(test_set: list) -> int:
    '''
    Cauculate overall accuracy
    :param test_set: a list of tuple. [(question, answer), (question, answer)]
    :return: the accuracy
    '''
    correct = 0
    for question, ans in test_set:
        y_pred = get_movie_titles(question, limit=10)
        correct = score(y_pred, ans)
    acc = correct/len(test_set)
    return acc


def rouge_score(predict: str, gold: str) -> dict:
    '''
    Calculate rouge score between two texts
    :param predict: generated text
    :param gold: reference text
    :return: rouge scores in format: {'rouge-1': {'p': , 'r': , 'f': }, 'rouge-2': {}, 'rouge-l': {}}
    '''
    rouge = Rouge()
    # Compute the scores
    score = rouge.get_scores(predict, gold)
    return score[0]

def calc_rouge(test_set: list) -> dict:
    '''
    :param test_set: a list of tuple. [(question, answer), (question, answer)]
    :return: a dict of average Rouge scores for the whole test set.
    '''
    total_score = {'rouge-1': {'r': 0, 'p': 0, 'f': 0},
                   'rouge-2': {'r': 0, 'p': 0, 'f': 0},
                   'rouge-l': {'r': 0, 'p': 0, 'f': 0}}

    for question, answer in test_set:

        plot = get_movie_plots(question, limit=1)[0]
        title = get_movie_titles(question, limit=1)[0]
        llm_output = get_llm_output(question)

        rouge = rouge_score(llm_output, title + " " + plot)
        for total, curr in zip(total_score.items(), rouge.items()):
            total_score[total[0]]['r'] += rouge[curr[0]]['r']
            total_score[total[0]]['p'] += rouge[curr[0]]['p']
            total_score[total[0]]['f'] += rouge[curr[0]]['f']
    for value in total_score.values():
        value['r'] /= len(test_set)
        value['p'] /= len(test_set)
        value['f'] /= len(test_set)

    return total_score

if __name__ == "__main__":
    test_set = sample_test_set(dataset)[:100]
    print(accuracy(test_set))
    # print(calc_rouge(test_set))




