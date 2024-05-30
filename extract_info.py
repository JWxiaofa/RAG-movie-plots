from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

client = MongoClient(
        "mongodb+srv://jiaweiwu:99992199q@cluster0.msd6xkt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
print("------connect successfully-------")
db = client['movie_plots']
collection = db['movie']


def get_result(query: str, limit: int):
    '''
    embed the query and retrieve from database using atlas search
    :param query: user input text
    :param limit: how many results you want to show
    :return: a certain numbers of relevant objects retrieved from database
    '''
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embded = model.encode(query).tolist()

    results = collection.aggregate([
        {"$vectorSearch": {
            "queryVector": query_embded,
            "path": "embedding",
            "numCandidates": 100,
            "limit":limit,
            "index": "plot",
        }}
    ]);

    return results


def get_movie_titles(query: str, limit: int) -> list:
    '''
    :param query: user input text
    :param limit: how many results you want to show
    :return: a list of movie titles
    '''
    results = get_result(query, limit)
    movie_titles = [document['Title'] for document in results]
    return movie_titles


def get_movie_plots(query: str, limit: int) -> list:
    '''
    :param query: user input text
    :param limit: how many results you want to show
    :return: a list of plots
    '''
    results = get_result(query, limit)
    plots = [document['Plot'] for document in results]
    return plots


if __name__ == '__main__':
    query = "what movies are about quebec?"
    res = get_result(query, limit=3)
    print(type(res))
    # for document in res:
    #     print(f'Movie Name: {document["Title"]},\nMovie Plot: {document["Plot"]}\n')

    # titles = get_movie_titles(query, limit=1)
    # print(titles)
    # plot = get_movie_plots(query, limit=1)
    # print(plot)

