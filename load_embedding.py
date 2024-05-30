import pandas as pd
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

file_name = "data/wiki_movie_plots_deduped.csv"

df = pd.read_csv(file_name)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# plot = "A bartender is working at a saloon, serving drinks to customers. After he fills a stereotypically Irish man's bucket with beer, Carrie Nation and her followers burst inside. They assault the Irish man, pulling his hat over his eyes and then dumping the beer over his head. The group then begin wrecking the bar, smashing the fixtures, mirrors, and breaking the cash register. The bartender then sprays seltzer water in Nation's face before a group of policemen appear and order everybody to leave.[1]"
# print(model.encode(plot))

df['embedding'] = df['Plot'].apply(lambda x: model.encode([x]).tolist()[0])
data = df.to_dict('records')

# Load all data entries to MongoDB in batch
for i in range(0, len(data), 100):
    client = MongoClient(
        "mongodb+srv://jiaweiwu:99992199q@cluster0.msd6xkt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

    db = client['movie_plots']
    collection = db['movie']

    to_insert = data[i: i+100]
    collection.insert_many(to_insert)

