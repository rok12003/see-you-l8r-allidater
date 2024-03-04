# %%
import pandas as pd
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# %%
model = SentenceTransformer('all-MiniLM-L6-v2')

# %%
df = pd.read_csv("../data/okcupid_profiles.csv")
#df_sample = df.sample(1000).reset_index(drop = True)
df_sample = df.copy()


# %%
df_sample.columns

# %%
# All essays but essay9, which is "you should message me if..."

df_demographics = df_sample[df_sample.columns.drop(list(df_sample.filter(regex="essay")))]
essays_df = df_sample.loc[:, ["essay0", "essay1", "essay2", "essay3", "essay4", 
                   "essay5", "essay6", "essay7", "essay8"]]
essays_df = essays_df.fillna(" ").astype(str)

essays_df.loc[:, "all_essays"] = essays_df.apply(" ".join, axis = 1)

df_all = pd.concat([df_demographics, essays_df.loc[:, ["all_essays"]]], axis = 1)

# %%
dir_path = os.path.dirname(os.path.realpath(__file__))

embedding_array = np.load(os.path.join(dir_path, 'sentence_transformer_embeddings.npy'))

# %%
def compute_cosine_similarity(target_vector, vectors):
    similarities = []
    for vector in vectors:
        similarity = 1 - cosine(target_vector, vector)  # 1 - cosine distance to get cosine similarity
        similarities.append(similarity)
    return similarities

# %%
from itertools import islice

def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))

# %%
def rank_matches(input_bio, pref_gender=False, pref_age_lower=False, pref_age_higher=False, top_n=5):
    df_possible = df_all.copy()
    if pref_gender:
        df_possible = df_possible.loc[df_possible.loc[:,'sex'] == pref_gender, :]
    if pref_age_higher:
        df_possible = df_possible[df_possible.loc[:, "age"] <= pref_age_higher]
    if pref_age_lower:
        df_possible = df_possible[df_possible.loc[:, "age"] >= pref_age_lower]


    user_embeddings = model.encode(input_bio)

    other_embeddings = [embedding_array[i] for i in df_possible.index]
    # Compute the cosine similarity between the user's weighted embedding vector and all possible matches
    cosine_similarities = compute_cosine_similarity(user_embeddings, other_embeddings)
    # Recover index to match back to original dataframe
    similarity_scores = {index:score for index, score in enumerate(cosine_similarities)}
    # Sort by similarity
    ranked_similarity = dict(sorted(similarity_scores.items(), key=lambda item: item[1], reverse = True))

    top_match = take(5, ranked_similarity.items())

    bios = [essays_df.loc[t[0], "all_essays"] for t in top_match]


    return bios




