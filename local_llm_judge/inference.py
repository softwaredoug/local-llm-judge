import pickle
import pandas as pd
from local_llm_judge import eval_agent
from local_llm_judge.wands_data import queries_sample
from local_llm_judge.log_stdout import enable as log_enable
import inspect
import logging
import functools
import sys
import argparse


logger = logging.getLogger(__name__)


log_enable(__name__)
log_enable("local_llm_judge")


class FeatureCache:
    """Remember the LLM evals we've already computed"""

    def __init__(self, overwrite=False):
        self.overwrite = overwrite
        self.path = "data/feature_cache.pkl"
        try:
            self.cache = pd.read_pickle(self.path)
        except FileNotFoundError:
            self.cache = {}
        logger.info(f"Loaded {len(self.cache)} cached option pair evals")

    def compute_feature(self, feature_fn, feature_name, query, option_lhs, option_rhs):
        key = (feature_name, query, option_lhs['id'], option_rhs['id'])
        if key in self.cache and not self.overwrite:
            return self.cache[key]
        logger.debug(f"Computing uncached feature: {feature_name} for {query} - {option_lhs['id']} vs {option_rhs['id']}")
        feature = feature_fn(query, option_lhs, option_rhs)
        self.cache[key] = feature
        with open(self.path, 'wb') as f:
            pickle.dump(self.cache, f)
        return feature


def parse_args():
    # Get feature names in argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    return args


def get_feature_fn(feature_name):
    both_ways = False
    if feature_name.startswith('both_ways_'):
        feature_name = feature_name.replace('both_ways_', '')
        both_ways = True

    eval_fn = eval_agent.__dict__[feature_name]
    # Bind inference_uri
    if both_ways:
        eval_fn = functools.partial(eval_agent.check_both_ways, eval_fn=eval_fn)
    return eval_fn


def preference_to_label(preference):
    if preference == 'LHS':
        return -1
    elif preference == 'RHS':
        return 1
    else:
        return 0


def row_to_dict(row):
    return {
        'id': row['product_id'],
        'name': row['product_name'],
        'description': row['product_description'],
        'class': row['product_class'],
        'category_hierarchy': row['category hierarchy'],
        'grade': row['grade']
    }


class InferenceModel:

    def __init__(self, model_path):
        self.clf = pickle.load(open(model_path, 'rb'))
        self.feature_names = self.clf.feature_names_in_
        self.cache = FeatureCache()

    def predict(self, query, lhs, rhs):
        feature_vector = [query, lhs, rhs]
        # Build feature vector of the same length as the feature_names
        feature_vector += [0] * (len(self.feature_names) - len(feature_vector))
        for i, feature_name in enumerate(self.feature_names):
            feature_fn = get_feature_fn(feature_name)
            feature_value = self.cache.compute_feature(feature_fn, feature_name, query, lhs, rhs)
            feature_vector[i] = preference_to_label(feature_value)
        as_df = pd.DataFrame([feature_vector], columns=self.feature_names)
        return self.clf.predict(as_df)[0]

    def rank(self, query, docs):
        pairwise_predictions = []
        logger.info(f"Ranking {len(docs)} documents for query: {query}")
        for idx, lhs in docs.iterrows():
            lhs_score = 0
            for idx, rhs in docs.iterrows():
                try:
                    lhs = row_to_dict(lhs) if isinstance(lhs, pd.Series) else lhs
                    rhs = row_to_dict(rhs) if isinstance(rhs, pd.Series) else rhs
                except KeyError as e:
                    logger.error(f"Could not convert row to dict: {lhs}, {rhs} -- {e}")
                    continue
                if lhs['id'] == rhs['id']:
                    continue
                score = self.predict(query, lhs, rhs)
                lhs_score += score
                logger.info(f"Comparing {lhs['id']} vs {rhs['id']} ({lhs['grade']} vs {rhs['grade']}) with score: {lhs_score}")
                pairwise_predictions.append((lhs['id'], rhs['id'], score))
        # Rank docs based on widd
        pairwise_preds = pd.DataFrame(pairwise_predictions, columns=['lhs', 'rhs', 'score'])
        scored = pairwise_preds.groupby('lhs').sum().reset_index()
        return scored.sort_values('score', ascending=False)


def wands_test_data(n=10):
    return queries_sample(num_queries=n, num_docs=10, seed=420)


def main():
    args = parse_args()
    inference_model = InferenceModel(args.model)
    test_data = wands_test_data()
    for query in test_data['query'].unique():
        docs = test_data[test_data['query'] == query]
        ranked_docs = inference_model.rank(query, docs)
        print(ranked_docs)


if __name__ == "__main__":
    main()
