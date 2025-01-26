import argparse
import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
import itertools


def preference_to_label(preference):
    if preference == 'LHS':
        return -1
    elif preference == 'RHS':
        return 1
    else:
        return 0


def build_feature_df(feature_names):
    feature_df = None
    feature_columns = []
    for feature in feature_names:
        df = pd.read_pickle(feature)[['query', 'product_id_lhs', 'product_id_rhs',
                                      'agent_preference', 'human_preference']]
        feature_name = os.path.basename(feature).split(".")[0]
        feature_columns.append(feature_name)
        df.rename(columns={"agent_preference": feature_name}, inplace=True)
        if feature_df is None:
            feature_df = df
        else:
            feature_df = feature_df.merge(df, on=['query', 'product_id_lhs', 'product_id_rhs', 'human_preference'],
                                          how='inner')
        feature_df[feature_name] = feature_df[feature_name].apply(preference_to_label)
    feature_df['human_preference'] = feature_df['human_preference'].apply(preference_to_label)
    feature_df = feature_df[~feature_df.index.duplicated(keep='first')]
    return feature_df, feature_columns


def parse_args():
    # Get feature names in argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_names", type=str, nargs="+")
    parser.add_argument("--num_test", type=int, default=1000)
    args = parser.parse_args()
    return args


def train_tree(train, test, feature_columns):
    clf = DecisionTreeClassifier()
    clf.fit(train[feature_columns],
            train['human_preference'])
    score = clf.score(test[feature_columns],
                      test['human_preference'])
    return clf, score


def predict_tree(clf, test, feature_columns, threshold=0.9):
    """Only assign LHS or RHS if the probability is above the threshold"""
    probas = clf.predict_proba(test[feature_columns])
    definitely_lhs = probas[:, 0] > threshold
    definitely_rhs = probas[:, 1] > threshold
    predictions = np.array([0] * len(test))
    predictions[definitely_lhs] = -1
    predictions[definitely_rhs] = 1

    test.loc[:, 'prediction'] = predictions
    same_label_when_pred = (
        test[test['prediction'] != 0]['human_preference'] == test[test['prediction'] != 0]['prediction']
    )
    print(feature_columns)
    precision = same_label_when_pred.sum() / len(same_label_when_pred)
    recall  = len(same_label_when_pred) / len(test)
    print(f"Precision: {precision} - Recall: {recall}")

    return predictions, feature_columns, precision, recall


def permute_features(feature_columns):
    """Return a list of all possible permutations of the feature columns"""
    permutations = []
    for i in range(1, len(feature_columns) + 1):
        permutations.extend(itertools.combinations(feature_columns, i))
    return permutations


def main():
    args = parse_args()
    feature_df, feature_names = build_feature_df(args.feature_names)
    results = []
    for permutation in permute_features(feature_names):
        permutation = list(permutation)
        train = feature_df.tail(len(feature_df) - args.num_test)
        test = feature_df.head(args.num_test)
        clf, score = train_tree(train, test, permutation)
        _, _, precision, recall = predict_tree(clf, test, feature_columns=permutation)
        results.append({'permutation': permutation, 'precision': precision, 'recall': recall})
        print(f"Permutation: {permutation} - Score: {score}")
    results_df = pd.DataFrame(results).sort_values('precision', ascending=False)
    for _, row in results_df.head(10).iterrows():
        print(row['permutation'], row['precision'], row['recall'])
    # print(export_text(clf, feature_names=feature_names))


if __name__ == "__main__":
    main()
