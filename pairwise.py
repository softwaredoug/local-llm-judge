import pandas as pd
import eval_agent


def pairwise_df(n=250):
    products = pd.read_csv('data/WANDS/dataset/product.csv', delimiter='\t')
    queries = pd.read_csv('data/WANDS/dataset/query.csv', delimiter='\t')
    labels = pd.read_csv('data/WANDS/dataset/label.csv', delimiter='\t')
    labels.loc[labels['label'] == 'Exact', 'grade'] = 2
    labels.loc[labels['label'] == 'Partial', 'grade'] = 1
    labels.loc[labels['label'] == 'Irrelevant', 'grade'] = 0
    labels = labels.merge(queries, how='left', on='query_id')
    labels = labels.merge(products, how='left', on='product_id')

    # Sample n rows
    labels = labels.sample(n * 10, random_state=42)

    # Get pairwise
    pairwise = labels.merge(labels, on='query_id')

    # Drop same id
    pairwise = pairwise[pairwise['product_id_x'] != pairwise['product_id_y']]

    # Drop same rating
    pairwise = pairwise[pairwise['label_x'] != pairwise['label_y']]

    return pairwise.sample(n, random_state=42)


def product_row_to_dict(row):
    if 'product_name_x' in row:
        return {
            'id': row['product_id_x'],
            'name': row['product_name_x'],
            'description': row['product_description_x'],
            'class': row['product_class_x'],
            'category_hierarchy': row['category hierarchy_x'],
            'grade': row['grade_x']
        }
    elif 'product_name_y' in row:
        return {
            'id': row['product_id_y'],
            'name': row['product_name_y'],
            'description': row['product_description_y'],
            'class': row['product_class_y'],
            'category_hierarchy': row['category hierarchy_y'],
            'grade': row['grade_y']
        }


def output_row(query, product_lhs, product_rhs, human_preference, agent_preference):
    return {
        'query': query,
        'product_name_lhs': product_lhs['name'],
        'product_description_lhs': product_lhs['description'],
        'product_id_lhs': product_lhs['id'],
        'product_class_lhs': product_lhs['class'],
        'category_hierarchy_lhs': product_lhs['category_hierarchy'],
        'grade_lhs': product_lhs['grade'],
        'product_name_rhs': product_rhs['name'],
        'product_description_rhs': product_rhs['description'],
        'product_id_rhs': product_rhs['id'],
        'product_class_rhs': product_rhs['class'],
        'category_hierarchy_rhs': product_rhs['category_hierarchy'],
        'grade_rhs': product_rhs['grade'],
        'human_preference': human_preference,
        'agent_preference': agent_preference
    }


def human_pref(query, product_lhs, product_rhs):
    human_preference = product_lhs['grade'] - product_rhs['grade']
    print(f"Grade LHS: {product_lhs['grade']}, Grade RHS: {product_rhs['grade']}")
    if human_preference > 0:
        return 'LHS'
    elif human_preference < 0:
        return 'RHS'
    else:
        return 'Neither'


def results_df_stats(results_df):
    same_preference = len(results_df[results_df['human_preference'] == results_df['agent_preference']]) if (
            len(results_df) > 0) else 0
    no_preference = len(results_df[results_df['agent_preference'] == 'Neither']) if (len(results_df) > 0) else 0
    different_preference = len(results_df[(results_df['human_preference'] != results_df['agent_preference']) & (
            results_df['human_preference'] != 'Neither') & (results_df['agent_preference'] != 'Neither')]) if (
            len(results_df) > 0) else 0
    print(f"Same Preference: {same_preference}, Different Preference: {different_preference}, No Preference: {no_preference}")
    if (same_preference + different_preference) > 0:
        print(f"Percentage same preference: {same_preference / (same_preference + different_preference) * 100}%")


def has_been_labeled(results_df, query, product_lhs, product_rhs):
    result_exists = len(results_df) > 0 and (results_df[(results_df['query'] == query) &
                                                        (results_df['product_id_lhs'] == product_lhs['id']) &
                                                        (results_df['product_id_rhs'] == product_rhs['id'])].shape[0] > 0)
    return result_exists


def eval(eval_fn=eval_agent.decide_ensemble, assert_on_diff=True):
    df = pairwise_df()
    func_name = eval_fn.__name__
    results_df = pd.DataFrame()
    try:
        results_df = pd.read_pickle(f'data/{func_name}.pkl')
    except FileNotFoundError:
        pass

    for idx, row in df.iterrows():
        query = row['query_x']
        product_lhs = product_row_to_dict(row[['product_name_x', 'product_description_x',
                                               'product_class_x',
                                               'product_id_x', 'category hierarchy_x', 'grade_x']])
        product_rhs = product_row_to_dict(row[['product_name_y', 'product_description_y',
                                               'product_class_y',
                                               'product_id_y', 'category hierarchy_y', 'grade_y']])
        if has_been_labeled(results_df, query, product_lhs, product_rhs):
            print(f"Already rated query: {query}, product_lhs: {product_lhs['name']}, product_rhs: {product_rhs['name']}")
            print("Skipping")
            continue
        human_preference = human_pref(query, product_lhs, product_rhs)
        agent_preference = eval_fn(query, product_lhs, product_rhs)
        print(f"Human Preference: {human_preference}, Agent Preference: {agent_preference}")

        results_df = pd.concat([results_df, pd.DataFrame([output_row(query, product_lhs, product_rhs, human_preference,
                                                                     agent_preference)])])
        results_df_stats(results_df)

        results_df.to_pickle(f'data/{func_name}.pkl')
    results_df_stats(results_df)



if __name__ == '__main__':
    eval()
