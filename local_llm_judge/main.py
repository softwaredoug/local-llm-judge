import argparse
import logging
import os
import functools

import pandas as pd
from prompt_toolkit.history import FileHistory

import local_llm_judge.eval_agent as eval_agent
from local_llm_judge.log_stdout import enable
from local_llm_judge.wands_data import pairwise_df

logger = logging.getLogger(__name__)


DEFAULT_SEED = 42


def parse_args():
    # List all functions in eval_agent
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-fn', type=str, default='unanimous_ensemble_name_desc')
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--destroy-cache', action='store_true', default=False)
    parser.add_argument('--check-both-ways', action='store_true', default=False)
    parser.add_argument('--llm-histroy', type=str, default=".llm_shell_history")
    args = parser.parse_args()
    all_fns = eval_agent.all_fns()
    # Funcs to string
    all_fns = [fn.__name__ for fn in all_fns]
    if args.eval_fn not in all_fns:
        logger.info(f"Invalid function name. Available functions: {all_fns}")
        exit(1)
    args.eval_fn = eval_agent.__dict__[args.eval_fn]
    args.cache_key = args.eval_fn.__name__
    if args.check_both_ways:
        cache_key = "both_ways_" + args.eval_fn.__name__
        args.eval_fn = functools.partial(eval_agent.check_both_ways, eval_fn=args.eval_fn)
        args.cache_key = cache_key

    if args.seed != 42:
        args.cache_key += f"_seed_{args.seed}"

    if args.verbose:
        enable("local_llm_judge")
        enable(__name__)
    return args


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
    logger.debug(f"Grade LHS: {product_lhs['grade']}, Grade RHS: {product_rhs['grade']}")
    if human_preference > 0:
        return 'LHS'
    elif human_preference < 0:
        return 'RHS'
    else:
        return 'Neither'


def results_df_stats(results_df):
    agent_has_preference = len(results_df[results_df['agent_preference'] != 'Neither']) if (len(results_df) > 0) else 0
    same_preference = len(results_df[results_df['human_preference'] == results_df['agent_preference']]) if (
            len(results_df) > 0) else 0
    no_preference = len(results_df[results_df['agent_preference'] == 'Neither']) if (len(results_df) > 0) else 0
    different_preference = len(results_df[(results_df['human_preference'] != results_df['agent_preference']) & (
            results_df['human_preference'] != 'Neither') & (results_df['agent_preference'] != 'Neither')]) if (
            len(results_df) > 0) else 0
    logger.info(f"Same Preference: {same_preference}," +
                f" Different Preference: {different_preference}, No Preference: {no_preference}")
    if (same_preference + different_preference) > 0:
        precision = same_preference / (same_preference + different_preference) * 100
        recall = agent_has_preference / len(results_df) * 100
        logger.info(f"Precision: {precision:.2f}% | Recall: {recall:.2f}% (N={len(results_df)})")


def has_been_labeled(results_df, query, product_lhs, product_rhs):
    result_exists = (len(results_df) > 0
                     and (results_df[(results_df['query'] == query) &
                          (results_df['product_id_lhs'] == product_lhs['id']) &
                          (results_df['product_id_rhs'] == product_rhs['id'])].shape[0] > 0))
    return result_exists


def main(eval_fn=eval_agent.unanimous_ensemble_name_desc,
         N=250,
         destroy_cache=False,
         history_path=".llm_shell_history",
         seed=42):
    if history_path:
        eval_agent.qwen.history = FileHistory(history_path)

    df = pairwise_df(N, seed)
    cache_key = args.cache_key
    results_df = pd.DataFrame()
    if destroy_cache and os.path.exists(f'data/{cache_key}.pkl'):
        os.remove(f'data/{cache_key}.pkl')
    try:
        results_df = pd.read_pickle(f'data/{cache_key}.pkl')
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
            logger.info(f"Already rated query: {query}, " +
                        f"product_lhs: {product_lhs['name']}, product_rhs: {product_rhs['name']}")
            logger.info("Skipping")
            continue
        human_preference = human_pref(query, product_lhs, product_rhs)
        agent_preference = eval_fn(query, product_lhs, product_rhs)
        if agent_preference != 'Neither' and human_preference != agent_preference:
            logger.warning(f"Disagreement - Human Preference: {human_preference}, Agent Preference: {agent_preference}")
        logger.info(f"Human Preference: {human_preference}, Agent Preference: {agent_preference}")

        results_df = pd.concat([results_df, pd.DataFrame([output_row(query, product_lhs, product_rhs, human_preference,
                                                                     agent_preference)])])
        results_df_stats(results_df)

        results_df.to_pickle(f'data/{cache_key}.pkl')
    results_df_stats(results_df)


if __name__ == '__main__':
    args = parse_args()
    main(args.eval_fn, args.N, args.destroy_cache, seed=args.seed)
