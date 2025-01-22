
## Local LLM Search Relevance Judge

(Runs on Apple Silicon only with MLX)

Using the [WANDS dataset](https://github.com/wayfair/WANDS/tree/main), use a local LLM (Qwen 2.5) to try to evaluate pairwise search relevance relevance.

The LLM strategies here attempt to recover the pairwise relevance preference of the WANDS human labelers. Blog post series:

* [Turning my laptop into a Search Relevance Judge with local LLMs](https://softwaredoug.com/blog/2025/01/13/llm-for-judgment-lists)
* [Check twice, cut once with LLM search relevance eval](https://softwaredoug.com/blog/2025/01/19/llm-as-judge-both-ways)
* [Classic ML to cope with Dumb LLM Judges](https://softwaredoug.com/blog/2025/01/21/llm-judge-decision-tree)

## To run:
```
$ poetry install
```

Download [WANDS into data folder](https://github.com/wayfair/WANDS/tree/main) 

Get Qwen from Hugging face, convert to MLX format
```
$ mkdir -p ~/.mlx
$ poetry run mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path ~/.mlx/Qwen2.5-7B-Instruct/ -q\n
```

Run local judge

```
$ poetry run python -m local_llm_judge.main --verbose --eval-fn name
```

Optionally - Talk to Qwen

```
poetry run python -m local_llm_judge.shell
```

## Double check or not

You can double check the variants, by asking `--check-both-ways`

$ poetry run python -m local_llm_judge.main --verbose --eval-fn name --check-both-ways

## Letting agent choose neither / say it doesn't know

The variants look at different fields, with a version of prompts that allow the agent to chicken-out and say Neither if it doesn't know. This improves precision, sacrificing coverage/recall.

```
$ poetry run python -m local_llm_judge.main --verbose --eval-fn name_allow_neither --check-both-ways
```

## Results output

Result dataframes put into the data/ directory

`data/both_ways_[eval_fn]` or just `data/[eval_fn].pkl`


## Training an ensemble

Run `./collect.sh N` (ie `/.collect.sh 7000`) to run a large set of variants with different setting permutations (allowing neither, double checking or not).

Then the train script will try to train a prediction using all the different agent permutations:

```
$ poetry run python -m  local_llm_judge.train --feature_names data/both_ways_category.pkl data/both_ways_name.pkl  data/both_ways_desc.pkl data/both_ways_classs.pkl data/both_ways_category_allow_neither.pkl data/both_ways_name_allow_neither.pkl data/both_ways_desc_allow_neither.pkl data/both_ways_class_allow_neither.pkl
```

Then you can see a precision / recall tradeoffs of a decision tree trained to predict the first 1000 labels:

```
['both_ways_desc_allow_neither', 'both_ways_class_allow_neither'] 1.0 0.013
['both_ways_name', 'both_ways_class_allow_neither'] 0.9861111111111112 0.072
['both_ways_category', 'both_ways_name', 'both_ways_classs', 'both_ways_name_allow_neither', 'both_ways_class_allow_neither'] 0.9673366834170855 0.398
['both_ways_category', 'both_ways_name', 'both_ways_classs', 'both_ways_class_allow_neither'] 0.9668508287292817 0.362
['both_ways_name', 'both_ways_desc_allow_neither', 'both_ways_class_allow_neither'] 0.9666666666666667 0.09
['both_ways_desc', 'both_ways_desc_allow_neither', 'both_ways_class_allow_neither'] 0.9666666666666667 0.06
['both_ways_desc', 'both_ways_class_allow_neither'] 0.9666666666666667 0.06
['both_ways_category', 'both_ways_name', 'both_ways_classs'] 0.9665738161559888 0.359
['both_ways_category', 'both_ways_name', 'both_ways_desc', 'both_ways_classs', 'both_ways_category_allow_neither'] 0.9659367396593674 0.411
```

## List of variants

There are a lot of prompts / variants listed [in this file](https://github.com/softwaredoug/local-llm-judge/blob/main/local_llm_judge/eval_agent.py#L197). The function names are the arguments to the `eval-fn` argument at command line.
