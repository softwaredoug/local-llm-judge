
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

## List of variants

There are a lot of prompts / variants listed [in this file](https://github.com/softwaredoug/local-llm-judge/blob/main/local_llm_judge/eval_agent.py#L197). The function names are the arguments to the `eval-fn` argument at command line.
