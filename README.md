
## Local LLM Search Relevance Judge

(Runs on Apple Silicon only with MLX)

Using the [WANDS dataset](https://github.com/wayfair/WANDS/tree/main), use a local LLM (Qwen 2.5) to try to evaluate pairwise search relevance preferenec.

The LLM strategies here attempt to recover the pairwise relevance preference of the WANDS human labelers. See [this blog post]()

## To run:
```
$ poetry install
```

Get Qwen from Hugging face, convert to MLX format
```
$ mkdir -p ~/.mlx
$ poetry run mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path ~/.mlx/Qwen2.5-7B-Instruct/ -q\n
```

Run:

```
$ poetry run python -m local_llm_judge.main --verbose --eval-fn name
```
