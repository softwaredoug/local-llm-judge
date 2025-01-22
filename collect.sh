#!/bin/bash

N="${1:-3000}"

poetry run python -m local_llm_judge.main --verbose --eval-fn category_allow_neither --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn category --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn category_allow_neither --check-both-ways --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn category --check-both-ways --N $N

poetry run python -m local_llm_judge.main --verbose --eval-fn class_allow_neither --check-both-ways --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn classs --check-both-ways --N $N

poetry run python -m local_llm_judge.main --verbose --eval-fn name_allow_neither --check-both-ways --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn name --check-both-ways --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn name_allow_neither --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn name --N $N

poetry run python -m local_llm_judge.main --verbose --eval-fn class_allow_neither --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn classs --N $N

poetry run python -m local_llm_judge.main --verbose --eval-fn desc_allow_neither --check-both-ways --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn desc --check-both-ways --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn desc_allow_neither --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn desc --N $N

