import os
import json

from mlx_lm import load, generate
from mlx_lm.utils import ModelNotFoundError


class NoOpHistory:
    def store_string(self, message):
        pass


_no_op_history = NoOpHistory()


class Qwen:

    def __init__(self,
                 path="~/.mlx/Qwen2.5-7B-Instruct",
                 history=_no_op_history,
                 system=None):
        try:
            path = os.path.expanduser(path)
            self.model, self.tokenizer = load(path, tokenizer_config={"eos_token": "<|im_end|>"})
            self.system = system
            self.history = history

        except ModelNotFoundError as e:
            cmd = f"mlx_lm.convert --hf-path [HUGGING FACE MODEL] --mlx-path [YOUR PATH CMD (ie {path})]"
            qwen_cmd = "mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path mlx/Qwen2.5-7B-Instruct/ -q"
            error = f"""
            MLX Model not found. You need to convert from Hugging Face model to MLX model.

            Please run the following command:

            {cmd}

            Such as this for Qwen:

            {qwen_cmd}

            Original Error:

            {e}
            """
            raise ModelNotFoundError(error)

    def __call__(self, messages, max_tokens=512, verbose=False):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list) and isinstance(messages[0], str):
            messages = [{"role": "user", "content": message} for message in messages
                        if isinstance(message, str)]
        if isinstance(messages, list):
            if len(messages) == 0:
                raise ValueError("Messages must not be empty.")
            for msg in messages:
                if not isinstance(msg, dict):
                    raise ValueError("Messages must be a list of strings or a list of dictionaries.")
                if "content" not in msg:
                    raise ValueError("Messages must have a 'content' key.")
                if "role" not in msg:
                    raise ValueError("Messages must have a 'role' key.")

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        self.history.store_string(json.dumps(messages))

        response = generate(self.model,
                            self.tokenizer,
                            prompt=text,
                            verbose=verbose,
                            max_tokens=max_tokens)

        return response


class Agent:

    def __init__(self, qwen, system=None):
        self.qwen = qwen
        if not system:
            system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        self.messages = [{"role": "system", "content": system}]

    def __call__(self, messages, max_tokens=512, verbose=False):
        self.messages.extend([{"role": "user", "content": message} for message in messages])
        response = self.qwen(messages, max_tokens=max_tokens, verbose=verbose)
        self.messages.append({"role": "assistant", "content": response})
        return response
