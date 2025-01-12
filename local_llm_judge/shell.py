from local_llm_judge.qwen import Qwen
from local_llm_judge.qwen import Agent
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
import argparse
import json
import enum


# Use a history file for persistence


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen shell")
    parser.add_argument("--agent", action="store_true", help="Agent interaction mode")
    parser.add_argument("--history", type=str, help="History file path",
                        default=".llm_shell_history")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--system", type=str, help="System prompt",
                        default="You are Qwen, a helpful conversational assistant.")
    return parser.parse_args()


class CmdStyle(enum.Enum):
    SINGLE_PROMPT = 1
    JSON = 2
    JSON_WITH_SYSTEM = 3


def _cmd_style(shell_line) -> CmdStyle:
    try:
        json_lines = json.loads(shell_line)
        for line in json_lines:
            if "role" not in line or "content" not in line:
                return CmdStyle.SINGLE_PROMPT

        if json_lines[0]['role'] == 'system':
            return CmdStyle.JSON_WITH_SYSTEM
        return CmdStyle.JSON
    except json.JSONDecodeError:
        return CmdStyle.SINGLE_PROMPT


def _to_stdout(text):
    print(text)  # noqa


def _agent_for_shell(system=None, use_agent=False):
    qwen = Qwen(system=system)
    agent = qwen
    if use_agent:
        _to_stdout("Qwen starting -- agent interaction mode")
        agent = Agent(qwen)
    else:
        _to_stdout("Qwen starting -- single-turn mode")
    return agent


def _echo_user_prompt(system, messages):
    _to_stdout("\n")
    if system:
        _to_stdout(f"System: {system}\n")
    for message in messages:
        _to_stdout(f"User: {message}\n")


def agent_shell(system=None, use_agent=False, echo=False):
    agent = _agent_for_shell(system, use_agent)
    history = FileHistory(".llm_shell_history")
    session = PromptSession(history=history)
    while True:
        try:
            messages = session.prompt(">>> ")
            style = _cmd_style(messages)
            if style == CmdStyle.JSON:
                messages = json.loads(messages)
                messages = [m["content"] for m in messages]
            elif style == CmdStyle.JSON_WITH_SYSTEM:
                messages = json.loads(messages)
                _to_stdout(f"New Agent with System: {messages[0]['content']}") # noqa
                system = messages[0]["content"]
                messages = [m["content"] for m in messages if m["role"] == "user"]
                agent = _agent_for_shell(system, use_agent)
            if isinstance(messages, str) and messages.lower() in ["exit", "quit"]:
                _to_stdout("Adios muchachos!")
                break
            if isinstance(messages, str):
                messages = [messages]
            if echo:
                _echo_user_prompt(system, messages)
            response = agent(messages)
            _to_stdout(f"Qwen: {response}")
        except KeyboardInterrupt:
            _to_stdout("Adios muchachos!")
            break
        except EOFError:
            _to_stdout("Adios muchachos!")
            break


def main():
    args = parse_args()
    agent_shell(args.system, args.agent, echo=args.verbose)


if __name__ == "__main__":
    main()
