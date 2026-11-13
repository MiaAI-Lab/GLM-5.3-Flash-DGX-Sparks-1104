"""Emit the synthetic 32K release restore request; no model, network or dependencies."""
import json


def request():
    instruction = (
        'Write a numbered list of fifty distinct practical observations about this passage. '
        'For each observation write one complete sentence with a different opening phrase. '
        'Begin immediately with item 1 and continue until item 50; do not summarize or ask questions.'
    )
    return {
        'model': 'GLM-5.3-Flash', 'stream': False, 'temperature': 0, 'seed': 20261112,
        'max_tokens': 128, 'chat_template_kwargs': {'enable_thinking': False},
        'messages': [{'role': 'user', 'content': (
            'Compact long context 32K independent session 0.\n'
            + 'The quick brown fox jumps over the lazy dog.\n' * 3200 + '\n' + instruction
        )}],
    }


if __name__ == '__main__':
    print(json.dumps(request(), indent=2))
