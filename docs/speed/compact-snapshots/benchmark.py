"""One fresh-process arm of the synthetic eight-session/four-turn cache experiment.

Python standard library only. Run legacy/committed/committed/legacy at each cap.
The output directory must be new. Compare subsequent arms with the first arm.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

SENTENCE = 'The quick brown fox jumps over the lazy dog.\n'
FOLLOWUP = SENTENCE * 200 + '\nReply with exactly KDA-CAPTURE-OK and nothing else.'


def initial_body(session):
    return {'model': 'GLM-5.3-Flash', 'stream': False, 'temperature': 0, 'seed': 20261112,
            'max_tokens': 32, 'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'user', 'content': (
                f'Independent session {session}.\n'
                'KSEO KDA real-state capture 20261112. Read the following text carefully.\n'
                + SENTENCE * 960 + '\nReply with exactly KDA-CAPTURE-OK and nothing else.')}]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8888')
    p.add_argument('--cap-gib', type=float, choices=(2.5, 5), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.out / (name + '.json')).write_text(json.dumps(value, indent=2), encoding='utf-8')

    def idle():
        started, stable = time.monotonic(), None
        while time.monotonic() - started < 120:
            with urllib.request.urlopen(args.url + '/metrics', timeout=15) as r:
                text = r.read().decode()
            m = {k: float(v) for k, v in re.findall(r'^tensorfold:([a-z_]+) ([0-9.eE+\-]+)$', text, re.M)}
            assert m['prompt_cache_budget_bytes'] == args.cap_gib * (1 << 30)
            if m['requests_running'] == m['requests_waiting'] == m['generation_tokens_running'] == 0:
                if stable is None:
                    stable = time.monotonic()
                if time.monotonic() - stable >= 2:
                    return m
            else:
                stable = None
            time.sleep(.25)
        raise TimeoutError('engine did not settle')

    before = idle()
    assert before['prompt_tokens_total'] == 0, 'Restart both ranks before each arm'
    save('before', before)
    bodies = [initial_body(i) for i in range(8)]
    rows = []
    started = time.monotonic()
    for turn in range(4):
        for session, body in enumerate(bodies):
            label = f'turn{turn}-session{session}'
            if turn:
                body['messages'].append({'role': 'user', 'content': FOLLOWUP})
            prior = None
            if args.reference:
                prior = json.loads((args.reference / (label + '.json')).read_text(encoding='utf-8'))
                assert prior['request'] == body, ('input mismatch', label)
            t0 = time.monotonic()
            request = urllib.request.Request(args.url + '/v1/chat/completions',
                json.dumps(body).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=600) as r:
                response = json.load(r)
            wall = time.monotonic() - t0
            save(label, {'request': body, 'response': response, 'wall_seconds': wall})
            u, tf = response['usage'], response['tensorfold']
            cached = u['prompt_tokens_details']['cached_tokens']
            if turn == 0:
                assert cached == 0, ('cold request hit', label)
            if prior:
                other = prior['response']
                assert other['choices'] == response['choices'], ('choices mismatch', label)
                assert other['tensorfold']['token_sha256'] == tf['token_sha256']
                assert all(other['usage'][k] == u[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
            save(label + '-idle', idle())
            rows.append({'turn': turn, 'session': session, 'wall_seconds': wall,
                'prompt_tokens': u['prompt_tokens'], 'completion_tokens': u['completion_tokens'],
                'cached_tokens': cached, 'ttft_seconds': tf['time_to_first_token'],
                'prefill_seconds': tf['prefill_seconds'], 'token_sha256': tf['token_sha256'],
                'request_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()})
            print(json.dumps(rows[-1]), flush=True)
            body['messages'].append({'role': 'assistant', 'content': response['choices'][0]['message']['content']})
    save('summary', {'rows': rows, 'foreground_wall_seconds': sum(x['wall_seconds'] for x in rows),
                    'elapsed_including_idle_seconds': time.monotonic() - started, 'after': idle()})


if __name__ == '__main__':
    main()
