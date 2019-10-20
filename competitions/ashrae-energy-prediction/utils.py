import json
from datetime import datetime
from collections import MutableMapping


def make_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def read_json(fp):
    with open(fp, 'r') as f:
        return json.load(f)


def save_dict(fp, d):
    with open(fp, 'w') as f:
        f.write(json.dumps(d))


def print_divider(title):
    print('\n' + '-' * 25, title, '-' * 25 + '\n')


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
