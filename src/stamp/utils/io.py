'''
STAMP: input/output utilities
'''

# toml_none_to_empty: map any None instances to an empty string for TOML serialisation
def toml_none_to_empty(obj):
    if isinstance(obj, dict):
        return {k: toml_none_to_empty(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [toml_none_to_empty(v) for v in obj]
    return '' if obj is None else obj