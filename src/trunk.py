import json

def update_namespace_with_json(args):
    with open(args.json_file_path, 'r') as f:
        json_data = json.load(f)
    args_dict = vars(args)
    for key, value in json_data.items():
        if key in args_dict:
            args_dict[key] = value
        else:
            setattr(args, key, value)
    return args