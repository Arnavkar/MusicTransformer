import json

def read_maestro_json(path='./data/raw/maestro-v3.0.0/maestro-v3.0.0.json'):
    with open(path) as f:
        data = json.load(f)
    return data

def get_dup_list(data):
    keys_to_delete_list = []
    dup_set = set()

    assert len(data["canonical_composer"]) == len(data["canonical_title"])
    for i in range(len(data["canonical_title"])):
        title = data["canonical_title"][str(i)]
        title_stripped = ''.join(e for e in title if e.isalnum()).lower()

        composer = data["canonical_composer"][str(i)]
        composer_stripped = ''.join(e for e in composer if e.isalnum()).lower()

        title_composer = title_stripped + '-' + composer_stripped

        if title_composer not in dup_set:
            dup_set.add(title_composer)
        else:
            if data["split"][str(i)] == "train":
                keys_to_delete_list.append(i)

    return keys_to_delete_list

def find_outliers(data):
    for i in range(len(data["canonical_title"])):
        title = data["canonical_title"][str(i)]
        title_stripped = ''.join(e for e in title if e.isalnum()).lower()

def write_maestro_no_dups(data,del_list):
    maestro_no_dups = {}
    for key ,sub_dict in data.items():
        maestro_no_dups[key] = {int(k):v for k,v in sub_dict.items() if int(k) not in del_list}

    print("Num of duplicates found: ", len(del_list))
    print("Num of files remaining: ", len(maestro_no_dups["canonical_title"]))
    #print by split
    print("Num of files remaining in train: ", len([k for k,v in maestro_no_dups["split"].items() if v == "train"]))
    print("Num of files remaining in validation: ", len([k for k,v in maestro_no_dups["split"].items() if v == "validation"]))
    print("Num of files remaining in test: ", len([k for k,v in maestro_no_dups["split"].items() if v == "test"]))
    
    
    with open('./data/raw/maestro-v3.0.0/maestro-v3.0.0-no-dups.json', 'w') as f:
        json.dump(maestro_no_dups, f)

if __name__ == "__main__":
    json_data = read_maestro_json()
    keys_to_delete = get_dup_list(json_data)
    write_maestro_no_dups(json_data,keys_to_delete)