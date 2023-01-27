import string


def load_doc(filename):
    """
    Loads a text file and returns its content as a string.
    """
    try:
        with open(filename, 'r', encoding='utf-8-sig') as file:
            text = file.read()
    except:
        print("Error occured while reading the file")
        return None
    return text


def load_descriptions(doc):
    """
    Loads a document and returns a mapping of image IDs to their corresponding
    descriptions.
    """
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)
    return mapping


def clean_descriptions(descriptions):
    """
    Cleans the descriptions by removing punctuation, lowercasing all words, and
    removing words that are not alphabetic or have a length less than 2.
    """
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)


def to_vocabulary(descriptions):
    """
    Extracts a set of all unique words from the descriptions.
    """
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


def save_descriptions(descriptions, filename):
    """
    Saves the descriptions to a text file.
    """
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    with open(filename, 'w') as file:
        file.write(data)


if __name__ == '__main__':
    filename = 'dataset/captions.txt'
    doc = load_doc(filename)
    descriptions = load_descriptions(doc)
    # print(descriptions)
    print(len(descriptions))
    clean_descriptions(descriptions)
    vocabulary = to_vocabulary(descriptions)
    # print(vocabulary)
    print(len(vocabulary))
    save_descriptions(descriptions, 'descriptions.txt')


save_descriptions(descriptions, 'descriptions.txt')
