import string


def load_doc(filename):
    """
    Loads a text file and returns its content as a string.
    """
    try:
        with open(filename, 'r', encoding='utf-8-sig') as file:
            text = file.read()
    except IOError as e:
        print(e)
        print("Error occured while reading the file")
        return None
    return text


def load_descriptions(doc):
    """
    Loads a document and returns a mapping of image IDs to their corresponding
    descriptions.
    """
    mapping = dict()
    for i, line in enumerate(doc.split('\n')):
        # skip the first iteration (removing header)
        if i == 0:
            continue
        tokens = line.split(',')
        if len(tokens) < 3:
            continue
        image_id, image_desc_id, *image_desc = tokens
        # image_id = image_id.split('.')[0]
        image_id = image_id
        image_desc_id = image_desc_id.strip()
        image_desc = ', '.join(image_desc).strip()
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append((image_desc_id, image_desc))
    return mapping


def clean_descriptions(descriptions):
    """
    Cleans the descriptions by removing punctuation, lowercasing all words, and
    removing words that are not alphabetic or have a length less than 2.
    """
    table = str.maketrans('', '', string.punctuation)
    for image_name, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc_id, desc = desc_list[i][0], desc_list[i][1]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = (desc_id, ' '.join(desc))


def to_vocabulary(descriptions):
    """
    Extracts a set of all unique words from the descriptions.
    """
    all_desc = set()
    for img_id in descriptions:
        for desc in descriptions[img_id]:
            all_desc.update(desc[1].split())
    return all_desc


def save_descriptions(descriptions, filename):
    """
    Saves the descriptions to a text file in the format "image_name, comment_number, comment".
    """
    lines = list()
    for img_id in descriptions.keys():
        img_descs = descriptions[img_id]
        for desc in img_descs:
            lines.append(img_id + ', ' + desc[0] + ', ' + desc[1])
    data = '\n'.join(lines)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(data)


filename = 'subset_dataset/subset_captions.txt'
doc = load_doc(filename)
descriptions = load_descriptions(doc)
# print(descriptions)
print(len(descriptions))
clean_descriptions(descriptions)
vocabulary = to_vocabulary(descriptions)
# print(vocabulary)
print(len(vocabulary))
save_descriptions(descriptions, 'descriptions.txt')
