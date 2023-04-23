with open('dataset_flickr8k/captions.txt', 'r') as f_in, open('dataset_flickr8k/captions_with_id.txt', 'w') as f_out:
    counter = 0
    for line in f_in:
        filename, caption_with_tab = line.strip().split('#', 1)
        caption = caption_with_tab.split('\t')[1]
        caption_id = counter % 5
        f_out.write(f"{filename},{caption_id},{caption}\n")
        counter += 1
