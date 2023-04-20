with open('subset_data/captions.txt', 'r') as f_in, open('subset_data/captions_with_id.txt', 'w') as f_out:
    counter = 0
    for line in f_in:
        filename, caption = line.strip().split(',', 1)
        caption_id = counter % 5
        f_out.write(f"{filename},{caption_id},{caption}\n")
        counter += 1
