def flat_to_bio(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    bio_annotations = []
    sent=[]
    for i, line in enumerate(lines):
        line = line.strip().split()
        if not line:
            sent.append((" ", " ", " "))
            bio_annotations.append(sent)
            sent=[]
        else:
            filename, token, tag = line
            if tag != 'O':
                if not sent or sent[-1][2].startswith('O'):
                    sent.append((filename, token, 'B-' + tag))
                else:
                    sent.append((filename, token, 'I-' + tag))
            else:
                sent.append((filename, token, 'O'))

    space=" " 
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent in bio_annotations:
            for filename, token, tag in sent:
                f.write(f"{filename} {token} {tag}")
                f.write("\n")

# Example usage:
#input_file = 'flat_annotations.txt'
#output_file = 'bio_annotations.txt'
#flat_to_bio(input_file, output_file)

# Example usage:
#input_file = 'flat_annotations_with_filenames.txt'
#output_file = 'bio_annotations_with_filenames.txt'
#flat_to_bio(input_file, output_file)


# Example usage:
input_file = '/working/abul/RadGraph/data/train.txt'
output_file = '/working/abul/RadGraph/data/bio_train.txt'
flat_to_bio(input_file, output_file)
input_file="/working/abul/RadGraph/data/test_MIMIC_CXR_labeller1.txt"
output_file="/working/abul/RadGraph/data/bio_test_MIMIC_CXR_labeller1.txt"
flat_to_bio(input_file, output_file)
input_file="/working/abul/RadGraph/data/test_MIMIC_CXR_labeller2.txt"
output_file="/working/abul/RadGraph/data/bio_test_MIMIC_CXR_labeller2.txt"
flat_to_bio(input_file, output_file)
input_file="/working/abul/RadGraph/data/test_CheXpert_labeller1.txt"
output_file="/working/abul/RadGraph/data/bio_test_CheXpert_labeller1.txt"
flat_to_bio(input_file, output_file)
input_file="/working/abul/RadGraph/data/test_CheXpert_labeller2.txt"
output_file="/working/abul/RadGraph/data/bio_test_CheXpert_labeller2.txt"
flat_to_bio(input_file, output_file)