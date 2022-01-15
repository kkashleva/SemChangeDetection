import argparse
import os
import random


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--targets_path', type=str, help="Path to a .txt file with target words")
    arg_parser.add_argument('--corpora_paths', type=str, help="Paths to corpora separated with ;")
    arg_parser.add_argument('--output_folder', type=str, help="Path to a folder for processed files")
    args = arg_parser.parse_args()
    if not args.targets_path:
        print('Please specify path to target words (--targets_path)')
    elif not args.corpora_paths:
        print('Please specify path to corpora separated with ; (--corpora_paths)')
    elif not args.output_folder:
        print('Please specify output folder (--output_folder)')
    else:
        return args


arguments = parse_args()
if not arguments:
    exit(1)
with open(arguments.targets_path) as f:
    targets = [i.strip() for i in f.readlines()]
corpora = arguments.corpora_paths.split(';')
all_lines = []
os.makedirs(arguments.output_folder, exist_ok=True)
for file in corpora:
    print(f'Processing file {file}')
    lines = []
    for line in open(file):
        clean_line = line
        words = line.split()
        for target in targets:
            bare_word = target[:-3]  # stripping part of speech tag
            if bare_word in words:
                break  # it means that the line contains a word that is identical to a target word, but has a
                # different part of speech. We drop such lines for simplicity.
            if target in words:
                clean_line = clean_line.replace(target, bare_word)  # we substitute target words with their versions
                # without a POS tag
        else:  # if break didn't trigger, we add such a line to the result
            lines.append(clean_line)
    with open(os.path.join(arguments.output_folder, f'processed_{os.path.basename(file)}'), 'w') as f:
        f.writelines(lines)
    all_lines.extend(lines)
random.shuffle(all_lines)
train_part = int(len(all_lines) * 0.9)
with open(os.path.join(arguments.output_folder, 'train.txt'), 'w') as f:
    f.writelines(all_lines[:train_part])
with open(os.path.join(arguments.output_folder, 'test.txt'), 'w') as f:
    f.writelines(all_lines[train_part:])
