import argparse
import os
import random
import re


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--targets_path', type=str, help="Path to a .txt file with target words")
    arg_parser.add_argument('--corpora_paths', type=str, help="Paths to corpora separated with ;")
    arg_parser.add_argument('--corpora_language', type=str, help="english, german, swedish, latin or spanish")
    arg_parser.add_argument('--output_folder', type=str, help="Path to a folder for processed files")
    args = arg_parser.parse_args()
    if not args.targets_path:
        print('Please specify path to target words (--targets_path)')
    elif not args.corpora_paths:
        print('Please specify path to corpora separated with ; (--corpora_paths)')
    elif not args.output_folder:
        print('Please specify output folder (--output_folder)')
    elif not args.corpora_language or args.corpora_language not in ['english', 'german', 'swedish', 'latin', 'spanish']:
        print('Please specify a valid corpora language (--corpora_language)')
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
    with open(file, encoding='utf-8-sig') as f:
        print(f'Processing file {file}')
        if arguments.corpora_language == 'english':
            lines = []
            for line in f:
                words = line.split()
                for target in targets:
                    bare_target = target[:-3]  # stripping part of speech tag
                    if bare_target in words:
                        break  # it means that the line contains a word that is identical to a target word,
                        # but has a different part of speech. We drop such lines for simplicity.
                    if target in words:
                        words = [i if i != target else bare_target for i in words]
                        # we substitute target words with their versions without a POS tag
                else:  # if break didn't trigger, we add such a line to the result
                    chunks = [words[i:i + 510] for i in range(0, len(words), 510)]
                    # BERT has a limit of 512 tokens, we reserve 2 tokens for ['SEP'] and ['CLS']
                    lines.extend([' '.join(chunk) for chunk in chunks])
        else:
            lines = []
            for line in f:
                clean_line = line
                if arguments.corpora_language == 'latin':
                    clean_line = re.sub(r'#\d', '', clean_line)
                words = clean_line.split()
                chunks = [words[i:i + 510] for i in range(0, len(words), 510)]
                # BERT has a limit of 512 tokens, we reserve 2 tokens for ['SEP'] and ['CLS']
                lines.extend([' '.join(chunk) for chunk in chunks])
    with open(os.path.join(arguments.output_folder, f'processed_{os.path.basename(file)}'), 'w', encoding='utf-8') as f:
        f.writelines([f'{line}\n' for line in lines])
    all_lines.extend(lines)
random.shuffle(all_lines)
train_part = int(len(all_lines) * 0.9)
with open(os.path.join(arguments.output_folder, 'train.txt'), 'w', encoding='utf-8') as f:
    f.writelines([f'{line}\n' for line in all_lines[:train_part]])
with open(os.path.join(arguments.output_folder, 'test.txt'), 'w', encoding='utf-8') as f:
    f.writelines([f'{line}\n' for line in all_lines[train_part:]])
