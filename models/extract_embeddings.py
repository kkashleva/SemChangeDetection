import argparse
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as padder
import tqdm
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

bert_layers = [i for i in range(0, 12)]  # default value


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_path', type=str, help="Path to a .bin PyTorch model")
    arg_parser.add_argument('--model_name', type=str, help="Name of a model used")
    arg_parser.add_argument('--targets_path', type=str, help="Path to a .txt file with target words")
    arg_parser.add_argument('--corpora_paths', type=str, help="Paths to corpora separated with ;")
    arg_parser.add_argument('--corpora_language', type=str, help="english, german, swedish or latin")
    arg_parser.add_argument('--output_path', type=str, help="Path to a result file with embeddings")
    arg_parser.add_argument('--bert_layers', type=str,
                            help="Bert layers to extract (from 0 to 11). Possible ways to provide: "
                                 "1) separated with commas (0,4,5); "
                                 "2) separated with dash (3-5 same as 3,4,5). "
                                 "If this argument is omitted, all 12 layers will be extracted")
    arg_parser.add_argument('--concat_layers', action='store_true',
                            help="Pass this argument if you want to concatenate the layers rather than sum them. "
                                 "Disabled by default")
    args = arg_parser.parse_args()
    if not args.model_path:
        print('Please specify path to PyTorch model (--model_path)')
    elif not args.model_name:
        print('Please specify model name (--model_name)')
    elif not args.targets_path:
        print('Please specify path to target words (--targets_path)')
    elif not args.corpora_paths:
        print('Please specify path to corpora separated with ; (--corpora_paths)')
    elif not args.output_path:
        print('Please specify output path (--output_path)')
    elif not args.corpora_language or args.corpora_language not in ['english', 'german', 'swedish', 'latin']:
        print('Please specify a valid corpora language (--corpora_language)')
    else:
        if args.bert_layers:
            global bert_layers
            bert_layers = []
            spl = [i.strip() for i in args.bert_layers.split(',')]
            for part in spl:
                subpart = [i.strip() for i in part.split('-')]
                try:
                    bert_layers.extend([i for i in range(int(subpart[0]), int(subpart[-1]) + 1)])
                except ValueError:
                    print('--bert_layers argument could not be parsed. Please refer to --help')
                    return
            if not bert_layers or any(i < 0 or i > 11 for i in bert_layers):
                print('--bert_layers argument could not be parsed. Please refer to --help')
                return
        return args


def strip_accents(word):
    translator = word.maketrans("äöü", "aou")
    return word.translate(translator)


def prepare_batch_for_bert(batch: list, tokenizer):
    """Add special tokens to sequences from corpus and pad them to have the same length."""
    batch_tensors = []
    texts = []
    max_len = 0
    for batch_line in batch:
        text = ['[CLS]'] + tokenizer.tokenize(batch_line) + ['[SEP]']
        max_len = max(max_len, len(text))
        indexed_tokens = tokenizer.convert_tokens_to_ids(text)
        batch_tensors.append(torch.tensor([indexed_tokens]))
        texts.append(text)
    batch_tensors = [padder.pad(i, (0, max_len - len(i[0]))) for i in batch_tensors]  # zero padding
    return torch.cat(batch_tensors), texts


def check_for_split_tokens_bert(tokens: list[str]):
    """Here we analyze a tokenized line and scan it for split words (for example, [ab, ##bauen]).
    We glue such words together and memorize their indices in the original tokenized line."""
    tokens_with_indexes = []
    current_index = len(tokens) - 1
    # We move in reverse direction, from the last token to the first one
    while current_index >= 0:
        current_token = tokens[current_index]
        if not current_token.startswith('##'):
            # Then it's not a split word, we just add it to the result with its index
            tokens_with_indexes.append((current_token, [current_index]))
            current_index -= 1
        else:
            # It's a split word. We are now standing at the tail of this word
            token_indexes = [current_index]  # First we add the tail index of this word to the list
            full_token = current_token.strip('##')  # Then we strip the word part of "##" and add this part to a string
            sub_index = current_index - 1  # Move one step towards the beginning of the line (and the head of the word)
            while tokens[sub_index].startswith('##'):
                # The word can be split into multiple parts.
                # This loop runs only if the word is split in more than two parts
                # This loop processes all middle parts of the word (non-head and non-tail)
                full_token = tokens[sub_index].strip('##') + full_token  # glue the part to the front of the word
                token_indexes.append(sub_index)
                sub_index -= 1
            # Finally we process the head of the split word. It doesn't have "##"
            full_token = tokens[sub_index] + full_token
            token_indexes.append(sub_index)
            token_indexes.reverse()  # Because we moved from the tail towards the head (done mostly for convenience)
            tokens_with_indexes.append((full_token, token_indexes))
            current_index = sub_index - 1
    tokens_with_indexes.reverse()  # Also for convenience
    return tokens_with_indexes
    # So this list contains tuples of full words and indices of their parts in BERT-tokenized line


def check_for_split_tokens_roberta(tokens: list[str]):
    """Here we analyze a tokenized line and scan it for split words (for example, [Ġcont, ineo]).
        We glue such words together and memorize their indices in the original tokenized line."""
    tokens_with_indexes = []
    current_index = len(tokens) - 1
    # We move in reverse direction, from the last token to the first one
    while current_index >= 0:
        current_token = tokens[current_index]
        if current_token.startswith('Ġ') or current_token in ['[SEP]', '[CLS]'] or tokens[current_index - 1] == '[CLS]':
            # Then it's not a split word, we just add it to the result with its index
            tokens_with_indexes.append((current_token.strip('Ġ'), [current_index]))
            current_index -= 1
        else:
            # It's a split word. We are now standing at the tail of this word
            token_indexes = [current_index]  # First we add the tail index of this word to the list
            full_token = current_token  # Then we initialize the full token with its tail
            sub_index = current_index - 1  # Move one step towards the beginning of the line (and the head of the word)
            while not tokens[sub_index].startswith('Ġ'):
                # The word can be split into multiple parts.
                # This loop runs only if the word is split in more than two parts
                # This loop processes all middle parts of the word (non-head and non-tail)
                full_token = tokens[sub_index] + full_token  # glue the part to the front of the word
                token_indexes.append(sub_index)
                sub_index -= 1
            # Finally we process the head of the split word. It has to start with 'Ġ' token
            full_token = tokens[sub_index].strip('Ġ') + full_token
            token_indexes.append(sub_index)
            token_indexes.reverse()  # Because we moved from the tail towards the head (done mostly for convenience)
            tokens_with_indexes.append((full_token, token_indexes))
            current_index = sub_index - 1
    tokens_with_indexes.reverse()  # Also for convenience
    return tokens_with_indexes


def get_bert_embeddings(tokens_tensor, texts, model, dataset_name):
    """Calculate embeddings for target words and write them to total_embeddings"""
    segments_tensors = torch.ones(len(tokens_tensor), len(tokens_tensor[0]), dtype=torch.long)
    with torch.no_grad():
        try:
            outputs = model(tokens_tensor, segments_tensors)
        except (RuntimeError, IndexError) as e:
            print(e)
            return
        hidden_states = outputs.hidden_states[1:]  # the first state is the input state, we don't need it
    token_embeddings = torch.stack(hidden_states, dim=0)
    # initial order: layer; batch number; token (word); embeddings dimension
    token_embeddings = token_embeddings.permute(1, 2, 0, 3)  # permuting dimensions for easy iterating
    # after permuting: batch number; token (word); layer; embeddings dimension
    for i_batch, batch in enumerate(token_embeddings):  # iterating over batches
        if 'roberta' in arguments.model_name:
            tokens_with_indices = check_for_split_tokens_roberta(texts[i_batch])
        else:
            tokens_with_indices = check_for_split_tokens_bert(texts[i_batch])
        for token_indices_tuple in tokens_with_indices:
            word = token_indices_tuple[0]
            indices = token_indices_tuple[1]
            if word in targets:  # we only want embeddings for target words
                average_embeddings = sum(token_embeddings[i_batch][indices]) / len(indices)
                # If the word is split and has embeddings for each of its part, we calculate the average of its part
                # embeddings for each layer separately
                if arguments.concat_layers:
                    sum_vec = torch.concat(tuple(average_embeddings[bert_layers]), dim=0).numpy()
                else:
                    sum_vec = torch.sum(average_embeddings[bert_layers], dim=0).numpy()
                if targets[word] not in total_embeddings[dataset_name]:
                    total_embeddings[dataset_name][targets[word]] = sum_vec  # initial embedding
                else:  # we just stack the new embedding at the bottom of the previous ones
                    total_embeddings[dataset_name][targets[word]] = np.vstack(
                        [total_embeddings[dataset_name][targets[word]], sum_vec])


arguments = parse_args()
if not arguments:
    exit(1)
print(f"The following BERT layers will be extracted: {bert_layers}. ", end="")
if arguments.concat_layers:
    print("The layers will be concatenated.")
else:
    print("The layers will be summed.")
fine_tuned_model = torch.load(arguments.model_path, map_location=torch.device('cpu'))
if 'roberta' in arguments.model_name:
    bert_model = RobertaModel.from_pretrained(arguments.model_name, output_hidden_states=True,
                                              state_dict=fine_tuned_model)
    bert_tokenizer = RobertaTokenizer.from_pretrained(arguments.model_name, state_dict=fine_tuned_model)
else:
    bert_model = BertModel.from_pretrained(arguments.model_name, output_hidden_states=True, state_dict=fine_tuned_model)
    bert_tokenizer = BertTokenizer.from_pretrained(arguments.model_name, state_dict=fine_tuned_model)
bert_model.eval()
with open(arguments.targets_path, encoding='utf-8-sig') as f:
    if arguments.corpora_language == 'english':
        # discarding part of speech tags in the end
        targets = {word.strip()[:-3]: word.strip() for word in f.readlines()}
    elif arguments.corpora_language == 'swedish':
        # stripping accents because BERT does it
        targets = {strip_accents(word).strip(): word.strip() for word in f.readlines()}
    else:
        targets = {word.strip(): word.strip() for word in f.readlines()}
datasets = arguments.corpora_paths.split(';')
total_embeddings = {i: {} for i in datasets}  # embeddings for all datasets here. Filled in get_bert_embeddings
for dataset in datasets:
    print(f'Processing file: {dataset}\n')
    current_batch = []
    batch_counter = 0
    with tqdm.tqdm(total=os.path.getsize(dataset)) as progress_bar:
        with open(dataset, encoding='utf-8-sig') as f:
            for line in f:
                progress_bar.update(len(line))
                for target in targets:
                    if arguments.corpora_language == 'swedish':
                        text_target = targets[target]
                    else:
                        text_target = target
                    if text_target in line.split():
                        current_batch.append(line.strip())
                        batch_counter += 1
                        if not batch_counter % 8:
                            tokenized_batch, tokenized_text = prepare_batch_for_bert(current_batch, bert_tokenizer)
                            get_bert_embeddings(tokenized_batch, tokenized_text, bert_model, dataset)
                            current_batch = []
                            batch_counter = 0
                        break

with open(arguments.output_path, 'wb') as f:
    pickle.dump(total_embeddings, f)
