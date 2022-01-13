## preprocessing.py

Preprocess corpora: strip target words of their POS tags and generate train/test sets

#### Arguments:

- --targets_path  
  Path to a .txt file with target words
- --corpora_path  
  Paths to corpora separated with ; The entire string with paths has to be surrounded with quotes
- --output_path  
  Path to a folder for processed files

#### Sample usage:

--targets_path Data/targets.txt --corpora_paths "Data/ccoha1.txt;Data/ccoha2.txt" --output_folder Test

## run_mlm.py

Fine-tune a pre-trained BERT model.

#### Key Arguments:

- --model_name_or_path  
  name of the model to fine-tune (from https://huggingface.co/models)
- --train_file  
  path to the file with train corpus
- --validation_file  
  path to the file with validation corpus
- --do_train  
  include this argument to perform training (requires --train_file)
- --do_eval  
  include this argument to perform validation (requires --validation_file)
- --output_dir  
  directory where all the output data will be saved. Specifying an empty directory is recommended. To overwrite
  non-empty existing directory, add --overwrite_output_dir argument
- --line_by_line  
  include this argument if you want each line in corpora to be treated as a single training sequence
- --save_steps (default=500)  
  Determines how many training sequences will be included in a single checkpoint. The less is the number, the more
  checkpoints will be created
- --save_total_limit (default=None)  
  initialize this argument with a number of maximum checkpoints to be created (delete older checkpoints)

#### Sample usage:

python run_mlm.py --model_name_or_path bert-base-uncased --train_file "Processed/train.txt" --validation_file
"Processed/test.txt" --do_train --do_eval --output_dir "Processed/MLM" --line_by_line --save_steps 5000
--save_total_limit 20

## extract_embeddings.py

Extract embeddings for target words (specific layers) from a fine-tuned model.

#### Arguments:

- --model_path  
  path to a .bin PyTorch model
- --model_name  
  name of a model used during fine-tuning
- --targets_path  
  Path to a .txt file with target words
- --corpora_path  
  Paths to PREPROCESSED corpora separated with ; The entire string with paths has to be surrounded with quotes
- --output_path  
  Path to a result file with embeddings
- --bert_layers  
  Bert layers to extract (from 0 to 11). Possible ways to provide:
    - separated with commas (0,4,5)
    - separated with a dash (3-5 is the same as 3,4,5)

  If this argument is omitted, all 12 layers will be extracted
- --concat_layers  
  Pass this argument if you want to concatenate the layers rather than sum them. Disabled by default

#### Result format:

- The embeddings are saved as a dict where:
    - keys are corpora names and values are dicts where:
        - keys are words and values are numpy arrays with embeddings stacked vertically

For example, calling EMBEDDINGS['Data/ccoha1.txt']['attack'][0] we can get the first embedding for the word 'attack'
from 'ccoha1' corpora. EMBEDDINGS['Data/ccoha1.txt']['attack'][1] is the second embedding, and so on.

#### Sample usages:

python extract_embeddings.py --model_path "Processed/pytorch_model.bin" --model_name bert-base-uncased --targets_path
"targets.txt" --corpora_path "Processed/processed_ccoha1.txt;Processed/processed_ccoha2.txt" --output_path "test.pickle"

python extract_embeddings.py --model_path "Processed/pytorch_model.bin" --model_name bert-base-uncased --targets_path
"targets.txt" --corpora_path "Processed/processed_ccoha1.txt;Processed/processed_ccoha2.txt" --output_path "test.pickle"
--bert_layers 0,5,7-9,11 --concat_layers