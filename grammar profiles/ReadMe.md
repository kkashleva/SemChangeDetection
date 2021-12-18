
Grammatical Profiling for Semantic Change Detection (Mario Giulianelli, Andrey Kutuzov, Lidia Pivovarova, 2021)

https://github.com/glnmario/semchange-profiling

- stanza_process.py produces parsed CONLL files from raw texts;
- collect_ling_stats.py reads a CONLL file and dumps frequencies for morphological and syntax properties of the target words into JSON files;
- compare_ling.py produces TSV files with binary and graded change predictions from two JSON files;
- merge.py reads two JSON files (e.g. with morphological and syntax properties) and produces averaged predictions.

Use eval.py to evaluate the resulting TSVs with regards to gold scores (in the test_data_truth subdirectory).

The raw profiles (feature counts) could be found in /features folder.
