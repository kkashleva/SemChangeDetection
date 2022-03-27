# python3
# coding: utf-8

import argparse
import logging
import numpy as np
import csv

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--input1", "-i1", help="Path to a tsv file 1", required=True)
    arg("--input2", "-i2", help="Path to a tsv file 2", required=True)
    arg("--average", "-a", help="If 'True', average score is used instead of max",
        required=False, type=bool, default=False, const=True, nargs="?")
    args = parser.parse_args()

    words1 = {}
    words2 = {}

    for line in open(args.input1, "r"):
        word, val = line.strip().split("\t")
        words1[word] = float(val)

    for line in open(args.input2, "r"):
        word, val = line.strip().split("\t")
        words2[word] = float(val)

    assert words1.keys() == words2.keys()

    rows = []

    for word in words1:
        if args.average:
            final = np.mean([words1[word], words2[word]])
        else:
            final = np.max([words1[word], words2[word]])
        if "binary" in args.input1:
            final = int(final)
        row = []
        row.append(word)
        row.append(final)
        rows.append(row)
        print(f"{word}\t{final}")

    head = ['word', 'final']

    with open('results.tsv', 'w', newline='') as csvfile:
     writer = csv.writer(csvfile, delimiter='\t')
     writer.writerow(head)
     for row in rows:
        writer.writerow(row)


