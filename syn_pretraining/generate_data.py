from ngram_generator import NGramGenerator, NGramGeneratorOnline, NGramGeneratorOnlineNumpy, RandomTokenGenerator
import argparse
import os
import csv
from tqdm import tqdm
import torch.multiprocessing as mp
import pandas as pd
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvocab", type=int)
    parser.add_argument("--ngram", type=int, default=0)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--raw", action="store_true", default=False)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--online", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--random", action="store_true", default=False)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.random:
        gen = RandomTokenGenerator(args.nvocab, args.seed, num_workers=args.num_workers)
    else:
        if args.online:
            gen = NGramGeneratorOnlineNumpy(ngram=args.ngram, nvocab=args.nvocab, seed=args.seed, temperature=args.temperature, num_workers=args.num_workers)
        else:
            gen = NGramGenerator(ngram=args.ngram, nvocab=args.nvocab, seed=args.seed, temperature=args.temperature)
    if args.random:
        filepath = os.path.join(args.outdir, "data_random_nvocab_" + str(args.nvocab) + ".csv")
    else:
        filepath = os.path.join(args.outdir, "data_ngram_" + str(args.ngram) + "_nvocab_" + str(args.nvocab) + "_temperature_" + str(args.temperature) + ".csv")
    with open(filepath, "w") as f:
        writer = csv.writer(f)
        print("saving to", filepath)
        for itr in tqdm(range(args.iterations)):
            batch = gen.generate(itr, max_length=args.length, batch_size=args.batch_size)
            if args.raw:
                writer.writerows(batch.tolist())
            else:
                raise NotImplementedError
