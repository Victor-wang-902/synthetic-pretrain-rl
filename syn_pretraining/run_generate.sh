#Generate MC data with different state sizes
python generate_data.py --nvocab 10 --ngram 1 --temperature 1.0 --raw --outdir /path/to/data/ --num_workers 8 --batch_size 256 --iterations 500 --online
python generate_data.py --nvocab 100 --ngram 1 --temperature 1.0 --raw --outdir /path/to/data/ --num_workers 8 --batch_size 256 --iterations 500 --online
python generate_data.py --nvocab 1000 --ngram 1 --temperature 1.0 --raw --outdir /path/to/data/ --num_workers 8 --batch_size 256 --iterations 500 --online
python generate_data.py --nvocab 10000 --ngram 1 --temperature 1.0 --raw --outdir /path/to/data/ --num_workers 32 --batch_size 1024 --iterations 125 --online
python generate_data.py --nvocab 100000 --ngram 1 --temperature 1.0 --raw --outdir /path/to/data/ --num_workers 48 --batch_size 1536 --iterations 84 --online


#Generate IID randomized data
python generate_data.py --nvocab 100 --raw --outdir /path/to/data/ --num_workers 32 --batch_size 1024 --iterations 125 --online --random
