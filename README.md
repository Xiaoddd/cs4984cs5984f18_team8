# CS4984/CS5984 F2018 Team8

## About the files
- Use `pip install -r requirements.txt` to install dependencies.
- `classification.py` is to classify documents into relevant and irrelevant
- `preproc_lda.py` is to preprocess the corpus and conduct LDA analysis, then visualize the results
- `preproc_lda2vec.py` is to preprocess the corpus and conduct lda2vec analysis, then visualize the results
- `make_datafiles_py3.py` is a Python 3 script to process data for pointer generator, refer to https://github.com/chmille3/process_data_for_pointer_summrizer
- `pointer-generator/run_summarization.py`: use pretrained model and vocabulary to run decode on our dataset.

## Pretrained model
Please download and unzip [pretrained model for TensorFlow 1.2.1](https://drive.google.com/file/d/0B7pQmm-OfDv7ZUhHZm9ZWEZidDg/view) from the [source git](https://github.com/abisee/pointer-generator). Create a "log" folder in the root, and then extract the content into it.

Note: only keep "eval" and "train" folders, remove the decoding result folder "decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410"

## Vocabulary
Please download and unzip [finished_files.zip](https://drive.google.com/uc?id=0BzQ6rtO2VN95a0c3TlZCWkl3aU0&export=download), and then extract the vocab file to "finished_files" folder


## Run beam search decoding
`python pointer-generator/run_summarization.py --mode=decode --data_path=finished_files/chunked/test_* --vocab_path=finished_files/vocab --log_root=log --exp_name=pretrained_model` or

`python pointer-generator/run_summarization.py --mode=decode --data_path=finished_files/chunked/test_* --vocab_path=finished_files/vocab --log_root=log --exp_name=pretrained_model  --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1`

See PGN notes in team drive folder [here](https://docs.google.com/document/d/1NBGj4VNMaDFwaC9hH2kEocSUTVGJ_36x6D7ckX08EXU/).

## Run Attention Visualizer (Attn-vis)
To run the visualizer, navigate to your root folder of the project (where "attn_vis" and "log" folders are located), then

in Python 2: run `python -m SimpleHTTPServer`

in Python 3: run `python3 -m http.server`

and then open http://localhost:8000/attn_vis/ to view. It random displays an article in each page refresh.

Note: The `SimpleHTTPServer` module has been merged into `http.server` in Python 3.0.
