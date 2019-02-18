Get `YouTubeClips.tar` from `here <http://www.cs.utexas.edu/users/ml/clamp/videoDescription/>`_.
Use ``tar -xvf YouTubeClips.tar`` to decompress the archive.
And, follow the link to download the annotations.

Now, visit `this repo <https://github.com/yaoli/arctic-capgen-vid>`_.
Follow the "preprocessed YouTube2Text download link."
We'll be throwing away the Googlenet features for now. We just need the captions there.
Use ``unzip youtube2text_iccv15.zip`` to decompress the files.

Get to the following directory structure: ::

    yt2t
    |-YouTubeClips
    |-youtube2text_iccv15

Change directories to ``yt2t``. We'll rename everything to follow the "vid#.avi" format:

.. code-block:: python

    import pickle
    import os


    YT = "youtube2text_iccv15"
    YTC = "YouTubeClips"

    with open(os.path.join(YT, "dict_youtube_mapping.pkl"), "rb") as f:
        yt2vid = pickle.load(f, encoding="latin-1")

    for f in os.listdir(YTC):
        hashy, ext = os.path.splitext(f)
        vid = yt2vid[hashy]
        fpath_old = os.path.join(YTC, f)
        f_new = vid + ext
        fpath_new = os.path.join(YTC, f_new)
        os.rename(fpath_old, fpath_new)

Now we want to convert the frames into sequences of CNN feature vectors.
Use `tools/img_feature_extractor.py`.

Now we turn our attention to the annotations:

.. code-block:: python

    import pickle
    import os


    YT = "youtube2text_iccv15"

    with open(os.path.join(YT, "CAP.pkl"), "rb") as f:
        ann = pickle.load(f, encoding="latin-1")

    vid2anns = {}
    for vid_name, data in ann.items():
        for d in data:
            try:
                vid2anns[vid_name].append(d["tokenized"])
            except KeyError:
                vid2anns[vid_name] = [d["tokenized"]]

    with open(os.path.join(YT, "train.pkl"), "rb") as f:
        train = pickle.load(f, encoding="latin-1")

    with open(os.path.join(YT, "valid.pkl"), "rb") as f:
        val = pickle.load(f, encoding="latin-1")

    with open(os.path.join(YT, "test.pkl"), "rb") as f:
        test = pickle.load(f, encoding="latin-1")

    train_files = open("yt2t_train_files.txt", "w")
    val_files = open("yt2t_val_files.txt", "w")
    test_files = open("yt2t_test_files.txt", "w")

    train_cap = open("yt2t_train_cap.txt", "w")
    val_cap = open("yt2t_val_cap.txt", "w")
    test_cap = open("yt2t_test_cap.txt", "w")

    for vid_name, anns in vid2anns.items():
        vid_path = vid_name + ".npy"
        for i, an in enumerate(anns):
            an = an.replace("\n", " ")
            split_name = vid_name + "_" + str(i)
            if split_name in train:
                train_files.write(vid_path + "\n")
                train_cap.write(an + "\n")
            elif split_name in val:
                val_files.write(vid_path + "\n")
                val_cap.write(an + "\n")
            else:
                assert split_name in test
                test_files.write(vid_path + "\n")
                test_cap.write(an + "\n")

Preprocess the data with

.. code-block:: bash

    python preprocess.py -data_type vec -train_src yt2t/yt2t_train_files.txt -src_dir yt2t/r152/ -train_tgt yt2t/yt2t_train_cap.txt -valid_src yt2t/yt2t_val_files.txt -valid_tgt yt2t/yt2t_val_cap.txt -save_data data/yt2t --shard_size 1000

Train with

.. code-block:: bash

    python train.py -data data/yt2t -save_model yt2t-model -world_size 2 -gpu_ranks 0 1 -model_type vec -batch_size 64 -train_steps 10000 -valid_steps 500 -save_checkpoint_steps 500 -encoder_type brnn -optim adam -learning_rate .0001 -feat_vec_size 2048