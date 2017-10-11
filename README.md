Neural Mention Ranking System for Coreference Resolution
========================================================

This is a reimplementation in Pytorch of the neural mention-ranking system by Sam Wiseman et al.
([ACL 2015](http://www.aclweb.org/anthology/P/P15/P15-1137.pdf)).

Here's how to use it:

1. For corpus preprocessing, you need the tools from Wiseman's original implementation, which you can
   in the [modifiedBCS](https://github.com/swiseman/nn_coref/tree/master/modifiedBCS) subdirectory of
   his [Github repository](https://github.com/swiseman/nn_coref). Follow the instructions there to
   generate the txt features files.

2. Convert the txt files into our own HDF5 format (which is not the same as Wiseman's) by running
   `features.py`. Note that the `main` function in `features.py` contains some hardcoded paths that
   you need to adapt to your system.

3. Train the model with
   ```
   python mention_ranking.py --train training.h5 --dev dev.h5 --checkpoint OUTPREFIX --train-config TRAIN_CONFIG --net-config NET_CONFIG
   ```
   Here, OUTPREFIX is the file name prefix of the model files that will be saved after each epoch.
   TRAIN_CONFIG and NET_CONFIG are JSON files to set up the training process and network configuration.
   If you leave out these options, the defaults in `mention_ranking.py` will be used. There you can
   also find the options that can be set in these files.

4. Create predictions with
   ```
   python mention_ranking.py --predict test.h5 --model MODEL_FILE
   ```
   MODEL_FILE is one of the checkpoint files created during the training run. The predictions will
   be output to stdout in the form of a backpointer file that can be processed with
   `modifiedBCS/WriteCoNLLPreds.sh` in [Wiseman's repository](https://github.com/swiseman/nn_coref).
