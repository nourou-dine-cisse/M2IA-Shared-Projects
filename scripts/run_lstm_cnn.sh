#!/bin/bash
# To run from the root of the repository: bash scripts/run_lstm_cnn.sh

MEDLINE=TP_ISD2020/QUAERO_FrenchMed/MEDLINE/MEDLINE
PRESS=TP_ISD2020/QUAERO_FrenchPress/fra4_ID

mkdir -p logs

for model in lstm cnn
do
    for embeddings in random w2v_cbow_med w2v_skipgram_med ft_cbow_med w2v_cbow_press w2v_skipgram_press ft_cbow_press
    do
        if [ "$embeddings" = "random" ]
        then
            embeddings_file=random
        else
            embeddings_file=embeddings/$embeddings.bin
        fi

        echo "MEDLINE | $model | $embeddings"
        python scripts/ner_lstm_cnn.py --model $model --embeddings $embeddings_file \
            --train ${MEDLINE}train_layer1_ID.conll --valid ${MEDLINE}dev_layer1_ID.conll --test ${MEDLINE}test_layer1_ID.conll \
            --epochs 100 > logs/medline_${model}_${embeddings}.txt

        echo "PRESS | $model | $embeddings"
        python scripts/ner_lstm_cnn.py --model $model --embeddings $embeddings_file \
            --train ${PRESS}.train --valid ${PRESS}.dev --test ${PRESS}.test \
            --epochs 40 > logs/press_${model}_${embeddings}.txt
    done
done
