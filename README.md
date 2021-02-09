# kdd_visiual_concept

## Requirements
  + tensorflow-gput==1.13.1
  + keras==2.2.4
  + gensim==3.8.0
  + nltk==3.5

## download images
```
  python visionTag/weblySupervised.py --action download -i 0 -j 73183
```

## webly-supervised initialization
```
  python visionTag/weblySupervised.py \
        --action filter \
        --candidate $candidate_negative_file \
        --image_path $image_path
```

## train the visual concept classifier
```
  python visionTag/train.py \
        --action train_with_bert \
        --lr $lr \
        --train visionTag/data/ILSVRC_train-1.txt \
        --test visionTag/data/test_data.txt \
        --model model-1.h5 \
        --log log.csv \
        --epoch 20 \
        [--load $load_model \ # if pre-trained model is available]
        [--with_image # if multi-modal model is used]
```

## get the predictions
```
  python visionTag/train.py \
        --action predict \
        --unlabel $unlabel_file \
        --load $load_file \
        --new_positive $new_positive \
        --new_negative $new_negative
```