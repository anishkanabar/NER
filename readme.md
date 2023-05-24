# Fine-tuning Transformers for Named Entity Recognition in Broadcast Police Communications

Abstract

Three Transformer based models – BERT, RoBERTa, and XLNet – were fine-tuned for custom
named entity recognition on a small dataset of police broadcast communications (BPC).
RoBERTa and XLNet demonstrated significantly better performance overall while task specific
pre-training enhanced F1 scores for all three models. Surprisingly, Recall of person names
declined across models despite the task specific pre-training set including person names as a
NER entity.

Introduction

Named Entity Recognition (NER) is essential to Natural Language Processing, both as part of a
pipeline for downstream tasks such as Dialogue Generation and by itself for detecting sensitive
information. Fine-tuning, meanwhile, is critical to NER for domains in which a limited amount
of labeled data is available. One such domain is broadcast police communications (BPC), which
has not yet been studied for NER and for which there is a need for implementation. BPC audio
data for the city of Chicago is publicly available and although transcriptions are not, if they are to
be made available, they would need to be stripped of entities such as names and addresses. If not,
people could be linked to crimes from the transcriptions. Another potential application of NER in
BPC is part of an automated reporting pipeline. Officers typically log their activity, but an
automated pipeline would save time and could potentially be more accurate since it would be
based on real time communications, as opposed to subsequent summarization.

Prior to this study, there was no NER annotated BPC data that could be used for training or
testing a model. However, the success of Transformers pre-trained as language models on vast
datasets in being fine-tuned for specific tasks suggests promise for application on a small BPC
dataset. Accordingly, a subset of a BPC dataset from the University of Chicago was annotated
for addresses, events, dispatch codes, and person names and three state of the art Transformer
based models – BERT, RoBERTa, and XLNet – were compared for performance in recognizing
these named entities. Moreover, each model was evaluated with and without additional pre-
training on the publicly available CoNLL-2003 NER dataset.

Related Work

The primary inspiration for this study is the paper “Evaluating Pretrained Transformer-based
Models on the Task of Fine-Grained Named Entity Recognition” by Lothritz et al (2020). This
paper attempted to answer three questions: Do transformer-based models outperform non-
transformer-based models for NER task? What are the strengths, weaknesses, and trade-offs of
each investigated model? How does the choice of the domain influence the performance of the
models? The dataset employed was the English Wikipedia Named Entity Recognition and Text
Categorization (EWNERTC), a collection of automatically categorized and annotated sentences
from Wikipedia articles. It consists of roughly 7 million annotated sentences, divided into 49
separate domains, though the authors limited the number of entity types per domain to 50 due to
limited examples for some entity types. The five models tested, mentioned earlier, were a CRF, a
Bi-LSTM-CNN-CRF, BERT, RoBERTa, and XLNet. At the time of testing, the BiLSTM-CNN-CRF was 
considered the state of the art model for NER, but the researchers found that the
transformer-based models performed better overall in terms of F1 score and displayed
significantly higher Recall. Finally, a comparison of the F1 scores across domains revealed that
all the models were similarly impacted by domain – either a domain was relatively easy for all
the models or relatively hard for all of them.

Another relevant paper is “Few-shot Learning for Named Entity Recognition in Medical Text”
by Hofer et al (2018). Here, the authors implemented several modifications to the state of the art
model at the time (BiLSTM-CNN-CRF) to optimize it for few-shot learning, which they defined
as learning from a small number of labeled examples, in the medical domain. Six medical NER
datasets were used: one for supervised training and testing, two for supervised pre-training of
weights, and three for unsupervised training of custom word embeddings. In addition, a non-
medical dataset (CoNLL-2003) was used for supervised pre-training of weights. The model
modifications, meanwhile, consisted of layer-wise initialization with pre-trained weights,
hyperparameter tuning, combining pre-training data, custom word embeddings, and removing
out-of-vocabulary words. The authors found the best results when pre-training separately on
datasets of the medical domain, rather than combining pre-training data or pre-training on
CoNLL-2003, using the Nadam optimizer, and customizing the word embeddings by pre-training
them. Ultimately, these improvements yielded an F1 score of 0.7887 compared to 0.6930 for the
baseline model.

Methods

Fine-tuning Dataset

The University of Chicago Urban Resiliency Initiative’s BPC archive consists of approximately
165,000 30-minute continuous recordings of radio transmissions involving members of the
Chicago Police Department and Chicago’s Office of Emergency Management and
Communication from 11 of Chicago’s 13 dispatch zones. Each dispatch zone corresponds to a
specific geographic area in the City of Chicago. Policing personnel operating within each zone
use the same radio frequency to broadcast messages to others (e.g., dispatchers directing officers
based on 911 calls, officers reporting their status, etc.). Official regulations indicate BPC are
highly structured, with transmissions minimized to ensure effective response to emergent
situations, such as officer-involved shootings.

Data Preprocessing

From this archive, over 70,000 transmissions have been manually transcribed. However, the
transmissions have not been annotated and many consist solely of a dispatch code, such as
“Seventy two Robert”, or do not contain any named entities. Therefore, it was necessary to select
transmissions that would be informative for model training. 500 such examples were selected
with the criteria that each must contain at least two different types of named entities. This subset
was then manually annotated for words and phrases that consisted of addresses, events, dispatch
codes, and person names using the doccano open-source text annotation tool.

Some transmissions contain brackets containing speech the transcriber was not certain how to
transcribe; this text was treated the same as unbracketed text. It is also worth noting that
addresses occur both in canonical format (“[number] [name] [street/road/etc.]”) and as
intersections or solely street names. An event, meanwhile, consists of a concise and often used
word or phrase rather than a specific description of what is occurring. For example, “domestic
battery” is an event while “he hit his wife” is not. Finally, dispatch codes and person names may
also occur in canonical form (“[a number] Robert” for dispatch codes and “[first name] [last
name]” for person names) or as fragments of the above.

Each transmission was pre-tokenized and aligned with the respective BILOU (Beginning Inside
Last Outside Unit) NER tags. As an example, “Central Park and Irving” was tagged as “B-
address, I-address, I-address, L-address”. Pre-tokenization resulted in 3,665 tokens for address,
1,116 tokens for event, 498 tokens for dispatch code, 169 tokens for person name, and 10,
tokens for non-entity (~65% of the total). These tokens were further segmented as required by
the respective model scheme.

Models:

I. BERT

BERT, which stands for Bidirectional Encoder Representations from Transformers (Devlin et al
2018), improves the Transformer architecture by introducing bi-directional sequence encoding.
To prevent each word from being able to “see itself”, particularly when multiple layers are
utilized, BERT is pre-trained using a masked language model: 15% of the tokens are masked at
random and the task is to predict those masked tokens. Furthermore, each training example is
comprised of a pair of sentences and a second task is to predict whether one sentence follows the
other. The training loss is the sum of the mean masked language model likelihood and the mean
next sentence prediction likelihood. Finally, inputs to the model are passed as WordPiece
embeddings with a 30,000 token vocabulary.

II. RoBERTa

The “Robustly Optimized BERT approach” (Liu et al 2019) implemented the following
advancements: dynamic masking of tokens, the removal of the next sentence prediction
objective, training with larger batches, and utilizing Byte-Pair Encoding as opposed to
WordPiece embeddings. Dynamic masking entailed duplicating the training data 10 times so that
each sequence could be masked in 10 different ways over the 40 epochs of training. This
prevented the same tokens from being masked in each epoch and thus lost from the vocabulary.

Increasing the batch size from 256 sequences to 2,000 and then 8,000 yielded decreases in model
perplexity while removing the next sentence prediction task did not degrade performance. The
switch from WordPiece embeddings to Byte-Pair Encoding with 50,000 subword units increased
the number of trainable parameters by 15M compared with BERT.

III. XLNet

XLNet (Yang et al 2019) employs an auto-regressive language model within the Transformer
architecture that conditions on all permutations of word tokens in a sentence, as opposed to just
those to the left or just those to the right of the target token. In this way, it enables learning
bidirectional contexts like BERT yet does not depend on masking to prevent data leakage and
thus does not suffer from a pretrain-finetune discrepancy. Furthermore, XLNet incorporates
segment-level recurrence from the Transformer-XL model. Segment-level recurrence entails
fixing the representation computed for the previous segment and reusing it as an extended
context for the next segment. This allows contextual information to persist across segment
boundaries, which often arbitrarily divide sentences.

Task Specific Pre-training Dataset

As an additional experiment, the models were pre-trained on a dataset that has been annotated for
NER. The CoNLL-2003 (English) data was taken from the Reuters Corpus, which consists of
Reuters news stories between August 1996 and August 1997 (Tjong, Sang, and De Meulder
2003). The data are annotated for the following entities: location (10,645 tokens), person name
(10,059 tokens), organization (9,323 tokens), and miscellaneous (5,062 tokens). 276,388 of the
tokens do not correspond to named entities (~89% of total tokens). Two thirds of the data were
designated by the authors as a training set and those same samples were used to pre-train the
Transformers tested here.

Training Parameters

The Hugging Face base implementation was used for all three models. 400 samples were
randomly selected for training and 100 for testing (80/20 split) with a batch size of 16 across all
runs. A weight decay of 1e-5, learning rate of 1e-4, and dropout of 0.1 was used for training the
models; all default parameters. For pretraining on CoNLL-2003, the learning rate was 2e-5, the
weight decay was 0.01 and the models were trained for 3 epochs; also default parameters. The
models were fine-tuned for between 2 and 10 epochs and the scores of the best performing
version by F1 score were reported. For BERT, this was the model trained for 4 epochs without
pre-training and 6 epochs with pre-training. For RoBERTa this was the model trained for 5
epochs without pre-training and 8 epochs with pre-training. For XLNet this was the model
trained for 8 epochs without pre-training and 5 epochs with pre-training.

Results

Figure 2: Fine-tuned Model Performance
```
          Accuracy  Precision Recall    F1
BERT      0.956     0.811     0.863     0.836
RoBERTa   0.951     0.886     0.911     0.898  
XLNet     0.949     0.847     0.917     0.881
```

Figure 3: Effect of Task Specific Pre-training 
```
          Accuracy            Precision           Recall              F1
BERT      0.956 0.941         0.811 0.852         0.863 0.886         0.836 0.869
RoBERTa   0.951 0.962         0.886 0.895         0.911 0.926         0.898 0.910
XLNet     0.949 0.954         0.847 0.875         0.917 0.907         0.881 0.891
```
Figure 4: Recall by Named Entity Type and Effect of Task Specific Pre-training
```
          Address             Event               Code                Name
BERT      0.875 0.914         0.898 0.932         0.836 0.821         0.737 0.455
RoBERTa   0.954 0.954         0.900 0.936         0.847 0.876         0.885 0.808
XLNet     0.944 0.954         0.940 0.937         0.839 0.799         0.759 0.586
```
Discussion

The three main findings of this study are: 1) RoBERTa and XLNet outperform BERT on NER in
the BPC domain (Figure 2), 2) pre-training on a general NER dataset improves the performance
of all three models in BPC NER (Figure 3), and 3) pre-training may not improve – and may even
degrade – performance in detecting an entity that occurs both in the task specific pre-training set
and the fine-tuning set (Figure 4). The fact that RoBERTa outperforms BERT is not surprising,
considering that RoBERTa was trained on 10 times as much data and contains 15M more
parameters. However, XLNet-base was trained on the same amount of data (16 GB) and contains
the number of parameters (110M), so it seems likely that replacing masked language modeling
with permutation language modeling in an architectural improvement. Since XLNet is forced to
make predictions in situations with less information, for example predicting the 4th word when
only the 2nd and 6th are known, it makes sense that it can extract more information from the same
amount of text. BERT, by contrast, only makes two predictions per example and neglects the
potential relationship between the two masked tokens. To examine the robustness of RoBERTa,
it would be interesting to observe whether RoBERTa-large would outperform XLNet-large,
which contain a similar number of parameters (340M vs 355M) and are trained on the same
amount of data (160GB).

The improvement of model performance following task specific pre-training stands in contrast to
what was observed by Hofer et al. In their study, pre-training on CoNLL-2003 yielded an F
score of 0.6847 compared with 0.6930 for a random weight initialization. However, when they
solely pre-trained the BiLSTM layer, the F1 increased to 0.7119. Therefore, the benefit of task
specific pre-training may be architecture dependent. Unfortunately, a BPC domain specific
dataset is not available, so it is not possible to determine whether pre-training on one would yield
further improvement in performance. There are other radio transmission datasets available, such
the Air Traffic Control archive, so if it is feasible to annotate this dataset for NER it may be
worth exploring as a pre-training option.

Nonetheless, the decrease in performance in detecting person names suggests that the transfer of
weights from one dataset to another is not as straightforward as it may seem. It is worth noting
that person name is the smallest entity category in the BPC dataset with only 169 tokens
compared to 10,059 tokens in the CoNLL-2003 dataset. Consequently, the weights may have
been overly influenced by how person names occur in the Reuters based pre-training set. For
instance, a common way in which names occur in BPC is “We have a [first name]” rather than
“[first name] [last name] called the police”, which one might read in a news article. Once the
BPC dataset is fully annotated and the models can be trained on more data, it is possible that bias
toward the pre-train set for person names will diminish or even that Recall will improve with
pre-training for this entity.

References

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805,
2018.

Maximilian Hofer, Andrey Kormilitzin, Paul Goldberg, and Alejo Nevado-Holgado. Few-shot
learning for named entity recognition in medical text. arXiv preprint arXiv:1811.05468,
2018.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike
Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert
pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

Erik F Tjong, Kim Sang, and Fien De Meulder. Introduction to the conll-2003 shared task:
Language-independent named entity recognition. In CoNLL, 2003.

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V
Le. XLNET: Generalized autoregressive pretraining for language understanding. In:
Advances in neural information processing systems, pp. 5754–5764, 2019.
