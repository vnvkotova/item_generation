In this repository you will find the code for my master’s thesis “Automatic item generation for personality questionnaires using Transformer models” and parts of the code for a joint paper with Björn Hommel and other researchers [“Transformer-based deep neural language modeling for construct-specific automatic item generation”](https://mediatum.ub.tum.de/doc/1704897/document.pdf).

# Why?
Manual item generation for personality questionnaires is a tedious and difficult task which requires a lot of effort, resources and expert knowledge. The researches have previously tried to automate it either by applying recurrent neural networks or by analyzing cognitive task components to derive schemata. In my thesis I proposed to utilize Transformer models to generate new item instances. 
Such models as BERT and GPT-2 had proven to be very efficient for a wide-range of natural language modelling tasks. However, they are inherently bad suited for short texts, require a lot of computational resources and training samples. Therefore, I modified the models to solve the task of automated item generation. In addition, new metrics to monitor the quality of the training process were introduced, and a new specification of classification task was applied and its effectiveness is evaluated. The generated items and assigned to them personality traits were analyzed by three experts as well as through a survey which is comprised of 50% randomly selected generated items and of 50% randomly selected database items.

# Related work
## Conditional text generation 
The conditional generation of text was inspired by [[1]](#1) and [[2]](#2). In [[2]](#2) Lee and Hsiang divided patent claims in so-called spans. Then they separated them by utilizing a special @@@ tag, add <|startoftext|> at the beginning and <|endoftext|> at the end of each claim and fed thereby prepared data into the middle-sized GPT-2 model. They did it for two reasons. Firstly, by calculating how many times the special tags occur in the generated text the researchers monitored how fast the model learned. Secondly, patent claim spans served as an approximation of an inventive step. 

# Approach 
## Training dataset 
For the research the International Personality Item Pool (IPIP) [[3, 4]](#3) was chosen which consists of 3,320 unlabeled and 1,943 unique labeled items. The main reason for its selection was that it is the largest publicly available dataset with questions for psychological testing. Secondly, it had been used in at least 840 publications (according to the IPIP’s official website1), including the most similar to this research [[1]](#1). 
All items in the pool start with ”I” and are constructed in such a way that they can be answered with one of the following options: ”strongly agree”, ”slightly agree”, ”cannot decide”, ”slightly disagree” or ”strongly disagree”. All labeled items are scored on one or more IPIP scales (in some publications they are called instruments) and have additionally the following characteristics: 
* alpha — Cronbach alpha reliability coefficient of the scale on which the item is scored (sometimes more than one alpha coefficient is reported); 
* key — keyed direction (positive or negative) of the item and its associated construct; 
* label — construct label from the alphabetical list of constructs. 
The goal of the work was to generate pairs (item, label). IPIP contains 254 labels with some of them corresponding to more than one scale. Since the decision has been made to not take scales into account, all items which were assigned the same labels corresponding to different scales were relabeled in such a way that an item does not have duplicated labels. In addition, some labels had only one letter difference between each other, for instance ”Achievement-striving” and ”Achievement-Striving”. In such cases one name was chosen and all other corresponding items were renamed. Therefore, items in the resulting dataset have from 1 to 13 unique labels.  

## Input/Output representations 
The goal of my master’s thesis was to prompt the model to generate label-specific items. Inspired by [[1, 5]](#1), the input to the model was formulated as described below: 
U =<|startoftext|>#l1#l2#...#ln@item<|endoftext|>
where l1, l2, ..., ln are labels which correspond to the item, 
\# separates labels from each other,
@ separates all labels from the item. 

When choosing the delimiters # and @ the advice from gpt-2-keyword-generation3 GitHub repository was followed. Both of the symbols are single, uncommon ASCII characters which are rarely used. An assumption has been made that it would help GPT-2 to learn characters’ significance. Additionally, since GPT-2 is built to write long paragraphs <|startoftext|> token in the beginning and <|endoftext|> at the end of each input sentence are added and the model is asked to stop generating text once it encounters the end token. Start and end tokens were added to the vocabulary, labels were not. 

## Quality metrics 
Prevention of overfitting is a challenging task in unsupervised learning. From the loss function alone it is difficult to understand whether the algorithm learns to generalize or simply ”memorizes” the input. The decrease of a loss value does not necessarily mean the improvement of the weights of the neural network. For supervised methods machine learning practitioners use validation datasets to calculate quality metrics on them during the training process. However, in this research the model was prompted to generate novel items which cannot be compared directly to IPIP constructs. Giving the whole output to a specialist and asking them to evaluate it was not feasible in this study. For these reasons 100 new items are generated after every epoch and the algorithm is evaluated based on them using 14 proposed metrics. The metrics are organized in 4 different groups which measure different aspects of the algorithm. 

### Overfit metrics 
Overfit metrics aim to visualize how well the model generalizes by assessing the similarity between generated and training items and sentences. As the similarity metric Levenshtein distance [[6, 7]](#1) is used as it is fast to compute. It measures the difference between two sequences as the minimum number of single-character edits (insertions, deletions or substitutions) required to change one word into another. Overfit metrics also help to understand other metrics better through showing the number of items/sentences which were generated more than once. 
* Similar items — shows how similar items which were generated after an epoch are to the training items.
* Similar sentences — shows how close sentences which were generated after an epoch are to the training sentences. 
* Repeated items — number of fully identical generated items which is normalized by N. 
* Repeated sentences — shows how many generated sentences are repeated after an epoch. 

### Classification metrics
Classification metrics evaluate the algorithm from the point of view of classification accuracy. Since correct answers are needed in order to calculate accuracy scores, these metrics can be applied only to items/sentences which are present in the training dataset. Therefore, such items are selected and the corresponding labels are examined.  
* Overfitted items — normalized by N number of generated items which are present in the training dataset. 
* Overfitted sentences — displays how many sentences repeat the training input. 
* Overfitted labels — shows whether all labels assigned to an item are correct. 
* F score — F score for items which have identical items in the training database and were assigned one or more wrong labels. 

### Semantic metrics
To assess the meaning of generated items it is proposed to utilize the augmented database. The assumption is that if a neural network is able to slightly modify its input and still preserve the meaning then it is probably capable of generalization. In addition, semantic metrics help to evaluate the classification capabilities of the algorithm better and categorize the generated output. This significantly decreases the time needed for the subsequent manual examination of items by experts. 
* Library items — shows how many generated items coincide with the ones in the augmented database. 
* F score — F score for all items which have fully identical corresponding items in the augmented database.

### Language model distribution metrics
In [[8]](#8) Gehrmann et al. propose to utilize statistical properties to analyze text generated by large language models and to detect fakes. Since most of the modern systems sample tokens from the head of the distribution, the generated text is biased. For instance, the models avoid low-probability words when the entropy for the previously generated context is low, i.e. when the model is (overly) sure of its next prediction. IPIP items have a specific structure which is distinct from sentences which people use either in their everyday life or while writing. Therefore, it is important to see how GPT-2 adapts to the training input. In addition, psychological questions have specific words and terms.  Language model distribution metrics evaluate generated items by comparing probability and entropy [[9]](#9) distributions between generated items and training items. The most similar approach to this one is introduced in [[10]](#10), however no clear pipeline is proposed in the paper.  
* Probability of generated items — shows whether the generated tokens are sampled from the top of the distribution.
* Average probability of training items — analog of probability of generated items calculated for training items. 
* Entropy of generated items — shows whether the generated context is well-known to the used GPT-2. It is calculated by obtaining the entropy for the top 10 tokens 
* Average entropy of training items — evaluates the positional entropy distribution of training items and is computed likewise entropy of generated items.

## Implementation 
The proposed system called item generation is built based on Transformers 3.0.2 python package [[11]](#11). All additional tensor computations and neural network manipulations were performed with PyTorch 1.6.0 [[12]](#12). The simplified architecture for item generation package is presented bellow: <br/>
![Architecture](https://github.com/user-attachments/assets/89e5f45e-a2ea-4504-9823-2c45d643a444)


# Ressources
<a name="1">[1]</a> Bryan McCann, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. The nat- ural language decathlon: Multitask learning as question answering. arXiv preprint arXiv:1806.08730, 2018. \
<a name="2">[2]</a> Jieh-Sheng Lee and Jieh Hsiang. Patent claim generation by fine-tuning openai gpt-2. arXiv preprint arXiv:1907.02052, 2019. \
<a name="3">[3]</a> Lewis R Goldberg et al. A broad-bandwidth, public domain, personality inventory measuring the lower-level facets of several five-factor models. Personality psychology in Europe, 7(1):7–28, 1999. \
<a name="4">[4]</a> Lewis R Goldberg, John A Johnson, Herbert W Eber, Robert Hogan, Michael C Ash- ton, C Robert Cloninger, and Harrison G Gough. The international personality item pool and the future of public-domain personality measures. Journal of Research in personality, 40(1):84–96, 2006. \
<a name="5">[5]</a> Ateret Anaby-Tavor, Boaz Carmeli, Esther Goldbraich, Amir Kantor, George Kour, Segev Shlomov, Naama Tepper, and Naama Zwerdling. Do not have enough data? deep learning to the rescue! In AAAI, pages 7383–7390, 2020. \
<a name="6">[6]</a> Gene Myers. A fast bit-vector algorithm for approximate string matching based on dynamic programming. Journal of the ACM (JACM), 46(3):395–415, 1999. \
<a name="7">[7]</a> Martin Šošić and Mile Šikić. Edlib: a c/c++ library for fast, exact sequence alignment using edit distance. Bioinformatics, 33(9):1394–1395, 2017. \
<a name="8">[8]</a> Sebastian Gehrmann, Hendrik Strobelt, and Alexander M Rush. Gltr: Statistical detection and visualization of generated text. arXiv preprint arXiv:1906.04043, 2019. \
<a name="9">[9]</a> Thomas Lavergne, Tanguy Urvoy, and Franc ̧ois Yvon. Detecting fake content with relative entropy scoring. PAN, 8:27–31, 2008. \
<a name="10">[10]</a> Erion Çano and Ondrej Bojar. Automating text naturalness evaluation of nlg systems. arXiv preprint arXiv:2006.13268, 2020. \
<a name="11">[11]</a> Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Huggingface’s transformers: State-of-the-art natural language processing. ArXiv, abs/1910.03771, 2019. \
<a name="12">[12]</a> Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems 32, pages 8024–8035. Curran Asso- ciates, Inc., 2019.
