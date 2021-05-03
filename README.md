Tristan Shah

17 Feb 2021

# Short abstract

In this paper we evaluate text generated by an LSTM based Language Model. The model is trained using a text corpus mined from Leo Tolstoy’s major writings. We show that increasing the number of context words given to the model allows it to produce text which scores similarly to Tolstoy on readability formulas.

# Medium abstract (\<200 words)

In this paper we train Long Short-Term Memory cell based Language Models to predict the next word in a sequence of text given a number of context words. Language Models with the ability to predict the next word in a sequence have a unique property of being able to generate text similar to that which it is trained on. Number of context words given to the models are varied to measure the effect on scores of readability formulas and perplexity on a sample of text. Using evaluation methods on language models which measure ability to write and read will give an indication of the effect of supplying additional context words on the model's performance. After the models are trained on a corpus of text data mined from the novels of Leo Tolstoy, they are evaluated using both methods. Our results show that increasing the number of context words allows the models to generate text which achieves similar scores to Tolstoy on readability formulas. They also show that more context reduces average perplexity on writings of Tolstoy which are withheld from the models during training.

# Introduction

Leo Tolstoy is considered by many to be one of the greatest Russian authors. His work is commonly studied in High School and College literature courses. The size and quantity of his novels creates an opportunity for analysis with Natural Language Processing (NLP). In this study NLP modeling techniques will be applied to a text corpus consisting of many of Tolstoy's major novels. The specific modeling techniques as well as the evaluation methods for the models will be discussed next.

Long Short-Term Memory (LSTM) cells are a type of Neural Network capable of processing sequential information. They have been shown to preform very well on many Language Modeling (LM) tasks including speech recognition^1^, sentiment analysis^2^, and next word prediction^3^. When LSTMs are used to predict the next word given a sequence of context words, they form an autoregressive process which is able to generate new text. Evaluation of LMs is usually done with measures such as perplexity^4^. The formula for perplexity is shown in Equation 1. Perplexity provides an indication of how well the LSTM-LM predicts next words on some sample text. High values of perplexity indicate that the LM is perplexed, and not accurately predicting the text. A shortcoming of this method is that it lacks the ability to evaluate the readability of generated text.

$$\text{PP}_{T}\left( p_{M} \right) = \frac{1}{{(\prod_{i = 1}^{t}{p_{M}(w_{i}|w_{1}\cdots w_{i - 1}))}}^{\frac{1}{t}}}$$

Equation : Formula for perplexity of a language model a sequence of text. Given in Evaluation Metrics for Language Models^4^. $\text{PP}_{T}$ is perplexity and $p_{M}$ is a language model which takes in context words $w_{1}\cdots w_{i - 1}$and predicts the probability of next word $w_{i}$ for sequence length $t$.

Readability Formulas (RF) are a class of simple algorithms which calculate a score for a piece of text. The score of an RF differs for each algorithm, as they are used to measure different qualities of the given text such as reading ease and grade level required for reading. Most RFs are a linear function on two components, sentence complexity and word complexity. However, they differ in the way they measure and weight each component. RF have been developed by the United States Navy^5^ to create training material for enlisted personnel. And in clinical trials^6,7^ to establish informed consent. We evaluate LSTM-LM generated text with two widely used RF: Automated Readability Index (ARI) and Gunning Fog Index (FOG).

The ARI formula shown in Equation 2 is used to predict the grade level required by a reader to understand a piece of text. Unlike most other RF it uses character level information to make a prediction. Under the ARI formula, more difficult text contains longer words and sentences.

$$ARI = 4.71 \bullet ACW + 0.5 \bullet AWS - 21.43$$

Equation : ARI for calculating U.S grade level. ACW and AWS represent average number of characters per word and average words per sentence.

The FOG formula Equation 3 is also used to calculate grade level; however, the formula uses number of syllables to make a prediction. Text with a high percentage of polysyllable words is assigned a higher grade level under FOG.

$$FOG = 0.4 \bullet (AWS + PPW)$$

Equation : FOG for calculating U.S grade level. AWS and PPW represent average words per sentence and percentage of polysyllable words. Polysyllable words are defined as words with more than three syllables.

Both formulas show some similarities in how they are calculated. They each are a linear function on two components: sentence complexity and word complexity.

In this research, we train an LSTM-LM to predict the next word given a variable number of context words. The training dataset is comprised of several novels written by Tolstoy: War and Peace, Anna Karenina, What Men Live By (and Other Tales), and The Forged Coupon (and Other Stories). Using the writings of Tolstoy as the dataset for the model provides two benefits. Firstly, there is a large amount of text which can be used for training and testing our model. Secondly, we can establish an accurate baseline value Tolstoy's writing style for each RF on a validation dataset not used in training. We compare the RF scores of the LSTM-LM generated text at a range of context words with that of samples from Tolstoy’s writings. Additionally, we evaluate the perplexity of the LSTM-LM models on the validation dataset.

# Literature Review

In this section we review work related to our project. Specifically, we reviewed papers related to language processing using recurrent neural networks, word vector embeddings, and evaluation metrics.

LSTMs have been extensively used for LM. Barman and Boruah (2018)^3^ used an LSTM-LM to do phonetic transcription for the Assamese language. Similar to our approach, their training dataset was gathered from a novel. Their model was aimed at text completion while typing. Sak et al. (2014)^1^ introduced a new LSTM architecture which allowed them to preform speech recognition on a vocabulary size of 2.6 million. Their architecture is now implemented in PyTorch­^12^.

In almost all LMs, word vector embeddings are used to provide meaningful features to the model. This concept of using a continuous vector representation of a word was first introduced by Hinton et al. (1984)^8^. Methods to efficiently learn the vectors of a vocabulary of words in a text have been proposed by Mikolov et al (2013)^9^. They introduced the CBOW and Skip-gram algorithms that produce high quality embeddings. They demonstrated that the embeddings can produce meaningful results by adding or subtracting them.

Investigation of evaluation methods for Language Models has been done by Chen et al. (1998)^4^. In their work, they show that perplexity can accurately predict word-error rate for language models within domain data. They introduced a new evaluation metric "M-ref" which is able to outperform perplexity in predicting word-error rate on out of domain data. Yet another evaluation metric known as Angular Embedding Similarity was proposed by Moeed et al. (2020)^13^. This metric measures the distance between embedded vectors. It was used to measure the performance of Language Models in generating headline summaries of portions of text.

# Model

Training an LSTM-LM model requires a large amount of text data. The longest of Tolstoy's novels were selected for the dataset as well as two other collections of short stories. Each of the novels selected from Tolstoy's work was downloaded from an open-source repository of literature known as Project Gutenberg^10^. Sequences of text were extracted from the novels based on blank lines separating portions of the text. Each sequence is split into words and punctuation, all of which are individually referred to as tokens. These sequences sometimes contain chapter headings and short portions of dialogue. Sequences less than four tokens are removed in order to provide the model with useful data. A vocabulary of 22,357 unique tokens is extracted from the sequences. Tokens are converted to their corresponding index in the vocabulary, and the indexes each correspond to an embedding vector^8,9^. The total number of cleaned sequences is 19809. They are split into (85%, 10%, 5%): train, test, and validation sequences.

In order to make a prediction of the next word our LSTM-LM needs a number of context words (N) as input. Using seed text is a common practice to start the text generation, however that would introduce some bias into evaluation using RFs. The seed text itself could be removed, but it may have influence on the rest of the generated text. Instead, we pad the beginning of each sequence with N "start of sequence" (SOS) tokens and one "end of sequence" (EOS) token to the end. Finally, we split each sequence into a number of training examples equal to the length of the sequence. Each training example is collected by taking the N preceding words for the input, and the next word as the label, then sliding the N preceding words over one to repeat the process. The LSTM-LM is trained to minimize Cross Entropy Loss^11^ defined in Equation 4. This loss function helps the model learn to make the target next word in the sequence more likely under its probability distribution. Once the model is trained on this dataset it is able to generate new text with the only seed text required being the N SOS tokens. Another benefit of padding the sequences as described is that the total number of training examples in the dataset is the same regardless of N. The total number of training, testing, and validation examples are 996423, 112826, and 59188. The training set will be used to tune the parameters of the model, the testing set will inform the selection of hyperparameters, and the validation set will be used to evaluate RF scores and perplexity.

$$\text{loss}(x,\ class) = - x\left\lbrack \text{class} \right\rbrack + log(\sum_{j}^{}{exp(x\lbrack j\rbrack))}$$

Equation : Cross Entropy Loss given in the PyTorch documentation. Where $x$ represents the probability distribution produced by the LM and $\text{class}$ represents the target next word.

The procedure to generate text with our LSTM-LM is to give it an input sequence (for the first prediction, input will be N SOS tokens). The model will produce a categorical probability distribution over all words in the vocabulary from which the next word will be selected. Selecting the next word using the most likely word produces repetitive text. By selecting words with some stochasticity, the text becomes more interesting. Words will continue to be added to the sequence by the model until a final EOS token Is predicted or a maximum sequence length of 600 is reached.

Evaluations^\*^ for RFs will be done by generating sequences from our LSTM-LM models for each N. To ensure accurate evaluation on the RFs, sequences will be generated until the total number of words reaches 1500. The baseline score for Tolstoy's text will be established by evaluating the validation dataset with the RFs. Perplexity will be measured for each N by averaging the models' perplexity on each sequence in the validation dataset.

*\* Readability formulas were calculated using: https://readabilityformulas.com/*

# Results

We trained our LSTM-LM on $N = 2,\ 3,\ 4,\ 5,\ 6,\ \ and\ 7$ context words. All runs were done with the same set of hyperparameters shown in Table 1. Additionally, runs are compared after training for 5 epochs using a Nvidia GeForce RTX 2060.

<img src="media\image1.png" style="width:2.56in;height:1.92in" /><img src="media\image2.png" style="width:2.56in;height:1.92in" />

Figure 1: Loss and accuracy evaluated on the test dataset after each epoch on the train dataset. Cross Entropy Loss (left) and Accuracy (right). Each metric is evaluated at each number of context words.

Evaluation of the LSTM-LMs on the test dataset are shown in Figure 1 after each epoch. Final Cross Entropy Loss on the test dataset decreases with each additional context word used in the model up to N = 5. Surprisingly N = 6 and 7 show higher loss at the end of the training process. Similar results are shown for accuracy in Figure 1 with N = 2 through 5 showing an increase in accuracy and N = 6 and 7 preforming slightly worse. An explanation of why 6 and 7 context word models did not perform as well could be that additional training epochs may be necessary. Their respective loss curves have not plateaued compared to the others. Additionally, as with all deep neural networks, more data will always help.

<img src="media\image3.png" style="width:2.56in;height:1.92in" /><img src="media\image4.png" style="width:2.56in;height:1.92in" />

Figure 2: Comparing RF scores for each number of prediction words supplied to the LSTM-LM models. The baseline scores for Tolstoy are shown in orange. ARI (left) and GF (right).

After training the models, we evaluate them using the ARI and FOG tests shown in Figure. 2. As the number of prediction words increases, we can see that the scores on both tests approach the baseline score of Tolstoy's text from the validation dataset. In the case of ARI, this means that using more context words makes the models generate text which contain words with more characters and longer sentences. For the FOG RF, increasing number of prediction words results in generated text with more polysyllable words and longer sentences. Additionally, all of the tests in Figure B.1 provide support that an increased context words trend towards Tolstoy's scores for each test.

<img src="media\image5.png" style="width:2.56in;height:1.92in" />

Figure 3: Perplexity of the LSTM-LMs on the test dataset evaluated at each number of context words.

Average sequence perplexity of the LSTM-LMs on the validation dataset are shown in Figure. 3. For N = 2 to 5 there is a sharp drop in perplexity, indicating that increasing number of prediction words up to this N allows for better prediction. Similarly, to final loss and accuracy shown in Figure 1, 6 and 7 context word models preform worse than expected. However, N = 6 and 7 models still produce less perplexity than N = 2 and 3.

# Conclusions

In this paper we successfully trained Language Models for next word prediction on text written by Leo Tolstoy. The models were able to achieve around 20% prediction accuracy on the testing dataset. We compared the effect of giving the models different amounts of context for it to make a prediction. Our results show that increasing the number of context words is helpful up to a certain amount for the measures of accuracy and perplexity. After this amount, performance decreases slightly. One explanation for the poor results of 6 and 7 context words is underfitting due to insufficient data. Each model was trained with the same number of datapoints; therefore, it is logical to expect that at a certain sequence length more data will be required for training. With a larger amount of training data perhaps the trend of increased performance with more context will continue. Interestingly, the results of the readability formulas show no performance decrease with higher number of context words. Instead, the generated text from the models more closely matches that of Tolstoy's baseline in terms of sentence and word complexity. Directions for future work that could validate the results shown in this study could be to train the language models on different authors. While Tolstoy scores relatively high complexity on readability formulas, it would be interesting to see how this system behaves for an author with lower scores.

All code for this project is available at: <https://github.com/gladisor/TextGenerator>

# Bibliography

\[1\] Speech recognition: <https://arxiv.org/pdf/1402.1128.pdf>

\[2\] Sentiment analysis: <https://arxiv.org/pdf/1605.01478.pdf>

\[3\] Next word transcription: <https://www.sciencedirect.com/science/article/pii/S1877050918320556>

\[4\] Perplexity / word-error rate: <https://www.cs.cmu.edu/~roni/papers/eval-metrics-bntuw-9802.pdf>

\[5\] Readability formulas for Navy: <https://apps.dtic.mil/dtic/tr/fulltext/u2/a006655.pdf>

\[6\] Clinical trials: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5915809/>

\[7\] Clinical trials: <https://officeofresearch.ucsc.edu/compliance/services/irb22_consent_readability.html>

\[8\] Embeddings: <https://web.stanford.edu/~jlmcc/papers/PDP/Chapter3.pdf>

\[9\] CBOW / Skip-gram: <https://arxiv.org/pdf/1301.3781.pdf>

\[10\] Project Gutenberg: <https://www.gutenberg.org/>

\[11\] Cross Entropy Loss: <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss>

\[12\] LSTM implementation: <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM>

\[13\] Headline generation / evaluation metric: <https://www.aclweb.org/anthology/2020.lrec-1.222.pdf>

# 

# Appendix A

Sample generated sequences for each N.

N = 2:

to how he had an child , and that to ourselves to ease the rostóvs which that hesitation . he began reading the angel dresses and followed through their position , as with all my god , of his wife , while we ! and how little with princess drubetskáya . the hour he had brought some hussars . he had torn down through the very person . he was not due to the awkward garden . o god ! replied levin .

N = 3:

a declaration about prince andrew , who called me to draw as an appointed hélène , she told himself what he was talking , to give all her mind , says the lad do not sing moscow or go head on my hands , sunny , and so many detail that i should now , agreed for them . with their eyes turned downhill , and the coffee of which . day , my opinion my part put it . in some history always existed , the presence of propose going in to the emperor to the rare ... called one by run to know and there s no deal in which natásha was quite idle , death , and the live officer came to kiss himself . yes , no !

N = 4:

yes , yes , if you were about a less like its enemy s as he suddenly was at all sitting in the evening , feeling that it was necessary to see it , levin thought , i know that she will deprive her satisfaction there would be fresh secretary . they both stood up and the expression of his face , which was immediately wasted to prince vasíli silently they would remain speaking to him said more in hospital , l ma peuples ! nodded the one crowns glittering and in commiseration . from mytíshchi natásha s painting becomes drowned to her when she could scarcely comfort something as black as an agonizing accounts .

N = 5:

pierre was aware of what he had said doing what was already made such remorse . márya dmítrievna like questions and anxiety obviously or understood all the anxiety ?... i have a not yet to do something happened to go to this chair . when he had remained in the same year , the creeds he had seen with his mother , which was heard in the edge of the room . natásha went up to her . she mounted from her waist as soon as he listened to nesvítski s legs , communicated to him an trial in the season and followed his daughter and knew how strangely had they as a monk ! cried the officer . oblonsky , on the other hand , others are noticeable from bagratión again . the thoughts less than ever . napoleon went to the banquet and turned forward and sat down on his dressing hut about one of the left side of the wall . the polish army and this book began for rest , or she could not make thereby , terrible , anna rostóv said to her , the interpreter could not be talked of and sincerely who was gazing at the symmetry , a serf person and a taking it , then .

N = 6:

well , no ! he suddenly asked himself because he knew how that he showed in this man , of which they would have undertaken to see her as he by all this held food , which suddenly was sharpening them had love since we were dreadful with men and and triumph house , mending the herdsman the preobrazhénsk ; but they women vanguard . in favor of saying he ought to avoid all a score or evening and the matter as a unhappy one always orders when fresh , slave , beyond such thousand and extraordinary new faces . fond of affairs , catching in peasants and surroundings that was he yet to do that le uneasy exposed to honest anatole . the direct career , he scattered into that state . just as can kitty send it of us . one person and here has been attained in new way to his life , entreating his love , and straight up with him to natásha . but in spite of the whistling campfire , sitting and the countess , and of his uhlans sergey ivanovitch moved soon in the row , who had found the consciousness of it .

N = 7:

let us and no dear after us . you attack one , for them so then for a court dispositions with her easy before darya alexandrovna mounted his companions , sang only up , and carried him on the inner end of the blinds , a soldier , which before i can imagine of any thousand thousand thousand , prince vasíli was fixed in her , shut her ; while asked how he pleased the refreshment camp s room , faintly played smartly in the breast . only as he , the only effect inevitably most uppermost , had in the attacking governing men a third gardens , and on the church the old man was still too deeply so splendid in war , that if an officer is quickly . said sónya , waking ready to the same three minutes , whom he loved . in a week the cracked guards , in all about details . the whole aim that their assurance were to procure my hand , said dolgorúkov , standing in the hands of the subject , which saved his elder look agreeable from the last branches of which played as every of charge of a flush regiment and , shall i stay to the old service for our will . and you ? pierre wants not to see the official circles of me without all their former communal subjects enough . our pleasure before is quite specially carrying to visit the last society my neighbor without taking sure , catiche , caused the question .

# Appendix B

<img src="media\image6.png" style="width:2.57in;height:1.93in" /><img src="media\image7.png" style="width:2.56in;height:1.92in" /><img src="media\image8.png" style="width:2.56in;height:1.92in" /><img src="media\image9.png" style="width:2.56in;height:1.92in" />

Figure B.: RF scores for Flesch Reading Ease, Flesch-Kincaid, Linsear Write Formula, and SMOG.

# Appendix C

| Hyperparameter                   | Value |
|----------------------------------|-------|
| Embedding dim                    | 128   |
| Hidden dim                       | 512   |
| Number of LSTM layers            | 3     |
| Dropout                          | 0.2   |
| Number of fully connected layers | 2     |

Table 1: Hyperparameters of the Neural Network for each N of context words.
