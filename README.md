# InflDerivMorph

This repository contains the code to my Master's dissertation:**Differences between inflectional and derivational morphology: a study on the predictability in a distributional vector space**.

## Abstract
This thesis studies the differences between derivational and inflectional morphology based on their predictability in a distributional vector space. The focus lies on German, and three additional languages with different types of morphological systems and varying levels of morphological richness are analysed: French, Turkish and Russian.
The common criteria in linguistics which distinguish inflectional and derivational morphology imply that inflections tend to co-occur with other words in a more systematic manner. In other words, the \textit{systematicity of distribution} is expected to be lower for derivations. 
I analyse the extent to which derivations and inflections occur systematically in language by means of machine learning experiments The task is to predict the derived form (e.g. the agent noun \textit{singer}) or the inflected form (e.g.  third person singular \textit{sings}) from the base form (\textit{to sing}). I tune various feed-forward neural networks with different architectural features, such as number of layers and the application of non-linear functions. I use linear regression models to analyse to what extent linguistic as well as methodological features explain variance in performance. The results show that inflections are overall better predictable, and, therefore, more systematically distributed. However, it turns out that some derivations are actually well predictable, while some inflections are rather badly predictable. Thus, the results suggest a gradual view on the differences between inflectional and derivational morphology.

## Code
The code is written in  python and uses pytorch as the main library to implement the models. 

The most important files are:
- model.py which contains the model of the work (RelationFeedForward)
- rank\_evaluation.py which contains 
- baseline.py which contains the baseline 
- main\_relations.py which contains the main method to train and evaluate the model


If you have any questions, do not hesitate to contact me: eva.huber@student.uni-tuebingen.de 


