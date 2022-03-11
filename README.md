# Multimodal-Sentiment-Analysis
<h2>Background:</h2>  
With the rapid development of the Web, multimodal data consisting of text and images increases rapidly in recent years, so multimodal sentiment analysis on these two modalities is drawing more and more attention from researchers. Most of the previous works concentrate on the fusion mechanism of the textual and visual modality in the text-image pair level. But documents with multiple images are also becoming more and more popular in daily life, such as news, blogs and reviews. In these cases, there is a document with several images, different from the image-text pair, images are unaligned with text, and visual content is sparse compared with textual content for a document. So how to effectively leverage both textual and visual content is a key problem for document-level multimodal sentiment analysis. This paper proposed a novel neural network model using images to enhance the saliency of sentiment expressed by imagerelated sentences in a document through the gate and attention mechanism, instead of fusing the textual and visual features for the final sentiment classification. Because generally each image can represent an entity or aspect, for example, a food review may includes several aspects like Drink, Hamburger and noodle, so we call it visual aspect in this paper. Experiments on two public multimodal datasets demonstrate the effectiveness of the proposed model.
<h2>Experiment:</h2>
1. The classification accuracy of different models on the Yelp dataset. Here, TF denotes Textual Features, VF denotes Visual Features, HS denotes Hierarchical Structure, VA denotes Visual Aspect, and SA denotes Sentence Attention. (Yelp Data)


<img src="https://github.com/lln1997/Multimodal-Sentiment-Analysis/blob/d975191241b5269947405337dbcce73ed5d8f46a/images/yelp.png" alt="drawing" width="500" height="300"/>
2. Experimental results of different models on Multi-ZOL.
<img src="https://github.com/lln1997/Multimodal-Sentiment-Analysis/blob/d975191241b5269947405337dbcce73ed5d8f46a/images/multi-zol.png" alt="drawing" width="500" height="250"/>
3. Ablation analysis of the proposed model. Here, HS denotes Hierarchical Structure, WA denotes Word Attention, and SAL denotes Sentence Later-Attention.
<img src="https://github.com/lln1997/Multimodal-Sentiment-Analysis/blob/c6e612115e24d3c04931ac5fd2335bb678f43463/images/ablation.png" alt="drawing" width="500" height="250"/>
4. Modality interaction of VAGU.
<img src="https://github.com/lln1997/Multimodal-Sentiment-Analysis/blob/8f1d4a46a97aebf5f4b6eea6eef45e890d01f4a9/images/interaction.png" alt="drawing" width="500" height="200"/>
5. Gate function of VAGU.
<img src="https://github.com/lln1997/Multimodal-Sentiment-Analysis/blob/8f1d4a46a97aebf5f4b6eea6eef45e890d01f4a9/images/gatefunction.png" alt="drawing" width="500" height="200"/>
