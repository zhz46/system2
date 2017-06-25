
# 2017.06.22 note for content_based_v1


## 1. Objective

Create a content-based recommender module. 

## 2. Summary

An unsupervised learning version of recommender module is built based on TF-IDF representation of SKU. The similarity between SKUs is calculated using self-defined weighted average of mixed features (Quantitative variables and categorical variables) with range [0, 1]. Furniture data is used. There are 193408 observations.

## 3. Method

### 3.1 Feature Extraction

title, brand, products and parentProducts, price_hint are extracted from json file to represent SKU.

### 3.2 Feature mapping and processing

Python's nltk module is used. Each title is tokenized and stemmed. Some stop words are removed. A vocabulary is generated. Each title is mapped into a 42328 dimension TF-IDF vector. SVD is used to reduct dimension (in NLP context, this is called Latent Semantic Analysis). Research has demonstrated that around 300 dimensions will usually provide the best results with moderate-sized document collections (hundreds of thousands of documents). 200 dimensions is used since titles are pretty short.

**Some notes:**
+ Current text cleanup method is simple. Will try Eliot's text cleanup function. SVD dimension would be chosen based on both variance scree plot and running time.
+ Faster python module, such as gensim, may be used to speed up later.

###  3.3 Distance Metric and Similarity Score

Distance is calculated based on weighted average of four groups feature: title, brand, products and parentProducts, price_hint with weights [0.2, 0.2, 0.5, 0.1] assigned using common sense and knowledge. The output similarity score is defined as 1 - distance.

#### products (P) and parentProducts (PP)

The distance between two SKU x and y is calculated based on string comparison:

+ 0.0 --- two SKUs have equal products value ($P_{x} == P_{y}$)
+ 0.1 --- one's product is same as another's parentProduct ($P_{x} == PP_{y} | P_{y} == PP_{x}$)
+ 0.3 --- one's parentProduct is part of another's product or one's product is part of another's product 
($P_{x} \simeq PP_{y} | P_{y} \simeq PP_{x} | P_{x} \simeq P_{y}$)
+ 0.8 --- Either one has missing products value
+ 1.0 --- Otherwise

#### brand (B)

The distance between two SKU x and y is calculated based on string comparison:

+ 0.0 --- two SKUs have equal brand value ($B_{x} == B_{y}$)
+ 1.0 --- Otherwise

#### title (TF_IDF)

The distance between two SKU x and y is calculated based on consine distance

#### price_hint (PH)

The distance between two SKU x and y is calculated based on:
$$ \frac{|PH_{x} - PH_{y}|}{PH_{x} + PH_{y}} $$


## 4. Output

For each SKU, 30 (or any k) most similar SKUs are listed. There are 2 output json files. 

+ data.json provides basic data summary with features ["id", "note_id", "brand", "image_url", "title]
+ content_rs.json provides pairswise similarity score of recommendations


