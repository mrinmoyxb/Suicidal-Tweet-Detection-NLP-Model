# Suicidal Tweet Detection NLP Model
Deep Learning Text Classification Model based on Natural Language Processing (NLP) Algorithm.


## Objective:
More than 700,000 people die due to suicide every year. For every suicide there are many more people who attempt suicide. A prior suicide attempt is an important risk factor for suicide in the general population. Suicide is the fourth leading cause of death among 15–29-year-olds. 

Suicide is a serious public health problem that can have lasting harmful effects on individuals, families, and communities. There are many factors that contribute to suicide. The goal of suicide prevention is to reduce factors that increase risk and increase factors that promote resilience.

Suicide is linked to mental disorders, particularly depression and alcohol use disorders, and the strongest risk factor for suicide is a previous suicide attempt.

10th September is celebrated as World Suicide Prevention Day

**The main objective of this NLP model is to identify real time tweets that has potential of expressing suicidal sentiments, in order to prevent suicide in our society.**



## About Dataset:
This is a binary text classification datatset of two classes: 
**Potential Suicide Post(1)** and **Not Suicide Post(0)**

This dataset provides a collection of tweets along with an annotation indicating whether each tweet is related to suicide or not. The primary objective of this dataset is to facilitate the development and evaluation of machine learning models for the classification of tweets as either expressing suicidal sentiments or not.

The dataset contains two columns:

1. **Tweet:** This column contains the text content of the tweets obtained from various sources. The tweets cover a wide range of topics, emotions, and expressions.

2. **Suicide:** This column provides annotations indicating the classification of the tweets. The possible values are:
* **Not Suicide post:** This label is assigned to tweets that do not express any suicidal sentiments or intentions.
* **Potential Suicide post:** This label is assigned to tweets that exhibit indications of suicidal thoughts, feelings, or intentions.

**Total Tweets:** 1787

**Total Not Suicide post:** 1127

**Total Potential Suicide post:** 660

Potential Applications:

**Suicidal Ideation Detection:** The dataset can be used to train models to automatically detect and flag tweets containing potential suicidal content, enabling platforms to take appropriate actions.

**Mental Health Support:** Insights from this dataset can be used to develop tools that offer mental health resources or interventions to users who express signs of distress.

**Sentiment Analysis Research:** Researchers can analyze the linguistic patterns and sentiment of both non-suicidal and potentially suicidal tweets to gain insights into the language used by individuals in different emotional states.

**Public Health Awareness:** The dataset can be used to raise awareness about mental health issues and the importance of responsible social media usage.


## Algorithm:
NLP stands for Natural Language Processing. It is a field of computer science that deals with the interaction between computers and human (natural) languages. NLP research has been highly successful in developing techniques for understanding and generating text, translating languages, and extracting information from text.

Some key concepts related to NLP model:

**1. Tokenization:** Tokenization is a fundamental step in NLP. It is a necessary step for many NLP tasks. It is the process of breaking a text into smaller units called tokens. These tokens can be words, characters, or subwords. Tokenization is a necessary step in many NLP tasks, such as part-of-speech tagging, named entity recognition, and sentiment analysis.

There are two main types of tokenization: **word tokenization** and **sentence tokenization**

* **Word tokenization** is the process of breaking a text into words. Sentence tokenization is the process of breaking a text into sentences. Word tokenization is usually done by using a regular expression to match words. For example, the regular expression \w+ matches any sequence of one or more alphanumeric characters.

* **Sentence tokenization** is usually done by looking for punctuation marks that indicate the end of a sentence, such as periods, exclamation points, and question marks.

**2. Padding:** Padding in NLP is the process of adding extra zeros to the end of shorter sequences so that they have the same length as the longest sequence. This is done to ensure that all sequences have the same shape, which is required by many NLP models.

**3. Embedding:** In natural language processing (NLP), an embedding is a representation of a word or phrase as a vector of real numbers. The vector space is typically of much lower dimension than the number of words in the vocabulary, allowing words with similar meanings to be represented by vectors that are close together.

Some of the other tasks in NLP include:
* **Named entity recognition:** This is the task of identifying named entities in a text, such as people, organizations, locations, etc.
* **Part-of-speech tagging:** This is the task of assigning a part-of-speech tag to each word in a sentence, such as noun, verb, adjective, etc.
* **Semantic parsing:** This is the task of converting a natural language sentence into a formal language that can be processed by a computer, such as a logical form or a tree structure.
* **Machine translation:** This is the task of translating text from one language to another.
## Language and library:

Language: Python 3.11.4

Library: TensorFlow and Matplotlib
