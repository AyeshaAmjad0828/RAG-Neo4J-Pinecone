# Comparative Analysis of RAG
## NEO4J vs PINECONE
> Refer to pre-requisite section for the environment set up prior to starting this experiment.
> 1. Clone this repository. 
> 2. Add relevant API keys and configurations in [.env](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/.env) file. 

### Technology Stack

| Neo4j | Pinecone | Langchain | TogetherAI | OpenAI | Hugging Face |
| ----- | -------- | --------- | ---------- | ------ | ------------ |
|   **![](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/README%20assets/Pasted%20image%2020240602005215.png)**    |   **![](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/README%20assets/Pasted%20image%2020240602005222.png)**    |   **![](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/README%20assets/Pasted%20image%2020240602005230.png)**    |   **![](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/README%20assets/Pasted%20image%2020240602005237.png)**    |   **![](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/README%20assets/Pasted%20image%2020240602005156.png)**    |   **![](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/README%20assets/Pasted%20image%2020240602005149.png)**    |
### Overview
**Purpose:**
Conducting a comprehensive analysis of RAG (Retriever-Augmented Generation) pipeline using Neo4J and Pinecone to test: 
1. Performance of graph vs vector databases
2. Effects of splitting methods
3. Retrieval methods and strategies
4. Quality of responses through various rag metrics
5. Reranking and multi-query on failed responses

**Significance**: 
1. Insights on capabilities and limitations of a RAG pipeline
2. Understanding the available technology stack
3. Exploring non-traditional uses of databases
4. Implementation of practical projects
5. Suitability for specific use-cases

**Experiment Design:**

**![](https://lh7-us.googleusercontent.com/u8VnvF-Xs3Cxg-_Va3Rqv82wkuO_upSmWJt9kWBDTw98BpeRPfxWwlbLXKL11-aLA1FflKdGb0u3lHU57oo0MsSBE1REDAUvCSC9trx4DCTLbel_2FQI4QObObpCGPIYAQ8yvgWv3owy)**

**DATA**

In this project all the experiments are done on the [Constitution of Pakistan](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/input/Constitution.pdf) pdf file.
To evaluate the RAG, questions with ground truth can be found here: [questions.json](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/input/questions.json)
### Running PINECONE Rag

Select your desired experiment by defining the following options in [RAG_with_Pinecone.py](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/RAG_with_Pinecone.py)
```
retrieval_method = 'cosine' #What you defined at the time of pinecone creation

chunker = 'recursive' ##recursive, semantic, sentence, character, paragraph

embeddingtype = 'openai'  #openai, HF, langchain, spacy, empty string will invoke gpt4all

llmtype = 'llama2' #llama2, llama3, Qwen, empty string will invoke Mixtral

embedding_dimension = 1536  ##change to 384=gpt4all embedding,

index_name = pinecone_index
```

This script will load the input file embeddings into pinecone index, generate responses to the questions and write it to a json file in [output](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/tree/main/output) folder. 

### Running SimpleNEO4J Rag
> You may run into errors while running this on your local system. 
> Suggestion: Run this on google colab

**Setup**: Traditional method where the exact data indexed is the data retrieved. Uses two approaches:
- *Similarity search*: only focuses on retrieving matches with top similarity score (cosine or mmr)
- *Hybrid search*: search takes into account prominent keywords in addition to the similarity coefficient

Select your desired experiment by defining the following options in [SimpleNeo4J_RAG.ipynb](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/RAG-NEO4J/For%20Colab/SimpleNeo4J_RAG.ipynb)
```
retrieval_method = 'cosine' #euclidean, mmr, cosine  (mmr was running into an error)

chunker = 'recursive' #recursive, semantic, sentence, character, paragraph

embeddingtype = 'langchain'  #openai, HF, langchain, spacy, empty string will invoke gpt4all

llmtype = 'gpt4' #llama2, llama3, Qwen, empty string will invoke Mixtral

embedding_dimension = 3072  ##change to 384=gpt4all embedding

index_name = "vector"  # default index name
```

This script will load the input file embeddings into neo4j instance, generate responses to the questions and write it to the json file in [output](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/tree/main/output) folder. 

### Running AdvancedNEO4J Rag
> Advanced NEO4J RAG strategies have been inspired by: [Implementing-Advanced-Retrieval-RAG-Strategies-With-Neo4j](https://github.com/pks20iitk/Implementing-Advanced-Retrieval-RAG-Strategies-With-Neo4j)
> You may run into errors while running this on your local system. 
> Suggestion: Run this on google colab

**Setup**: Four advanced strategies of retrieval were implemented to balance precision embeddings and context retention:
- *Parent retriever:* Instead of indexing entire documents, data is divided into smaller chunks called Parent and Child documents. Child documents are indexed to better represent specific concepts, while parent documents are retrieved to ensure context retention.
- *Hypothetical Questions*: Documents are processed to determine potential questions they might answer. These questions are then indexed for better representation of specific concepts, while parent documents are retrieved to ensure context retention.
- *Summaries:* Instead of indexing the entire document, a summary of the document is created and indexed. Similarly, the parent document is retrieved in a RAG application.

Select your desired experiment by defining the following options in [AdvancedNeo4J_RAG_with_strategies.ipynb](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/RAG-NEO4J/For%20Colab/AdvancedNeo4J_RAG_with_strategies.ipynb)
```
retrieval_method = 'cosine' #take it from LoadingDatatoNeo4j

chunker = 'semantic' #take it from LoadingDatatoNeo4j  #recursive, semantic, sentence, character, paragraph

embeddingtype = 'langchain'  #openai, HF, langchain, spacy, empty string will invoke gpt4all

llmtype = 'gpt4' #llama2, llama3, Qwen, empty string will invoke Mixtral

embedding_dimension = 1536  ##change to 384=gpt4all embedding,
```


### RAG Evaluation 
**Description**: Ragas provides several metrics to evaluate various aspects of your RAG systems.
**![](https://lh7-us.googleusercontent.com/Itt_HYtJA8vOsJx9rbbstPWEmkbE6sw9CjuK6U2zSFKR5056SaPFGyQ4rju-owBwnF3v2vZZHIUoHMrmHiF1rF4FIu3OKw0UswbH1zTKxc42dqtiA2y8hTXA5gQh0OoJwyP27YUabkrL)**
We are using six metrics to evaluate the RAG pipeline results:

1. [Faithfulness](https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html)
2. [Answer relevancy](https://docs.ragas.io/en/latest/concepts/metrics/answer_relevance.html)
3. [Context precision](https://docs.ragas.io/en/latest/concepts/metrics/context_precision.html)
4. [Context recall](https://docs.ragas.io/en/latest/concepts/metrics/context_recall.html)
5. [Answer correctness](https://docs.ragas.io/en/latest/concepts/metrics/answer_correctness.html)
6. [Answer similarity](https://docs.ragas.io/en/latest/concepts/metrics/semantic_similarity.html)

Run [Evaluation_Metrics.py](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/Evaluation_Metrics.py) to generate evaluation score for each question. Here is how the output of evaluation looks like: [scores.xlsx](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/output/scores/qa_results_hybridneo4j_langchain_cosine_recursive_gpt4_scores.xlsx)

### Pre-requisite Set up
1. Set up accounts on [Neo4j](https://neo4j.com/), [Pinecone](https://app.pinecone.io/), [OpenAI](https://openai.com/), [Hugging Face,](https://huggingface.co/) and [TogetherAI](https://www.together.ai/).
2. Add the API keys and configuration in [.env](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/.env) file.

TogetherAI is used to facilate the inference from several LLMs on local machine.

You can create upto 5 indexes in Pinecone free version. Create a new index for each dimension. 
**![](https://github.com/AyeshaAmjad0828/RAG-Neo4J-Pinecone/blob/main/README%20assets/Pasted%20image%2020240602020334.png)**
