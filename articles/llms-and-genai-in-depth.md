# LLMs and GenAI: In Depth
This article explores large language model architectures, pre-training strategies, and applications in generative AI, with a focus on healthcare use cases.
Gen AI and LLMs: In Depth


1.0 Introduction
1.1 What is AI?
In simple terms, Artificial Intelligence (AI) is the ability of a computer system or a machine to do tasks that involve human intelligence, such as reasoning, decision making, or solving problems. Use of computational models is done to implement such intelligence, where these models study data, gain knowledge from it, and use the acquired knowledge to complete the tasks.
 
Figure 1: AI Timeline (Malik, 2023)
As shown in Figure 1(Eddy Malik,2023), AI has transformed gradually with different breakthroughs occurring at different years and talking specifically about the current popular tool,  Generative Pre-Trained Transformer(GPT) by OpenAI which is a multimodal large language model, their latest version is GPT-4 used by all of us for some or the other task.



1.2 Generative AI and How it works?
A form of artificial intelligence known as "generative AI," or "GenAI," is capable of producing original writing, graphics, music, and even code. GenAI is intended to produce results that are similar to the data it was trained on, in contrast to standard AI, which usually analyses data or makes predictions.

This is how it works: 
1.	Learning from Data: GenAI models are trained on vast datasets, including thousands of images, articles, or sounds, in order to learn from them. The model gains knowledge of the structures, styles, and patterns in this data during training. 
2.	 Producing New Content: After training, GenAI may generate unique content that resembles but differs from what it has learnt. Based on its knowledge, it can, for instance, construct a tale, produce a realistic image, or recommend new music. 
3.	 Applications: ChatGPT for writing and discussion, DALL-E for creating images, and tools for design, gaming, and music are all powered by GenAI. 
To put it briefly, GenAI can serve as a creative assistant by applying its knowledge to create novel, entertaining, or aesthetically striking products.




1.3 Traditional AI  VS  Gen AI
Traditional AI	Generative AI
1.It is completely based on specific rules and
Algorithms.	1. It has self-learning models that understand patterns and correlations in data without the need of preset rules.
2. These systems are highly specialized as they are explicitly programmed with rules and logic by human experts.	2. These systems can handle a wider range of jobs as they learn to identify patterns and give outputs from large volumes of data.
3. They are comparatively transparent and interpretable, as are founded on clear rules and logic.	3. It can be hard to understand what happens inside as these models rely on intricate patterns discovered in data to make their decisions, making them opaque and hard to interpret.
	
Figure 2: Difference between Traditional AI and Gen AI (visionx, 2024)

2.0 How GenAI has evolved with different architectures?
GenAI has evolved with time where different pioneers came up with efficient architectures to generate data using different models. We would now dive deep into the different architectures that were involved in the process of evolution of GenAI and try to understand how they work.

2.1 Recurrent Neural Network(RNN) Architecture

A type of neural network called the “Recurrent Neural Network (RNN)”  is made to process sequential data, such as speech, text, or time-series data, by processing input one step at a time. Unlike standard neural networks, RNNs feature "memory" characteristics that allow them to remember information from earlier steps in the sequence, making them ideal for jobs that require context across time. 
RNNs process input sequentially, repeating through each time step and generating outputs based on both the current input and the prior hidden state. The network's recurrent connections enable it to "remember" information from earlier in the sequence, which is the cause of this memory effect.

Important RNN components include:
1.	 Hidden State: An RNN's hidden state, which is updated at each time step based on the new input and the previous hidden state, helps the network retain context across time steps. 
2.	 Backpropagation Through Time (BPTT): RNNs are trained using a back propagation technique that takes into account the sequential nature of the data, known as back propagation through time. However, this training method frequently results in issues with vanishing and exploding gradients, which makes it difficult for RNNs to learn long-range dependencies efficiently. 
3.	 Variants like LSTM and GRU: To overcome the shortcomings of basic RNNs, more sophisticated types such as LSTMs (Long Short-Term Memory networks) and GRUs (Gated Recurrent Units) were developed to overcome the shortcomings of basic RNNs.

It’s not like generative algorithms are invented now, previous models made use of RNNs. They did not have the required compute power and memory to perform very well at generative tasks. If we take the example of a next word prediction task, the model will have a look at the previous word and try to make a prediction for the sentence. As we scale the RNN implementation to be able to see more preceding words in the text, we also have to scale the resources that the model uses. And, after all of this the model still fails as it has not seen enough of the input words. So, in order to make a good prediction the model needs to have a good understanding of the whole sentence, to be more specific the model needs to have a good “contextual” understanding of all words in relation to each other and then it can make a good prediction. 



2.2	LLM Architecture
In simple words, Large language models (LLMs) are a type of machine-learning neural network that are trained on immense data available, which helps them understand the human language and generate a response related to it. The data that we are talking about can be of any form be it image, audio or text, and the model will generate a reply or information based upon our input. In other programming paradigms, we write computer code to interact with the machines but LLMs on the other hand can understand the human language like another human being.
LLMs can be said to fit within an intersection of Natural Language Processing (NLP) and Deep Learning. Understanding, processing and generating human language makes LLMs a part of NLP. They solve complex problems from answering questions to translating languages. And also, LLMs can understand the context of a sentence by simply looking at it once using a “Transformers” architecture ( which we will talk about in detail in the later part of this article), this is a “Deep Learning” architecture, thus making them a part of the  deep learning domain as well. LLMs play an important role in Gen AI as they don’t just understand the human language but generate it as well.






2.3 RNN Architecture VS LLM Architecture
RNN	LLM
•	Process data sequentially, meaning each time step depends on previous steps output and hidden state. 	•	Process data in parallel rather than sequential.
•	Struggle retaining long term dependencies due to vanishing gradients, even with variants like LSTM and GRU.	•	Have self-attention layers that help in understanding the long term dependencies.
•	Usually work slow on large datasets and are not scalable.	•	Transformers, used in LLMs can be scaled up to billions of parameters and can handle extensive data. Thus, making them ideal for large-language tasks.
•	Historically employed for time-series prediction, speech recognition, and language modelling. However, transformers are increasingly more frequently utilised for NLP tasks.
	•	Because of their exceptional         performance, scalability, and versatility, LLMs are primarily utilised for a variety of NLP tasks, including text generation, translation, summarisation, and more.


Figure 3: Difference between RNN and LLM Architecture.

2.4 Use Cases of LLMs
LLMs give us context-aware responses to the inputs we give them in human language, and they boost conversational AI in chat bots and virtual assistants (such as Google's Gemini and OpenAI’s GPT) to improve interactions between human agents and themselves.
These are some important use cases of LLMs:
•	Audio Data Analysis- If we are providing the model with an audio data then it can summarize the important points to be noted for us.
•	Natural Language to Machine Code- The model can output a code in our preferred general purpose programming language if we give it an input to write a particular code through the natural language.
•	Named Entity Recognition- For example, if we input the model with a news article and ask it to output the key people and places mentioned in the articles.
•	Sentiment Analysis- The models can be trained to understand textual and voice sentiments, which can help with customer feedback and reputation management.
•	Language Translation- Language barriers can be broken with the help of LLMs where they can do real time language translation and help with better communication.
•	AI Assistants- Chat bots can come in handy if we want a self-serve customer care solution to answer customer queries.
•	Content Generation- LLMs can help writers, bloggers and marketers by suggesting edits for their write ups and also generating initial drafts, which will eventually accelerate the content creation process.
One area of development is where we can augment LLMs connecting them to external data sources or using them to invoke external APIs. Here, we can provide the model with real-time data that it has not been pre-trained for and it can be powered to interact with the real world.



3.0 Neural Network Architectures for LLM

There are different Deep Learning architectures that GenAI models use, typically Neural Networks. Also, there are different types of networks used:
•	Transformers: These are used to process data sequences like: words, image pixels, etc. and discover connections between them.
•	Variational Autoencoders(VAEs): Such models learn to encode the input data into a compressed lower dimensional space and then decode it back to remake the original data. This helps in the generation of new, same sample data and is most commonly used for generating images 
•	Generative Adversarial Networks(GANs): Usually, have two neural networks called “Generator” and “Discriminator”. The “Generator” generates data (for e.g., images) that will mimic a target distribution. The “Discriminator” compares if the dataset is real or fake. This enhances the ability of both the networks of generating realistic data and distinguishing of fake and real respectively. The most common use of this architecture is for image generation.   
 


3.1 Transformer Architecture 

In 2017, a research paper called “Attention is all you need ” was published by Google and some students of the University of Toronto. This paper had what they called as a “Transformer Architecture”. With the help of this architecture the model could be scaled efficiently with multi-core GPUs, the input data could be processed in parallel which would help the model process very large datasets, and most importantly it would help the model understand the contextual meaning of an input sentence and words.

Now, we might think why is it necessary for the model to understand the context of each word in a sentence, this is because languages are complicated and sometimes even humans find it difficult to understand the relevance of a sentence. Let’s try to understand this with an example sentence:
“The coach trained the boy with a bat.”
In the above sentence we can see there can be two understandings:
1.	The coach might have trained the boy with the help of a bat.
2.	Or the coach might have trained the boy that had the possession of the bat.
If such sentences are confusing for us humans then they would definitely be confusing for the model. Hence, it is important to understand the relevance of each word with every other word in the sentence. And, this can be done by applying “attention weights” to the relationships which will increase the ability of the model to understand some information like: “who has the bat?”, “who could have the bat?”  or if it is even relevant to understand the possession of the bat. These attention weights are learned during LLM training and higher the weight , higher is the relevance of words with each other.
 



Figure 4: Attention Map.

In Figure 4, we can see that the word “Bat” is strongly connected to the words “Boy” and “Coach”, this is called “Self-Attention”. The ability of the model to learn attention in this way helps the model to encode the natural language more accurately.

Below, is the detailed diagram of the “Transformer Architecture” from the research paper ,“Vaswani, Ashish, et al. "Attention Is All You Need." Advances in Neural Information Processing Systems, vol. 30, Curran Associates, Inc., 2017”.
 
Figure 5: Transformer- model architecture. (Ashish Vaswani, 2017)


Let’s simplify this diagram to have a better understanding of the architecture.

 

Figure 6: Simplified Diagram of Transformers. (DeepLearning.AI)

In Figure 6, we can see that the transformers architecture is divided in two main components called the “Encoder” and “Decoder”. The first step of the transformers is the input, the model only understands numbers and not the normal words of the natural language as it is a giant statistical calculator. Thus each word of the input is assigned to a token id using a tokenizer.





523	847	1294	523


the	coach
trained	the





In figure 7, each word is assigned with a unique token id. These token are now passed to the “Embedding” layer, this layer is a trainable vector  embedding space. Here, each token occupies a unique position within the space and is matched to a multi-dimensional vector. These vectors learn to encode the meaning and context of the vectors in the input sequence. There is an extra positional embedding that helps maintain the word position as the embedding after tokenization happens in parallel. At last, the sum of the vectors of the positional embedding and token embedding are passed to the multi-headed self-attention.

When the resultant vector is passed to the multi-headed self-attention, the model analyses the relationship between the tokens and the input sequence to capture the contextual dependencies. There are multiple sets of self-attention weights that are learnt in parallel and are independent of each other. The different heads learn different aspects of the language, for example: Head1 might learn relation between people entities , Head2 might learn rhyming words and so on.

Now that the attention weights are applied to the input data, this output is passed to fully connected “Feed Forward Network”. The output of this network is the vector of logits that is proportional to the probability score of each token in the tokenizer dictionary. These logits are passed to the final “Softmax Layer” where, they are normalized into a probability score of each word. Now, the last output has probability for all words in the vocabulary and one of the token will be just higher than the rest of the scores and will be the exact output.

We can use this transformer architecture with the two components for “Sequence to Sequence Tasks”, for example: Language Translation. With further modifications we use the architecture for “Generative Tasks”.

The most interesting part about this architecture is that we can use individual components or variants of it for different tasks, below are some examples for the same:

1.	Encoder only Transformers:
These models are also used for “Sequence to Sequence Tasks”, and if we add additional layers to these models we can use the models for “Classification Tasks” like: Sentiment Analysis. A popular example of encoder only models is “BERT”.




 
Figure 8: Encoder Only Model

2.	Decoder only Transformers:
These are the most commonly used models in todays time and are used for “Generative Tasks”. Examples of such models are: GPT, Bloom, LlaMa , Jurassic and many more.

 
Figure 9: Decoder Only Model

3.	Sparse Transformers: Sparse transformers are helpful for tasks like document-level processing and extended context analysis because they analyse lengthy sequences using mechanisms like local attention, dilated attention, and random/global attention. By focussing on certain input segments rather than the complete sequence, they are made to manage lengthy sequences more effectively while lowering the computing effort.  Some examples of such transformers are: BigBird, Reformer, and Longformer .


4.	Mixture of Experts (MOE) Transformers: MoE models reduce computing demands while enabling effective scaling to handle big datasets by dynamically routing input to specialised "experts" within the model. MoE layers are composed of several expert models, and for each input, only a subset of these experts are activated. Resource use is optimised by this selective routing. Some examples are: Switch Transformer and GLaM (Google's Language Model).

5.	Mutlimodal Transformers: By working with various data kinds (such as text and photos), they allow models to comprehend and produce text and visual information at the same time. To handle both kinds of inputs and simulate their interactions, multimodal transformers frequently have dual streams or cross-attention layers. Here are a few examples: Contrastive Language–Image Pretraining, or CLIP, DALL-E.

3.2 Variational Autoencoders
There is a simple problem statement in Machine Learning that all of us come across which is when the dimensionality of the data is too high and we need to compress and reduce the dimensionality of data into a smaller space, one solution can be VAE.  In 2013, a research paper was published “Auto-Encoding Variational Bayes (Diederik P. Kingma, 2013)”, where the authors introduced the concept of combining probabilistic graphical models with neural networks, creating a generative model capable of learning latent representations. Before, we get into VAEs lets understand how an Autoencoder works. The input for an AE is a text , image or a vector with high dimensionality which is run through a neural network and compressed into a smaller representation. Then , it is reconstructed according to the original data from the compressed representation using a neural network.

 
Fig.10  Autoencoders Architecture (Birla, 2019)
The Autoencoder architecture has four important components:
1.	Encoder: This can be a fully connected or a convolutional layer, it compresses the input data into a smaller-dimensional latent space representation. Usually the compressed data is not like the original data and is distorted.
2.	Bottleneck Layer: The compressed representation (latent space) in the middle forces the model to capture the most important features of the input data.
3.	Decoder: This can again be a fully connected or a convolutional layer, and reconstructs the latent space representation back to the original dimension. The decoded data is an approximate estimation of the original data.
4.	Reconstruction Loss: This loss will make sure that the autoencoder learns to compress and reconstruct the input data as correctly as possible. The choice of the loss function will be according to the type of data :- Mean Squared Error(MSE: Most commonly used for continuous data, Binary Cross Entropy(BCE: Mostly used for binary or normalised data that have values between 0 and 1).


There are some common applications of Autoencoders:
1.	Denoising Autoencoders: Removes noise from images or signals.
2.	Dimensionality Reduction: Reduces the feature space for tasks like visualization.
3.	Generative Models: Act as the basis for advanced models like Variational Autoencoders (VAEs).
Now, that we have an idea about what Autoencoders are let us try to understand what is different with Variational Autoencoders. VAEs are a type of generative model and autoencoder that learns a probabilistic distribution over latent variables to generate new data similar to the input data. Unlike traditional autoencoders, which encode data to a fixed representation, VAEs encode data to a distribution.
 
Fig.11  Variational Autoencoder Architecture( (Shende, 2023))
Variational Autoencoders (VAEs) differ from traditional autoencoders in their approach to representing input data in the latent space. While a conventional autoencoder maps input data to a specific point in the latent space, VAEs encode the input as a probability distribution over the latent space. The decoder then samples from this distribution to reconstruct or generate new data. This probabilistic framework enables VAEs to learn a structured and continuous representation of the latent space, which proves beneficial for tasks such as generative modeling and data synthesis.
Transitioning from a traditional autoencoder to a VAE requires two significant adjustments. First, the encoder's output must represent a probability distribution rather than a single point. To achieve this, the encoder generates parameters of the distribution, such as the mean and variance, typically assuming a multivariate Gaussian distribution. However, other distributions, such as Bernoulli, can also be used depending on the application.

The second key change involves modifying the loss function by introducing the Kullback-Leibler (KL) divergence term. This term quantifies the difference between the learned latent space distribution and a predefined prior distribution, which is often a standard normal distribution. By minimizing this divergence, the model ensures that the latent space remains well-structured and aligns with the prior distribution, aiding in regularization and promoting meaningful latent space organization.

 
The optimization objective for VAEs is known as the Evidence Lower Bound (ELBO). The VAE loss function comprises two components: the reconstruction loss and the KL divergence loss. The reconstruction loss evaluates how closely the decoder's output matches the original input, similar to the loss function in traditional autoencoders. The KL divergence loss penalizes deviations between the latent distribution learned by the model and the prior distribution, contributing to a well-regularized latent space.

Compared to traditional autoencoders, VAEs offer several advantages. They excel at generative tasks, enabling the creation of new data points from the latent space distribution. The continuous nature of the latent space allows for smooth interpolation between points, facilitating the generation of novel data samples. Additionally, the probabilistic encoding mechanism makes VAEs less prone to overfitting, as it encourages the model to capture more generalized representations of the data.
Despite these benefits, VAEs pose some challenges. Training them can be computationally demanding, and achieving stable convergence often requires careful tuning of the model architecture and hyperparameters. Furthermore, interpreting the latent space representation may be complex, and the quality of the generated data is heavily influenced by the underlying model design and the training dataset.

3.3 Generative Adversarial Networks
In 2014, there was a research paper published ”Generative Adversarial Nets (Ian J. Goodfellow, 2014) ” when GANs were first devised. In such a model there are two neural networks that work simultaneously, one is called “Generator” and the other is called “Discriminator”. Given, a distribution of inputs (X) and labels (Y) the “Discriminative” network models the conditional distribution P(Y|X) and the “Generative” network model the joint distribution P(X,Y). The training for such models is done by a pair of “Adversaries”[two players with conflicting loss functions]
 
Fig.12  GANs (Silva, 2018)
In Figure 12, we can see that we provide random noise to the “Generator” and it tries to produce data (i.e. an image in this case) from some probability distribution, basically to fool the discriminator making it unable to identify the fake image. Also, we can see that the “Discriminator” gets an input from the training set and the generated data from the generator simultaneously which allows the discriminator to act like a judge and decide whether the input comes from the generator or a training set. Usually this game follows with:
1.	The Generator trying to maximize the probability of making the discriminator mistakes its inputs are real.
2.	And the discriminator guiding the generator to produce more realistic images.
GANs are used in a variety of applications due to their ability to generate the realistic data:
1.	Data Augmentation: GANs generate high-quality synthetic data, that can be used to train Machine Learning models when real life data is scarce or is expensive to collect.
2.	Image and Video Applications: GANs are used for image generation (creating realistic images of non-existent objects or people), image-to-image translation (e.g., colorizing black-and-white images or day-to-night transformations), super-resolution (enhancing image quality and details), and deepfake creation (producing convincing synthetic media like face swaps).
3.	Gaming and VR: In gaming and virtual reality, GANs are used for environment generation (creating lifelike virtual worlds) and character creation (designing realistic virtual characters).
4.	Speech and Text: GANs are used in speech and text applications for text-to-speech (TTS) (creating realistic synthetic voices) and speech synthesis (enhancing the naturalness of audio in virtual assistants or dubbing).

3.4 Difference between all the architectures

Feature	Transformers	VAEs (Variational Autoencoders)	GANs (Generative Adversarial Networks)
Purpose	Sequence modeling, contextual learning, and generative tasks.	Latent space learning and probabilistic data generation.	Data generation through adversarial training.
Core Mechanism	Self-attention mechanisms to capture global dependencies.	Encoder-decoder architecture with latent variable sampling.	Two networks: Generator and Discriminator competing against each other.
Input Type	Sequential data (e.g., text, audio, time-series) and non-sequential (e.g., images with positional encoding).	Structured or unstructured data (e.g., images, text).	Unstructured data (e.g., images, audio, video).
Output Type	Sequences,  embeddings,  or generated text/images.	Reconstructions or samples from the learned latent distribution.	Realistic data samples resembling the training set.
Training Objective	Minimize prediction loss (e.g., cross-entropy for classification or next-token prediction).	Minimize reconstruction loss and KL divergence.	Minimax game: Generator minimizes discriminator success, while discriminator maximizes accuracy.
Probabilistic Nature	Not explicitly probabilistic but can model uncertainty via attention mechanisms.	Explicitly probabilistic using latent variable sampling (e.g., Gaussian distributions).	Implicitly probabilistic, generating data to fool the discriminator.
Scalability	Highly scalable with parallel computation (e.g., GPUs, TPUs).	Scalable but computationally heavy due to latent sampling.	Computationally intensive due to adversarial training and stability challenges.
Applications	- NLP (e.g., BERT, GPT) 
- Vision (e.g., ViT) 
- Time-series analysis 
- Audio synthesis	- Image reconstruction and generation 
- Anomaly detection 
- Data imputation	- Image synthesis (e.g., deepfakes, art generation) 
- Video generation 
- Super-resolution
Strengths	- Captures long-range dependencies. 
- Effective in large-scale language and vision tasks. 
- Parallelizable.	- Structured latent space representation. 
- Effective for generative modeling with interpretability. 
- Handles uncertainty well.	- Generates sharp and realistic samples. 
- Adversarial nature enables creative data generation.
Weaknesses	- Computationally expensive for long sequences (attention is O(n2)O(n^2)O(n2)). 
- Requires large datasets.	- May produce blurry outputs for high-dimensional data. 
- Computationally expensive latent space sampling.	- Difficult to train (e.g., mode collapse, vanishing gradients). 
- Sensitive to hyperparameter tuning.
Key Innovations	- Attention mechanisms 
- Positional encodings 
- Transfer learning (pretrained models)	- Reparameterization trick 
- Probabilistic modeling in neural networks	- Adversarial training 
- Use of generator-discriminator interplay
Real-World Examples	- GPT-4, BERT, DALL-E, Vision Transformers	- Variational Autoencoders for face reconstruction or anomaly detection	- StyleGAN, CycleGAN, DeepFake networks
Learning Process	Supervised or unsupervised (e.g., masked token prediction).	Unsupervised, optimizing reconstruction and KL divergence loss.	Unsupervised, based on the adversarial minimax game.
Latent Space	Implicitly learned via attention.	Explicit, structured latent space.	Implicitly learned via generator-discriminator dynamics.
Best For	- Text generation 
- Contextual understanding 
- Multi-modal tasks	- Learning interpretable latent representations 
- Anomaly detection	- Generating realistic high-quality samples.
			

Fig.13 (Transformers) Vs (VAEs) Vs (GANs).

4.0 How to create an LLM from scratch?
After having a good understanding of all the concepts above we are ready to answer a very important question which is “How do we create an LLM?”. I’ll try to simplify and explain the answer to this question with an example so that we can have a better idea of creating an LLM.  We will try to map the steps out to comply it with the “LLM Project Life Cycle”. Let us take an example where we will create a model that can query the user prompts for an uploaded document. 
1.	Define the use case:
It is crucial to set clear objectives before starting the development:-
•	What kinds of files can we upload? (For instance, CSVs, text files, and PDFs) 
•	Which types of inquiries will be accepted? (For instance, summaries, context-specific, and fact-based)
A precise definition will help us use the right tools and tech stack for implementation.
2.	Pipeline setup for Data Processing:
We need to create a machine-readable format from the uploaded document. There are some key actions to follow:
•	Upload and Parse Documents: 
Documents should be allowed form a user interface or API. The uploaded documents should be parsed into text using libraries like: PyPDF/pdfplumber(For PDFs), python-docx(For Word Documents), Tesseract (For scanned images).
•	Preprocessing text:
The text should be now prepared for querying by removing unnecessary characters, normalizing the text by converting into lowercase and removing punctuations,  and tokenize the text into words, sentences or paragraphs for easier indexing.
3.	Selection of a backend LLM:
Selecting an LLM is very important as it will be the core for our system. We have a few choices for that as well:
•	Pretrained LLM APIs: Models like  OpenAI’s GPT, Cohere, or Anthropic’s Claude for faster integration.
•	Open Source Models: Some models like  Hugging Face's transformers (e.g., GPT-2/3, BERT) or Llama for complete customization.
In case, if we choose an Open Source Model we need to fine tune the model according to our use case so that it is relevant. Hugging Face’s Trainer API tool can be used to streamline fine-tuning. Also, for the contextual knowledge, we need to do a similarity search based on vectors. Text embeddings can be indexed by libraries such as Weaviate of FAISS(Facebook AI Similarity Search). For indexing the document we need to divide the text into digestible sections(512 tokens, for example).Use an LLM(Such as GPT or BERT) to transform text segments into embeddings. To facilitate fast lookup, store embeddings in a vector database.
4.	Implement Query Understanding:
Convert the query into tokens in natural language. Analyze intent using  the LLM, then compare it to the content of the documents. Create custom query parsers for structured inquiries, such as sql like requests.
5.	Retrieval-Augmented Generation(RAG):
We can combine the LLM’s generative abilities with document retrieval:
•	Cosine similarity or other metrics can be used to search the indexed document for most relevant chunks.
•	Now, we pass the retrieved chunks with the user query to the LLM for generating the response.
This method helps the model to generate responses related to the uploaded document and it would not go off the track.
6.	Model Evaluation:
This is a step where we need to test the system on sample documents and queries to ensure accuracy of the model:
•	Measure the Precision: Check how often the system retrieves relevant document sections.
•	Evaluate recall: Check how systematically it identifies relevant content.
•	Asses the quality of the response: Compare the generated output with your desired output.
To enhance the performance, iteratively update the model, retrieval process and user feedback mechanisms. This step directly maps to the evaluation phase in the project life cycle.
7.	Optimization and Deployment:
To enhance the performance we can use caching for repetitive queries. Optimization of embeddings and indexing algorithms can be done for speed. Use cloud platforms, containerization, and orchestration tools directly to addresses deployment concerns. Try to build a user-friendly front-end which is one of the important parts of the deployment phase, ensuring the system is accessible to end-users.
In summary, these are some basic steps that can be followed to create an LLM where the model will generate responses according to the user query for a particularly uploaded document. We can create LLMs with different types of outputs like audio, images , video, synthetic data and etc.  depending on what use case we want to work with.
Generative AI Project Life Cycle (In General)

 

 



Bibliography
Ashish Vaswani, N. N. (2017, June 12). Attention is all you need. Retrieved from arxiv.org: https://arxiv.org/abs/1706.03762
Birla, D. (2019, March 12). Retrieved from medium.com: https://medium.com/@birla.deepak26/autoencoders-76bb49ae6a8f
DeepLearning.AI. (n.d.). Generative AI with Large Language Models. Retrieved from coursera.org: https://www.coursera.org/learn/generative-ai-with-llms/lecture/R0xbD/generating-text-with-transformers
Diederik P. Kingma, M. W. (2013, December 20). arxiv.org/abs. Retrieved January 20, 2025, from arxiv.org: https://arxiv.org/pdf/1312.6114
Ian J. Goodfellow, J. P.-A.-F. (2014, June 10). arxiv.org/labs. Retrieved 01 20, 2025, from arxiv.org: https://arxiv.org/pdf/1406.2661
Malik, E. (2023, November 9). Artificial Intelligence(AI) and ChatGPT Timelines. Retrieved from officetimeline.com: https://www.officetimeline.com/blog/artificial-intelligence-ai-and-chatgpt-history-and-timelines
Shende, R. (2023, April 19). Retrieved from medium.com: https://medium.com/@rushikesh.shende/autoencoders-variational-autoencoders-vae-and-%CE%B2-vae-ceba9998773d
Silva, T. (2018, January 7). Retrieved from freecodecamp.org: https://www.freecodecamp.org/news/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394/
visionx. (2024, April 16). Traditional AI Vs. Generative AI: What’s the Difference? Retrieved from visionx.io: https://visionx.io/blog/traditional-ai-vs-generative-ai/

