# Tiny Scale Is All I Can Spare To Play With Transformer.
<p align="center">
Srijan Srivastava<br>
India<br>
Srivastavavsrijan321@gmail.com<br>
QCoreNest@gmail.com
</p>

v2, July 2026


## Abstract
Introduction of the Transformer neural network architecture in the famous `Attention Is All You Need` paper has created a huge wave of AI development in recent years. The scaled dot-product attention allows for information to be processed with higher efficiency and quality, which the previous RNN-based models lacked. However Transformer-based models comes with their own set of challenges, particularly with parameter efficiency for tiny scale models. At such tiny scale a Transformer model essentially uses more parameter than it really should. This regime is very under-explored and for good reasons however exploring it might allow us to discover interesting insights about the Transformer. So here-in this paper I am introducing Silia, a novel neural network architecture designed for efficient modelling & classification tasks under severe parameter budget. Training against Andrej Karpathy's nanoGPT, Silia achieves comparable loss and generation quality with significantly less parameters. Along with when a 117M parameters model even after being under-trained on a ~100M tokens of synthetic dataset, Silia achieves losses similar to what scaling loss predict for a Transformer based 117M parameters model trained on ~100M tokens.


## 1. Introduction
The dominant trend in Transformer-based language models has been scaling: larger models, more data, more compute in pre-training and RL for CoT based reasoning capabilities to consistently yield better performance. This means that the smallest practical models such as Qwen, Gemma & GPT-OSS lie anywhere between 1B-20B parameters. Despite them being called "small", these models are still billions in parameters and are trained on trillions of tokens. This trajectory, as useful as it is, has widened the gap between general-purpose frontier research experiments and task-specific small research experiments.

This is where I introduce my neural network architecture which merges the _Attention_ layer with the _SwiGLU Feed Forward_ layer from the Transformer to save lots of parameters while preserving much of the original performance. This new architecture is what I call __Silia__ or __Silu in Attention__. **Silia** aims to reduce the number of parameters per block, especially at much smaller scale (100 million parameters or less) while achieving competitive performance and quality as a standard Transformer.


## 2. Model Architecture
<img src="img/arch.png" alt="youforgeta1000thingseverydaymakesurethisisoneofthem" style="width:100%;">

Merging Attention and SwiGLU was inspired from 2 core ideas.

Attention is dynamic and smart about which information to mix, but it has no strong non-linearity to actually transform that information. SwiGLU has the strong non-linearity but it cannot transform input in a way that Attention does.

In Google's PaLM technical report _PaLM: Scaling Language Modeling with Pathways_

instead of doing

$$y = x + MLP(RMSNorm(x + Attention(RMSNorm(x)))$$

the authors did

$$y = x + MLP(RMSNorm(x)) + Attention(RMSNorm(x))$$

and noted 15% faster training speed with small to zero degradation in quality.

### 2.1. Transformer Block
**Attention** is the heart of Transformer and it needs no introduction.

$$Q = XW_Q, K=XW_K, V=XW_V$$

$$A = \mathrm{softmax} \left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

$$\text{Y} = \text{A}W_O \tag{1}$$

Where,

$W_Q \in R^{c \times (d \times h)}$, $W_K \in R^{c \times (d \times h)}$, $W_V \in R^{c \times (d \times h)}$, $W_O \in R^{(d \times h) \times c}$, $M$ is a causal attention mask and $d_k$ is dimension of key matrix.

The above equation is what was introduced in the now famous _Attention Is All You Need_ paper. This is the equation which is used in modern auto-regressive language models like GPT, Claude, Gemini, DeepSeek, Kimi and more.

**SwiGLU** was introduced and used by Google in their (N Shazeer, 2020) paper _GLU Variants Improve Transformer_. Prior to this, the standard Transformer architecture heavily relied on simpler activations like ReLU or GeLU. The paper demonstrated that replacing standard feed-forward network layers with Gated Linear Units (GLUs), specifically those utilizing the _Swish_ activation function (SwiGLU) significantly improved training convergence and downstream model accuracy. Since then SwiGLU has been a popular choice for researchers to use in their Transformer models.

Now here we pass $Y$ from equation $(1)$ into 2 weight matrices,

$$u = YW_u, v = YW_v$$

$$\text{SiLU}(u) = u \odot \sigma(u) = \frac{u}{1 + e^{-u}}$$

$$\text{uv} = \text{SiLU}(u) \odot v$$

$$Y = \text{uv}W_o$$

where,

$W_u \in R^{c \times (4 \times c)}$, $W_v \in R^{c \times (4 \times c)}$, $W_o \in R^{(4 \times c) \times c}$, $\odot$ is the element-wise multiplication operation and $\sigma$ is the $\text{sigmoid}$ activation function.

### 2.2. Hydra Latent Attention (HLA)
Hydra Latent Attention is an attention mechanism which incorporates 5 novel ideas:
- _Exclusive Self Attention (XSA)_ and _Attention Free Transformer (AFT)_ introduced by Apple.
- _Gated Attention_ and _Hydra Head_ introduced by Qwen Team.
- _Multi-Head Latent Attention_ introduced by DeepSeek-AI.

Let $X$ be our hidden state.

$$T = \text{RMSNorm}(X)$$

$$\text{Q}_↓ = TW_{Q↓}, \text{K}_↓ = TW_{K↓}, \text{V}_↓ = TW_{V↓}$$

$$\text{Q}_↑ = QW_{Q↑}, \text{K}_↑ = KW_{K↑}, \text{V}_↑ = VW_{V↑}$$

Where,

$W_{Q↓} \in R^{d_i \times d}$, $W_{K↓} \in R^{d_i \times d}$, $W_{V↓} \in R^{d_i \times d}$, $W_{Q↑} \in R^{d \times (d \times h)}$, $W_{K↑} \in R^{d \times (d \times h)}$ and $W_{V↑} \in R^{d \times (d \times h)}$

The $\text{K}_↓$, $\text{V}_↓$ down projections will be stored in the $\text{KV}$ cache to save memory as proposed by DeepSeek-AI in their paper _DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model_. While $\text{Q}_↓$ down projection doesn't contribute in the $\text{KV}$ cache, it does help in reducing the number of parameters due to low-rank factorization.

$$G=TW_G$$

Where $W_G \in R^{d_i \times (d \times \frac{h}{2})}$

The $G$ term was introduced by Qwen Team in their paper _Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free_ where they noted performing an element-wise multiplication of the $\text{sigmoid}$ gate with attention results in improved performance of the model along with mitigating the _attention-sink_ problem.

Now this is where things get a little interesting. In another paper by the Qwen Team titled _HydraHead: From Head-Level Functional Heterogeneity to Specialized Attention Hybridization_ they introduced the concept of head-wise Full-Attention (FA) Linear-Attention (LA) hybrid instead of using a layer-wise hybrid which is more commonly used. In HydraHead the authors selected a some heads from $Q$, $K$ and $V$ tensors and allocated to FA which the rest was allocated to FA. This resulted in major reduction in $\text{KV}$ cache along with attention computation.

In the paper the authors used _Gated DeltaNet (GDN)_ as their Linear Attention function however for _Hydra Latent Attention_ I choose to use Apple's _Attention Free Transformer_. There is no concrete and scientific reason behind it. It is purely a design decision mainly due to AFT's quick and simple implementation. We will refer to Attention Free Transformer as $\text{AFT}$ in our mathematical formulation.

Assuming $\text{Q}_↑$, $\text{K}_↑$ and $\text{V}_↑$ has $n$ number of heads where $n$ is a positive integer divisible by 2. are placed in such a way:

$$[H_1, H_2, H_3, H_4, H_5, H_6, H_7, H_8, ..., H_{n-3}, H_{n-2} H_{n-1}, H_n]$$

We break the head dimension in half with an interleaving pattern and assign first half of it to _Exclusive Latent Attention (XLA)_ which is Multi-Head Latent Attention + Gated Attention + Exclusive Self Attention, and the second half to _Apple's Attention Free Transformer ($AFT$)_.

$$\text{QKV-XLA} = [H_1, H_3, H_5, H_7, ..., H_{n-3}, H_{n-1}]$$

$$\text{QKV-AFT} = [H_2, H_4, H_6, H_8, ..., H_{n-2}, H_n]$$

Now we are ready to perform attention.

$$Q_{XLA} = \text{RoPE}(Q_{XLA}), K_{XLA} = \text{RoPE}(K_{XLA}), V_{XLA} = \text{RoPE}(V_{XLA})$$

$$Q_{XLA} = \text{RMSNorm}(Q_{XLA}), K_{XLA} = \text{RMSNorm}(K_{XLA})$$

$$Q_{AFT} = \text{RMSNorm}(Q_{AFT}), K_{AFT} = \text{RMSNorm}(K_{AFT})$$

$$A = \mathrm{softmax} \left(\frac{{Q_{XLA}}{K_{XLA}}^\top}{\sqrt{d_{k_{XLA}}}} + M\right){V_{XLA}}$$

$$O = \text{A} \odot \sigma(G)$$

$$Y_1 = \text{XSA}(O)$$

$$Y_2 = \text{AFT}(Q_{AFT}, K_{AFT}, V_{AFT})$$

Now concatenate both $Y_1$ and $Y_2$ head-wise in the same interleaving manner

$$C = \text{Concat}(Y_1, Y_2)$$

$$Y = \text{RMSNorm}(C)$$

This concatenates our heads from $[H_1, H_3, H_5, H_7, ..., H_{n-3}, H_{n-1}]$, $[H_2, H_4, H_6, H_8, ..., H_{n-2}, H_n]$ back to $[H_1, H_2, H_3, H_4, H_5, H_6, H_7, H_8, ..., H_{n-3}, H_{n-2} H_{n-1}, H_n]$ then applies $\text{RMSNorm}$ to normalize.

### 2.3. Silu in Attention (Silia)
We will use our new __Hydra Latent Attention__ mechanism which reduces the $\text{KV}$ cache along with memory & compute cost. This change was made compared to the original version of Silia which used the standard Multi-Head Attention because an open reviewer pointed out that due to MHA, Silia's memory and compute cost would increase by 2.5x despite the parameter savings, because in Silia we use the attention layer twice in a single block compared to a standard Transformer block where the attention layer was used only once per block.

First we'll calculate attention over our hidden state $X$ where $X \in R^{b \times t \times d}$

$$O = \text{HLA}(X)$$

$$u = OW_u, v = OW_v \tag{1}$$

Where $W_u \in R^{(d \times h) \times d_o}$, $W_v \in R^{(d \times h) \times d_o}$

Now on the 2 outputs $U$ and $V$, we will apply the $\text{SiLU}$ activation function.

$$H = \text{SiLU}(u) \odot v \tag{2}$$

Where $H \in R^{b \times t \times d_o}$

Now in equation $(2)$ we have a non-linear transformation of our hidden state $X$ processed with $HLA$. Now we will pass equation $(2)$ into $HLA$.

$$Y = X + \mathrm{HLA}(H)W_O$$

Where $W_O \in R^{(d \times h) \times d}$ and $Y \in R^{b \times t \times d}$

After passing equation $(2)$ into $HLA$ we take a dot-product of it with an output projection matrix $W_O$ and add our original hidden state $X$ for create a residual connection to ensure rich gradients in deep neural networks similar to Transformer.

## 3. Why Silia
Why do I think replacing linear layers in SwiGLU Feed-forward Network with Attention is a good idea?

Attention as we know is mostly a linear transformation over our hidden state but it isn't simple, regular transformation like Feed-Forward network. We can think of attention as "smart" linear transformation. Such a linear transformation which tells us relevancy of every token, especially at longer sequence lengths. However the attention mechanism lacks a "strong" non-linearity. Attention does use the _softmax_ activation function which is a non-linear activation function but _softmax_ only decides which token attend to which other tokens. This makes _softmax_ a not so "strong" activation function.

SwiGLU feed-forward network however does have a strong activation function which is the _silu_ activation, in-fact at small scales (less parameters and smaller context windows) feed-forward networks such as SwiGLU can approximate exactly what attention does with high accuracy, and this does make sense after all feed-forward networks are _universal function approximators_. However as the model parameters and the context length scales feed-forward networks get worse at approximating the attention mechanism which results in worse performance compared to Transformer.

This is what **Silia** is about. Introducing a new class of feed-forward networks which use attention mechanism for transforming our input and hidden states linearly and using activation functions like _silu_ for transforming that information non-linearly. Instead of running both separately and wasting parameters on overlapping functionality, Silia replaces the static linear matrices in SwiGLU with attention getting dynamic mixing and strong non-linearity in one unified operation.


## 4. Parameter Analysis
### 4.1. Parameters Per Layer In Transformers
In a traditional Transformer the Attention layer has $W_Q$, $W_K$, $W_V$ and $W_O$ matrices, and SwiGLU has $W_u$, $W_v$ and $W_o$ matrices.

Where,

$W_Q \in R^{c \times (d \times h)}$, $W_K \in R^{c \times (d \times h)}$, $W_V \in R^{c \times (d \times h)}$, $W_O \in R^{(d \times h) \times c}$, $W_u \in R^{c \times (4 \times c)}$, $W_v \in R^{c \times (4 \times c)}$, $W_o \in R^{(4 \times c) \times c}$ and $d = \frac{c}{h}$.

Adding all Attention layer shapes,

$$c*(3 \cdot h \cdot d) + (h \cdot d) \cdot c = 4 \cdot c^2 \tag{1}$$

Now add all SwiGLU layer shapes,

$$c \cdot (2 \cdot 4 \cdot c) + (4 \cdot c) \cdot c = 3 \cdot 4 \cdot c^2 \tag{2}$$

Now add equation $(1)$ and $(2)$ together,

$$4 \cdot c^2 + 3 \cdot 4 \cdot c^2 = (4 \cdot c)^2 = 16 \cdot c^2$$

So we have a total of $16 \cdot c^2$ parameters per layer in a traditional Transformer.

### 4.2. Parameters Per Layer In Silia
Unlike Transformer in Silia we merge both Attention and SwiGLU FFN together as discussed above. In Silia we have $W_{Q↓}$, $W_{K↓}$, $W_{V↓}$, $W_{Q↑}$, $W_{K↑}$, $W_{V↑}$, $W_G$, $W_u$, $W_v$ and $W_O$.

Where,

$W_{Q↓} \in R^{d_i \times d}$, $W_{K↓} \in R^{d_i \times d}$ and $W_{V↓} \in R^{d_i \times d}$, $W_{Q↑} \in R^{d \times (d \times h)}$, $W_{K↑} \in R^{d \times (d \times h)}$, $W_{V↑} \in R^{d \times (d \times h)}$, $W_G \in R^{d_i \times (d \times \frac{h}{2})}$, $W_u \in R^{(d \times h) \times d_o}$, $W_v \in R^{(d \times h) \times d_o}$ and $W_O \in R^{(d \times h) \times d}$

Adding all shapes we get,

$$[\frac{h}{2}d^2 + d^2 + 8 \cdot h \cdot d^2] + [(2 \cdot h \cdot d^2 + 4 \cdot d^2 + h \cdot d^2)] + 6 \cdot h \cdot d^2 = [5 + \frac{35}{2} \cdot h] \cdot d^2$$

For simplicity let's switch $\frac{35}{2}$ with $\frac{36}{2}=18$,

$$[5 + \frac{36}{2} \cdot h] \cdot d^2 = [5 + 18 \cdot h] \cdot d^2$$

So we have a total of $[5 + 18 \cdot h] \cdot d^2$ parameters per layer in Silia.

### 4.3. Comparing Parameters Per Layer In Silia & Transformer
Even though it's clear from the above mathematics that the number of parameters per layer in Silia is less than that of the Transformer, we'll still use inequality to prove the same mathematically.

$$[5 + 18 \cdot h] \cdot d^2 < 16 \cdot c^2$$

$$\because d = \frac{c}{h} \implies c = d \cdot h$$

$$[5 + 18 \cdot h] \cdot d^2 < 16 \cdot (d \cdot h)^2$$

$$[5 + 18 \cdot h] \cdot d^2 < 16 \cdot d^2 \cdot h^2$$

$$5 + 18 \cdot h < 16 \cdot h^2$$

This can be rearranged into a quadratic equation,

$$16 \cdot h^2 - 18 \cdot h - 5 > 0$$

Upon solving the quadratic equation we get range for $h$,

$$h < −0.2396$$

$$h > 1.3021​$$

Since $h$ is always a positive integer we can discard $-0.2396$ and round $1.3021​$ to $2$ resulting in a simple logic that states: as long as the number of attention heads ($h$) is greater than or equals to $2$ ($h \geq 2$), the number of parameters per layer in Silia will be less than the same of the Transformer.


## 5. Training
### 5.1. Training Data
I trained on a Fineweb-edu dataset consisting of about ~350M characters. All the characters were encoded using byte-pair encoding, which has a shared source-target vocabulary of 512 token. The dataset after tokenization contained ~275M tokens. The train split had ~220M tokens and val split had ~55M tokens.

### 5.2. Hardware
I trained the base model on Google Colab Tesla T4 GPU using the hyperparameters described throughout the paper.

### 5.3. Optimizer
The AdamW optimizer with ${\beta}_1 = 0.9$, ${\beta}_2 = 0.95$ and $\epsilon = 10^{−8}$ was used to train the embedding layers, while the Muon optimizer with $\mu = 95$ was used to train rest of the non-embedding layers.

The learning rate over the course of training, according to the formula:

$$
\eta(t)=
\begin{cases}
\eta_{\max}, & \text{if learning-rate decay is disabled},\\[6pt]
\displaystyle
\eta_{\max}\frac{t+1}{T_w+1},
& 0\le t<T_w,\\[10pt]
\eta_{\max},
& T_w\le t\le T_c,\\[8pt]
\displaystyle
\eta_{\min}
+\frac{\eta_{\max}-\eta_{\min}}{2}
\left[
1+\cos\!\left(
\pi\frac{t-T_c}{T_d-T_c}
\right)
\right],
& T_c<t\le T_d,\\[12pt]
\eta_{\min},
& t>T_d,
\end{cases}
$$

where,

$$
\begin{aligned}
t &:\ \text{training iteration},\\
\eta_{\max} &:\ \text{initial learning rate},\\
\eta_{\min} &:\ \text{minimum learning rate},\\
T_w &:\ \text{number of warmup iterations},\\
T_d &:\ \text{total learning-rate decay iterations},\\
c &:\ \text{cooldown fraction},\\
T_c &= (1-c)T_d \quad \text{(start of cosine cooldown).}
\end{aligned}
$$

This corresponds to increasing the learning rate linearly for the first $T_w$ training steps, and decreasing it thereafter using the cosine decay rule as formulated above.

### 5.3. Configuration
20k steps is ~5 epochs on the train split which makes up 1B tokens in total. A higher peak learning rate was chosen since smaller models can oftentimes tolerate them. Training on lower and more standard peak learning rate such as $7e-4$ resulted in sub-optimal training in same number of training steps.

| $t$    | $\eta_{max}$ | $\eta_{min}$ | $T_w$ | $T_d$  | $c$  | $T_c$  |
| ------ | ------------ | ------------ | ----- | ------ | ---- | ------ |
| 20,000 | 3e-3         | 3e-4         | 1000  | 20,000 | 0.45 | 11,000 |
<div style="text-align: center; margin-top: 0.2em;">
<i>Table 1.</i> Training configurations for my Silia model.
</div>


## 6. Results
**Silia** was trained on Google Colab Tesla T4 GPU for ~1.5 hours of wall-clock time with the given training configurations in _Table 1_ and model hyperparameters in _Table 2_.

| Model             | Parameters | Vocab Size | Context Length | Layers | $d_{ff}$ | $d_{model}$ | $d_{head}$ | $h_Q$, $h_{KV}$ |
| ----------------- | ---------- | ---------- | -------------- | ------ | -------- | ----------- | ---------- | --------------- |
| **Silia (mine)**  | 524,672    | 512        | 1024           | 3      | 256      | 64          | 64         | 2, 2            |
| **Quark-v2**      | 465,504    | 500        | 256            | 4      | 192      | 96          | 24         | 4, 4            |
| **Spark-v4**      | 4,980,736  | 4096       | 512            | 6      | 512      | 256         | 32         | 8, 8            |
| **Supra-Mini-v6** | 1,410,688  | 4096       | 1024           | 6      | 256      | 128         | 32         | 4, 2            |
<div style="text-align: center; margin-top: 0.2em;">
<i>Table 2.</i> Hyperparameters and number of parameters of Silia (mine), Quark-v2 and Spark-v4 by LH-Tech-AI, and Supra-Mini-v6 by SupraLabs. All the models used RoPE for positional embedding and Silu as the hidden activation function.
</div>

### 6.1. Benchmark Progression
In _Table 3_ you can find the detailed comparison of Silia with all models listed in _Table 2_ on the HellaSwag, PIQA and LAMBADA benchmarks along with the final validation loss of the models.

| Benchmark             | Silia (mine) | Quark-v2 | Spark-v4   | Supra-Mini-v6 |
| --------------------- | ------------ | -------- | ---------- | ------------- |
| HellaSwag (acc)       | **0.2804**   | 0.2615   | 0.2695     | 0.2674        |
| PIQA (acc)            | _0.5419_     | 0.5283   | **0.5593** | 0.5403        |
| LAMBADA (ppl)         | _1704_       | 3500     | **588**    | 2089          |
| Final validation loss | **2.393**    | 2.556    | 3.108      | 3.79          |
<div style="text-align: center; margin-top: 0.2em;">
<i>Table 3.</i> Silia is the best performing model for it's parameter size in every aspect.
</div>

Silia 0.5M performs the best in all benchmarks compared to Quark-v2 0.5M while it does lose to Spark-v4 5M in PIQA and LAMBADA benchmarks it remains in 2nd best model in both of these benchmarks.

### 6.2. Generation Outputs
#### 6.2.1. Silia 0.5M (mine)
_- Belief off then, few infrarested directed in Kanetware, deference. Compost, Vocanter, and David Harbor Kanet. An over 100-minded nature. And the amount of a book of curriculum, produced director offered by the part of the two books and how in the past form. The smallest, the experiment was made to morality and papers, finds the history of ruling. Since the vote of power months and history are used to preserve the <|end-text|>he just accepting the writing of a joint ages, and she says._
_In 1874, however, the person’s joint was told of the joint hospitals, or jointed the joint, and two-conformed to a series of different locations. Note that referred to as joint the group of scientists at the New York’s Being and the Sexual Sundays, and the French revolutionary possible. That is the dense, we will be what it has taken home to do with local countries over a local local papers and gives the period of papers that contribute to a smaller rock in the joint, a different scientists, and the agencies could be a rockeform that its actual agency do not easily translate._
_In this case, the Ecclesiah, the joint was a different transmission of the scientists and the holocument. Better hospitals have the local papers of the Australian Professional, jointly belongs to strength, and whether the Hospitals should not be present in the joint writing. The papers but only required to leave the Space County for jointly denominations._
_The Records Ana Woln, a n<|end-text|>ng remained into a bathway of a bathway in the accuracy. It is then as the legal accuracy of pencils against its traditional state of accuracy of the Ministry of San Department of Management, Taxa was about someone possible._
_So, what blows all does not convey the signals of the bathways and herbivore the traditional material - it’s just accuracy. They were supposed to be conducted to provide others into accuracy, fathers and others. It had remained from either with the entire tree of the father. It was able to the write._
_So, then, quarters brings the people to go out with them, to ground, and created what herb_

#### 6.2.2. Quark-v2 0.5M
_Artificial intelligence is very possible. In the early 19th century, it has been done in the brain and acids, where they are taking some of the most common reality. This can also have to be lower than any other studies that would not be able to use this factor. If you’ve seen the same part of the world’s little glaucoma, we should need to be able to maintain their important_

#### 6.2.3. Spark-v4 5M
_is that it gives some unlimited means to think about the universe. It helps us not only to think about how the universe is created but also how we think about the universe. In this way, an inner universe can be made to our own universe. This is because it is not a matter of fact and that is the object of the universe. It can take a lot of time to understand how it is created and why it must be made. In the first place, the Universe is a complex and interesting part of it. It can be a kind of a real, creative, and universal part of our universe. It can be just that the universe was created. It could be a kind of universe. It can be a kind of kind of complex concept. That could be something that does something that really needs to be a kind of universe, or something that_


## 7. Conclusion
### 7.1. Use Cases
1. It can be used as light-weight, attention-powered, on-device task-specific models for televisions, smart refrigerators, smart car screens, smart watches, old mobile phones and computers.
2. It can be used as on-device models to immediately generate one-linear captions/titles for social media posts, dialogues for non-playable characters in video games.
3. It can be also be used for simple & fast image/text/topic classification, sentiment/emotion analysis, intent/toxicity detection.

### 7.2. Limitations
The embedding dimension and head dimension are tied together meaning that scaling up the architecture by width becomes increasingly challenging compared to traditional transformer since to scale up the width you'd need to scale up either the number of heads or the embedding/head dimension.

A more practical way to scale up the architecture would be to make the model deeper which unfortunately would make both training and inference slower since there would be more layers that'll waiting for the previous layer to finish their computation.

For small models which only require a couple of heads, scaling embedding/head dimension to 128-256 isn't much of an issue if you can bare the resulted increase in compute and memory requirements, but for larger models up to 100M parameters or more, scaling heads or head dimension wouldn't be enough so scaling depth will be need as well. This implies that as this architecture is scaled up in it's raw form, the increase compute and memory requirements would quickly lead it to lose all it's parameter savings and other advantages against the standard transformer.

### 7.3. Closing Thoughts
Silia is a small idea for small scale. The sub-10M parameter space is under explored and for good reasons, there isn't much glory in it. But I think there's some genuine value in asking whether the standard Transformer block is the right design when you only have a few hundred thousand parameters to spare. Merging attention and SwiGLU into a single unified operation isn't a revolutionary idea, but the parameter savings are real and the results are encouraging enough to be worth sharing. I hope this paper is useful to someone working in the same constrained corner of the field that I am.


## Acknowledgements
This work used compute from Google Colab.


## References
Andrej Karpathy, (2022). nanoGPT. https://github.com/karpathy/nanogpt.

Ofir Press, Lior Wolf, (2017). Using the Output Embedding to Improve Language Models. _arXiv preprint arXiv:1608.05859_.

Omkar Thawakar, Ashmal Vayani, Salman Khan, Hisham Cholakal, Rao M Anwer, Michael Felsberg, Tim Baldwin, Eric P Xing, Fahad Shahbaz Khan, (2024). Mobillama: Towards accurate and lightweight fully transparent gpt. _arXiv preprint arXiv:2402.16840_.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut, (2019). Albert: Alite bert for self-supervised learning of language representations. _arXiv preprint arXiv:1909.11942_.

Jonathan Frankle, Michael Carbin, (2018). The lottery ticket hypothesis: Finding sparse, trainable neural networks. _arXiv preprint arXiv:1803.03635_.

Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko, (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2704–2713_.

X Ma, J Zhang, R Wang, Q Xu, D Lin, (2019). Tensorized embedding layers for efficient model compression. _Advances in Neural Information Processing Systems_.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, (2019). Language Models are Unsupervised Multitask Learners. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia et al, (2024). The Llama 3 Herd of Models. _arXiv preprint arXiv:2407.21783_.

Ronen Eldan, Yuanzhi Lim, (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent English? _arXiv preprint arXiv:2305.07759_.

Srijan Srivastava, (2026). Webtext Super Tiny. https://huggingface.co/datasets/Srijan-Srivastava/webtext-super-tiny.

Kamisori-daijin, (2026). Email datasets 20k. https://huggingface.co/datasets/Kamisori-daijin/email-datasets-20k.

icip-cas, (2023). ChatAlpaca: A Multi-Turn Dialogue Corpus based on Alpaca Instructions. https://github.com/icip-cas/ChatAlpaca.

tatsu-lab, (2023). Stanford Alpaca: An Instruction-following LLaMA Model. https://github.com/tatsu-lab/stanford_alpaca.

codelion, (2025). PleIAs/SYNTH Sampled Dataset (100,000,000 tokens). https://huggingface.co/datasets/codelion/synth-100M.

PleIAs, (2025). SYNTH. https://huggingface.co/datasets/PleIAs/SYNTH.

codelion, (2025). Fineweb-edu-100M. https://huggingface.co/datasets/codelion/fineweb-edu-100M.

HuggingFaceFW, (2025). FineWeb-Edu. https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, Noah Fiedel, (2022). PaLM: Scaling Language Modeling with Pathways. _arXiv preprint arXiv:2204.02311_.

Noam Shazeer, (2020). GLU variants improve transformer. _arXiv preprint arXiv:2002.05202_.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, (2017). Attention is all you need. _arXiv preprint arXiv:1706.03762_.

Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin, (2025). Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free. _arXiv preprint arXiv:2505.06708_.

Shuangfei Zhai, (2026). Exclusive Self Attention. _arXiv preprint arXiv:2603.09078_.

DeepSeek-AI, (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. _arXiv preprint arXiv:2405.04434_.

Shuangfei Zhai, Walter Talbott, Nitish Srivastava, Chen Huang, Hanlin Goh, Ruixiang Zhang, Josh Susskind, (2021). An Attention Free Transformer. _arXiv preprint arXiv:2105.14103_.

Zhentao Tan, Wei Chen, Jingyi Shen, Yao Liu, Xu Shen, Yue Wu, Jieping Ye, (2026). HydraHead: From Head-Level Functional Heterogeneity to Specialized Attention Hybridization. _arXiv preprint arXiv:2606.20097_.

DeepSeek-AI, (2026). DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence. _arXiv preprint arXiv:2606.19348_.


## Citation

```
@software{Silia,
    author={Srijan Srivastava},
    title={Silia},
    url={https://github.com/SrijanSriv211/Silia},
    version={0.2.0},
    year = {2026}
}
```

<img src="img/requiem.png" alt="lookwhosback" style="width:100%;">
