---
created: Saturday, December 28th 2024, 3:24:29 pm
date modified: Monday, January 20th 2025, 3:45:32 pm
---
⭐ - None
⭐⭐ - Know a small tiny bit
⭐⭐⭐ - I studied it but I think I forgot it
⭐⭐⭐⭐ - Know it
⭐⭐⭐⭐⭐ - Know it and have practice in it

| Topic                                   | Familiarity (0-5) |
| --------------------------------------- | ----------------- |
| Logistic Regression                     | ⭐⭐⭐               |
| Probabilistic Programming & Bayesian DL | ⭐                 |
| Information Theory                      | ⭐                 |
| Deep Learning Calculus                  | ⭐⭐⭐⭐              |
| Deep Learning: NN Ensembles             | ⭐                 |
| Deep Learning CNN Feature Extraction    | ⭐⭐⭐               |
| Deep Learinng                           | ⭐⭐⭐⭐              |

----
# Interview Prep

Question $\hspace{0pt}4$ for each chapter

## My Questions
### Chapter 2
[[Deep-learning_interview_questions.pdf#page=30&annotation=6223R|PRB-7    CH.PRB- 2.4. True or False:  In machine learning terminology, unsupervised learning refers to the mapping of input covariates to a target response variable that is attempted at being predicted when the labels are known]]

nope, in unsupervised learning we try to identify patterns, groups or features in the data without any labels, we can also have dimensionality reduction as a form of unsupervised learning.
### Chapter 3
![[Pasted image 20250110112859.png]]
[[Deep-learning_interview_questions.pdf#page=60&annotation=6226R|PRB-33    CH.PRB- 3.4. Find the probability mass function (PMF) of the following random variable: X  ∼  Binomial(n, p)  (3.2)]]

The binomial distribution is the coin flip distribution, where we the coin can or not be weighted, so we have $n$ trails, with a probability of success given by $p$.

to deduce the formula let's consider some examples:
given SSFFFSF the probability for that specific sequence is:
we have $p^{3}(1-p)^{4}$, so if we take this to the general case, we have
$p^{k}(1-p)^{n-k}$, where $k$ tells us how many successes we had.

but we need to consider all the possible ways to arrange $k$ successes in $n$ trials, so we have:
$$
X\sim \text{binomial} (n,p) = \begin{pmatrix}
n \\
k 
\end{pmatrix}
p^{k}(1-p)^{n-p}
$$
![[Deep-learning_interview_questions.pdf#page=61&rect=42,168,505,330|Deep-learning_interview_questions, p.61]]

$$
P(A|B) = \frac{P(A,B)}{P(B)}= \frac{P(B|A)P(A)}{P(B)}
$$
this comes from:
$$
P(A,B)=P(A|B)P(B)=P(B|A)P(A)
$$
- $P(A|B)$ - posterior probability
- $P(B|A)$ - likelihood
- $P(A)$ - prior probability
- $P(B)$ - Marginal

![[Interview Prep DL 2025-01-20 14.35.58.excalidraw|300]]





### Chapter 4
![[Deep-learning_interview_questions.pdf#page=106&rect=45,75,500,277&color=yellow]]
For an event that's certain to happen we have no entropy, because the entropy measures the uncertainty present
![[Deep-learning_interview_questions.pdf#page=107&rect=45,450,438,600&color=yellow]]The entropy is maximum here.
for the discrete case:
$$
\begin{align}
\mathcal{H}(X) = \sum_{i} p(x_{i})\log_{2}(p(x_{i})) & = \sum_{N} p(x_{n})\log_{2}\left( \frac{1}{p(x_{n})} \right) \\
 & = N \frac{1}{N} \log_{2}\left( N\right)  = \log_{2}(N)
\end{align}
$$
for the continuous case:
$$
\mathcal{H}(X) = \int_{a}^{b} p(x) \log_{2}\left( \frac{1}{p(x)} \right)??
$$
![[Deep-learning_interview_questions.pdf#page=107&rect=42,156,501,241|Deep-learning_interview_questions, p.107]]

It's a measure of information, and it's the amount of information needed to resolve a binary problem, such as a Bernoulli with $p=0.5$, where we need only one bit of information since there's a 50% chance of each outcome.

these are used to quantify the information gained by reducing entropy.

### Chapter 5
![[Deep-learning_interview_questions.pdf#page=144&rect=44,166,490,408&color=yellow]]
$$
g(x) = 1-x+2x^{2}
$$
$$
\begin{align}
\frac{dg}{dx}  & = \lim_{ h \to 0 } \frac{g(x+h)-g(x)}{h}  \\
 & =
\lim_{ h \to 0 } \frac{\cancel{ 1 }-(\cancel{ x }+h)+2(x+h)^{2} - \cancel{ 1 + x } - 2x^{2}}{h}  \\
 & = \lim_{ h \to 0 } \frac{h+ 2(x+h)^{2}-2x^{2}}{h} = \lim_{ h \to 0 } \frac{h+4hx+2h^{2}}{h}  \\
 & = 1+ 4x
\end{align}
$$
![[Pasted image 20250120145123.png]]

$$
f(x + h) = f(x) + f'(x)h + f''(x) \frac{h^{2}}{2!} + \dots
$$
$$
\log_{n}(x) = \frac{\ln (x)}{\ln (10)} \implies \frac{d}{dx} \log (x)= \frac{1}{x \ln (10)} \implies \frac{d^{2}}{dx^{2}} \log (x) = -\frac{1}{x^{2}\ln (10)} 
$$
if we center the expansion at some point $h$:
$$
\log(h+x) = \log(h)+ \frac{1}{h \ln (10)} x- \frac{1}{h^{2}\ln (10)} \frac{x^{2}}{2} + \frac{2}{h^{3}\ln 10} \frac{x^{3}}{6} - \dots
$$



### Chapter 6
![[Pasted image 20250110113206.png]]
![[Deep-learning_interview_questions.pdf#page=205&rect=47,387,491,597&color=yellow]]

This seems like bootstrapping, like the first step of bagging.
![[Deep-learning_interview_questions.pdf#page=207&rect=42,239,497,485|Deep-learning_interview_questions, p.207]]
![[Deep-learning_interview_questions.pdf#page=208&rect=40,397,496,601|Deep-learning_interview_questions, p.208]]

I'm guessing this is just averaging the outputs from the different models, so this is just averaging.

![[Pasted image 20250120152619.png]]
wtf, not even resnet?

1. 19
2. 16 convolutional and 3 dense ones
3. 224x224
4. 64
5. substracted
6. small 3x3
7. 1
8. 5 Conv layers with 3x3 kernels and stride of 1 padding 1 - 5 pooling layers with stride of 2
9. ReLU
10. dense
11. 4096 higher level features each
12. 1000 features
13. softmax
14. dropout is not being used

### Chapter 7
![[Deep-learning_interview_questions.pdf#page=224&rect=47,276,492,342&color=yellow]]

Yes, because in the latter layers you might find more useful features than in the earlier ones, so these are the ones most useful for further training.
### Chapter 8
![[Deep-learning_interview_questions.pdf#page=249&rect=48,104,490,194&color=yellow]]

True

![[Pasted image 20250120154047.png]]
![[Pasted image 20250120154059.png]]

This approach is three fold CV



### Extra Questions

### Whats Image Segmentation and what Are Its Applications?

The idea of image segmentation is to assign labels at the pixel level, which would allows us to discern which parts of the image belong to that label.

To do this we need to scale up the neural networks after compressing the dimensions down if we're doing this with CNNs, some architectures like U-net were made to deal with this kind of problem.
One problem with this type of task is the amount of annotated data available, so it's common to use simulated data (from graphics plataforms) followed by DA (domain adaptation) or even use unsupervised domain adaptation (UDA), which are techniques to deal with distributional shift we might find between samples and real world cases.

Some important applications of image segmentation are in robot vision and autonomous driving.

### What is Object Detection, and how Does it Differ from Image Classification?

While image segmentation has pixel label resolution in identifying which label those pixels belong to, object detection only focuses on identifying where objects are at a lower resolution, such as placing a box over the detected object.