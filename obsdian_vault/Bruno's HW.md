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

![[Deep-learning_interview_questions.pdf#page=34&rect=40,111,498,225&color=important|Deep-learning_interview_questions, p.34]]
![[Deep-learning_interview_questions.pdf#page=35&rect=37,68,497,610&color=important|Deep-learning_interview_questions, p.35]]

for a bernouli distribution we have $P(X=1)= p$, so I assume we can just replace the values here
$$
p_{i} = \frac{e^{-6+0.05x_{1}+x_{2}}}{1+e^{-6+0.05x_{1}+x_{2}}} =  \frac{0.606530660}{1+0.606530660} = 0.377540669
$$
for the second part we need to derive $x_{1}$ for the case where $p_{i}=0.5$

$$
\begin{align}
p_{i} (1+\exp(-6+0.05x_{1}+x_{2}))  & = \exp(-6+0.05x_{1}+x_{2})  \\
p_{i}  & = \exp(-6+0.05x_{1}+x_{2}) (1-p_{i}) \\
\frac{p_{i}}{1-p_{i}}  & = \exp(-6+0.05x_{1}+x_{2}) \\
\ln  \frac{p_{i}}{1-p_{i}}  &= -6 + 0.05x_{1}+x_{2} \\
 x_{1}  & = 6+ \ln  \frac{p_{i}}{1-p_{i}} - x_{2}
\end{align}
$$



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

![[Bruno's HW 2025-01-27 11.28.57.excalidraw]]

![[Deep-learning_interview_questions.pdf#page=62&rect=43,354,498,412&color=important|Deep-learning_interview_questions, p.62]]

$$
p(y|x) = \frac{p(x|y) p(y)}{p(x)}
$$
where:
- $p(y|x)$ is the posterior
- $p(x|y)$ is the likelihood
- $p(y)$ is the prior 
- and $p(x)$ is the evidence or marginal likelihood and often we marginalize over it
$$
p(x) = \sum_{y} p(x|y) p(y)
$$





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

![[Deep-learning_interview_questions.pdf#page=108&rect=42,72,499,251&color=important|Deep-learning_interview_questions, p.108]]

we'll if we're using entropy as the metric for uncertainty, and we're using the of a uniform distribution where $p(X=k)= 1/k$, and we're increasing or decreasing the probability by adding more outcomes, then we get:
$$
\mathcal{H} = -\sum_{a=1}^{N} P_{a} \log_{2} P_{a} = - \sum_{a=1}^{K} \frac{1}{K} \log  \frac{1}{K} = -\log  \frac{1}{K} = \log  (K)
$$
okay this wasn't a good example because this is just a straight line.
Let's do a bernoulli with $p(X=1)= \theta$ instead.
$$
\mathcal{H} = - \sum_{a=1}^{N} P_{a} \log_{2} P_{a} = - (\theta \log _{2}\theta + (1-\theta) \log_{2}(1-\theta)) 
$$
![[Pasted image 20250127115652.png|400]]

The curve is symmetrical

![[Deep-learning_interview_questions.pdf#page=109&rect=38,554,496,607&color=important|Deep-learning_interview_questions, p.109]]
The curve rises to a maximum when the probabilities are equal.

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

![[Deep-learning_interview_questions.pdf#page=147&rect=43,293,500,474&color=important|Deep-learning_interview_questions, p.147]]

$$
\begin{align}
1. &  \lim_{ x \to 3 } \frac{e^{x^{3}}-e^{27}}{3x-9} \to \text{l'hopital} \to \lim_{ x \to 3 } \frac{3x^{2} e^{x^{3}}}{3} = 9e^{27} \\
2. &  \lim_{ x \to 0 } \frac{e^{x^{2}}-x-1}{3\cos x -x-3} \implies \lim_{ x \to 0 } \frac{2xe^{x^{2}}-1}{-3 \sin x - 1}=1 \\
3.  & \lim_{ x \to \infty } \frac{x-\ln x}{\sqrt[100]{x} +4} \implies \lim_{ x \to \infty } \frac{1-1/x}{\frac{1}{100}x^{-99/100}} = \frac{x - 1}{x} 100 x^{99/100} = \infty
\end{align}
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

![[Deep-learning_interview_questions.pdf#page=208&rect=42,62,498,105&color=important|Deep-learning_interview_questions, p.208]]
![[Deep-learning_interview_questions.pdf#page=209&rect=54,442,506,601&color=important|Deep-learning_interview_questions, p.209]]

1. True
2. True?
3. False, bootstraping is when we train using various random subsets sampled with replacement
4. True
### Chapter 7
![[Deep-learning_interview_questions.pdf#page=224&rect=47,276,492,342&color=yellow]]

Yes, because in the latter layers you might find more useful features than in the earlier ones, so these are the ones most useful for further training.

![[Deep-learning_interview_questions.pdf#page=228&rect=44,101,506,485&color=important|Deep-learning_interview_questions, p.228]]
![[Deep-learning_interview_questions.pdf#page=369&rect=119,437,424,603&color=important|Deep-learning_interview_questions, p.369]]
okay so to do this we need to exclude the last two layers because we want to stop at the last convolutional block.

```python
class ResNetBottom(torch.nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        # Extract layers up to the last two (conv4_x)
        self.features = torch.nn.Sequential(
            *list(original_model.children())[:-2]
        )

    def forward(self, x):
        # Pass the input through the extracted feature layers
        x = self.features(x)
        # Apply global average pooling to get a 512-dimensional feature vector
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # flatten vector into 2d where each row represents a feature vector
        x = x.view(x.size(0), -1)
        return x
```

### Chapter 8
![[Deep-learning_interview_questions.pdf#page=249&rect=48,104,490,194&color=yellow]]

True

![[Pasted image 20250120154047.png]]
![[Pasted image 20250120154059.png]]

This approach is three fold CV

![[Deep-learning_interview_questions.pdf#page=252&rect=40,209,497,344&color=important|Deep-learning_interview_questions, p.252]]

Both are correct.

The cross-correlation operator compares the similarity of two signals $f$ and $g$ across locations by doing:
$$
R_{fg}[n] = \sum_{k=-\infty}^{\infty} f[k]\cdot g[k+n]
$$
Autocorrelation is a special case of cross-correlation where the same signal is compared with a shifted version of itself
$$
R_{f}[n] = \sum_{k=-\infty}^{\infty} f[k]\cdot f[k+n]
$$
The result gives us how similar the same signal is to a shifted version of itself.
### Extra Questions

### Whats Image Segmentation and what Are Its Applications?

The idea of image segmentation is to assign labels at the pixel level, which would allows us to discern which parts of the image belong to that label.

To do this we need to scale up the neural networks after compressing the dimensions down if we're doing this with CNNs, some architectures like U-net were made to deal with this kind of problem.
One problem with this type of task is the amount of annotated data available, so it's common to use simulated data (from graphics plataforms) followed by DA (domain adaptation) or even use unsupervised domain adaptation (UDA), which are techniques to deal with distributional shift we might find between samples and real world cases.

Some important applications of image segmentation are in robot vision and autonomous driving.

### What is Object Detection, and how Does it Differ from Image Classification?

While image segmentation has pixel label resolution in identifying which label those pixels belong to, object detection only focuses on identifying where objects are at a lower resolution, such as placing a box over the detected object.

