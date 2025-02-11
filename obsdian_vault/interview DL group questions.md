---
project:
  - interview DL group questions
created: Wednesday, January 15th 2025, 11:17:39 am
  2025-01-15 11:17
subject:
- 
tags:
  - "#NormalNote"
date modified: Tuesday, January 21st 2025, 12:55:46 pm
---

# Interview DL Group Questions

## Chapter 1

![[Deep-learning_interview_questions.pdf#page=29&rect=41,230,497,318&color=important|Deep-learning_interview_questions, p.29]]

The main drawback is that we increase the chance of the model to overfit. It's only worth increasing the dimensionality of the problem with more independent variables.
also more variables = more compute.
![[Deep-learning_interview_questions.pdf#page=29&rect=42,159,494,230&color=important|Deep-learning_interview_questions, p.29]]
the ratio of the success event to the failure event.
$$
\text{odds of success} = \frac{p(x=\text{success})}{1-p(x=\text{success})}
$$
per example: if I really lasagna, and the probability of the cafeteria serving lasagna is 70%, then the odds of success would be:
$$
\text{odds of success} = \frac{0.7}{1-0.7} = \frac{0.7}{0.3} \approx 2.33
$$
So I'm $\hspace{0pt}2.33$ times more likely to eat lasagna than any other food.

![[Deep-learning_interview_questions.pdf#page=29&rect=42,69,497,157&color=important|Deep-learning_interview_questions, p.29]]

Interaction refers to when we use a combination of variables to produce an extra feature, given that the logistic regression model is linear, if we want to add an "interaction" term between two variables, like multiplication between them, we need to do it in the form of another feature.

Can't be a sum, because sums are already a part of the linear model $x_{3}=x_{1}+x_{2}$ does not contribute to the model, it's only redundant information.

![[Deep-learning_interview_questions.pdf#page=30&rect=41,575,496,601&color=important|Deep-learning_interview_questions, p.30]]
we could write the simplest interaction term as:
$$
x_{3}= x_{1}x_{2}
$$
so then the linear part of the model, would end up looking like:
$$
\alpha_{0}+\alpha_{1} x_{1}+ \alpha_{2}x_{2}+\alpha_{3}\underbrace{ x_{1}x_{2} }_{ x_{3} }
$$

![[Deep-learning_interview_questions.pdf#page=30&rect=40,553,495,576&color=important|Deep-learning_interview_questions, p.30]]
For this we can use some statistics such as wald chi-squared or Likelyhood ratio.

To apply the likelyhood ratio test, we would test the model against the reduced model ($\alpha_{3}=0$), and compute the test statistic as the ratio between the log likelyhoods:
$$
\Lambda = -2 [\ell(\text{reduced model})-\ell \text{(full umodel)}]
$$
then we compare this value with a chi-squared distribution, and look at the resulting $p$ value to figure out if it's worth it to add the interaction term.

![[Deep-learning_interview_questions.pdf#page=30&rect=45,451,494,519&color=important|Deep-learning_interview_questions, p.30]]
False, the given definition is for supervised learning, unsupervised learning tries to identify underlying structure in the data.

![[Deep-learning_interview_questions.pdf#page=30&rect=43,377,494,449&color=important|Deep-learning_interview_questions, p.30]]
in a group of binary or multi-class responses.

![[Deep-learning_interview_questions.pdf#page=30&rect=42,292,499,376|Deep-learning_interview_questions, p.30]]

In logistic regression the transformation applied to the response variable is sigmoid or a softmax function which is bounded between $\hspace{0pt}0$ and $\hspace{0pt}1$, which makes it a probability distribution.

and we consider this a more informative representation because we have a measure of how certain our model is in the output of the classification.

![[Deep-learning_interview_questions.pdf#page=30&rect=41,218,498,290&color=yellow|Deep-learning_interview_questions, p.30]]
minimizing the negative log likelihood of event $x$ given $y$, also means maximizing the likelihood of selecting the correct.

![[Deep-learning_interview_questions.pdf#page=30&rect=37,67,499,187&color=yellow|Deep-learning_interview_questions, p.30]]
![[Deep-learning_interview_questions.pdf#page=31&rect=41,574,500,604|Deep-learning_interview_questions, p.31]]
$$
\text{odds} = \frac{\text{P(A)}}{1-P(A)} =\frac{0.1}{1-0.1} = \frac{0.1}{0.9} = \frac{1}{9}
$$
$$
\log \text{odds} = \ln \frac{1}{9} = \ln 1 - \ln  9 = -2.197
$$

$odds = \frac{prob}{1-prob}$

$odds - odds \cdot prob = prob$

$prob = \frac{odds}{odds + 1}$ with odds = $\frac{1}{9}$

check: $0.1 = \frac{\frac{1}{9}}{\frac{1}{9} + 1}$

![[Deep-learning_interview_questions.pdf#page=31&rect=41,497,500,570&color=yellow|Deep-learning_interview_questions, p.31]]

$$
\text{odds} = 4 = \frac{P(A)}{1-P(A)}  = \frac{0.8}{1-0.8} = 4
$$
True!

![[Deep-learning_interview_questions.pdf#page=31&rect=40,427,499,497&color=yellow|Deep-learning_interview_questions, p.31]]
![[interview DL group questions 2025-01-21 16.09.00.excalidraw]]

![[Deep-learning_interview_questions.pdf#page=31&rect=41,326,499,423&color=yellow|Deep-learning_interview_questions, p.31]]
- Linear transformation of independent variable inputs - systematic component
- Distribution with an associated activation function - random component - distribution of y
- The function relating the link between random and systematic components, how the linear combination input variables $x$ relates to $Y$ which belongs to a probability distribution.

![[Deep-learning_interview_questions.pdf#page=31&rect=40,169,497,323&color=yellow|Deep-learning_interview_questions, p.31]]
![[interview DL group questions 2025-01-21 12.16.56.excalidraw]]
$$
\log\left(  \frac{0.5}{0.5} \right)= 0 = \theta_{0} + \theta^{T}x
$$
the 0.5 comes from the fact that on the boundary our probability of being in each class is equal, thus, 0.5.

![[Deep-learning_interview_questions.pdf#page=31&rect=40,97,497,168&color=yellow|Deep-learning_interview_questions, p.31]]
True, these are opposite of each other.

![[Deep-learning_interview_questions.pdf#page=32&rect=42,203,498,310&color=yellow|Deep-learning_interview_questions, p.32]]
$$
\begin{align}
\frac{d}{dx} \sigma(x)  &  = \frac{d}{dx} \frac{1}{1+e^{-x}}   \\
& = \frac{e^{-x}}{(1+e^{-x})^{2}}   \\
& = \frac{e^{-x}}{(1+e^{-x})(1+e^{-x})} \\
 & = \sigma(x) \frac{e^{-x}}{1+e^{-x}}
\end{align}
$$
![[Deep-learning_interview_questions.pdf#page=32&rect=44,57,497,198&color=yellow|Deep-learning_interview_questions, p.32]]
![[Deep-learning_interview_questions.pdf#page=33&rect=46,370,494,589&color=yellow|Deep-learning_interview_questions, p.33]]

1. $\text{logit}= \beta_{0}+\beta_{1}x_{1}+\beta_{2}x_{2}= -1$
2. odds
$$
\text{odds} = \frac{0.27}{1-0.27} = 0.37
$$
Fancy way:
$$
\text{logit}= \ln  \text{odds} \implies \text{odds}= e^{\text{logit}}
$$

3. probability
$$
P(y|x;\beta) = \frac{1}{1+e^{-(-1)}} \approx 0.27 
$$

![[Deep-learning_interview_questions.pdf#page=33&rect=40,66,509,343|Deep-learning_interview_questions, p.33]]
![[Deep-learning_interview_questions.pdf#page=34&rect=40,230,496,599&color=yellow|Deep-learning_interview_questions, p.34]]

1. The explanatory variable is the cancer type, because that's the different setting we're testing on, and the response variable is the tumour eradication, because that's what being quantified as response
2. 
   $$
\begin{align}
\text{Relative risk}:  & & \frac{P(\text{Yes}|\text{Breast})}{P(\text{Yes|Lung})}  &  & \frac{P(\text{No}|\text{Breast})}{P(\text{No|Lung})}   \\
\text{Odds ratio}:  &  &  \frac{\text{odds}(\text{Breast})}{\text{odds(Lung)}}
\end{align}
$$
Ratio between the probabilities of tumour eradication in each groups
$$
\frac{P(\text{Yes}|\text{Breast})}{P(\text{Yes|Lung})} = \frac{\frac{560}{560+260}}{\frac{69}{69+36}} = 1.0392
$$
Ratio between the probabilities of the tumour not being eradicated in each group
$$
\frac{P(\text{No}|\text{Breast})}{P(\text{No|Lung})} =0.93
$$
Odds ratio:
$$
\frac{\text{odds}(\text{breast})}{\text{odds}(\text{lung})} = \large{\frac{\frac{\text{success(breast)}}{\text{failure}(\text{breast})}}{\frac{\text{success(lung)}}{\text{failure(lung)}}}} = \frac{2.15}{1.92} = 1.12
$$
3. give that both the values of the relative risk and the values of the odds ratio are close to zero there's not a big association
4. this is computed based on $\log(\text{odds ratio})$ with an error given by $SE\times Z$.

We use the log to make this closer to a gaussian, and to compute this result we first need to find the standard error for the log of the odds ratio.

$$
\begin{align}
\log (\text{odds ratio})  & = \log \large{\frac{\frac{\text{success(breast)}}{\text{failure}(\text{breast})}}{\frac{\text{success(lung)}}{\text{failure(lung)}}}} = \log  \frac{s(b)f(l)}{s(l)f(b)}  \\
  & = \log  s(b) + \log  f(l) - \log s(l) - \log  f(b)
\end{align}  
$$
then we have that the variance of a Bernoulli is independent for each event?
$$
\text{Var} \log (\text{odds ratio})= \text{var}\log  s(b) + \text{var}\log  f(l) +\text{var} \log s(l) +\text{var} \log  f(b)
$$
then we have that the variance of the 
$$
\text{var}(\log x) \approx \frac{1}{x}
$$
then the standard error is:
$$
SE = \sqrt{ \frac{1}{s(b)}+ \frac{1}{f(l)} + \frac{1}{s(l)} + \frac{1}{f(b)}}
$$
And the $Z$ that gives us a confidence interval of 95% confidence (we find in a gaussian) is 1.96.

Then we just have to do:
$$
\begin{align}
\ln \text{odds ratio} \pm Z\times SE  & = \ln (1.12) \pm 1.96 \times 0.22 = \begin{cases}
0.54 \\
-0.31
\end{cases}
\end{align}
$$
and to take the log out of this.
$$
(e^{-0.31},e^{0.54}) = (0.73, 1.72)
$$
5.  according to this calculations the measurement is inside the 95% confidence interval, so there's no association

![[Deep-learning_interview_questions.pdf#page=34&rect=44,118,497,223&color=yellow|Deep-learning_interview_questions, p.34]]

1.  we can just substitute the probabilities in the sigmoid.
$$
p_{i} = \frac{\text{exp}(\beta_{0}+\beta_{1}x_{1}+\beta_{2}x_{2})}{1 + \text{exp}(\beta_{0}+\beta_{1}x_{1}+\beta_{2}x_{2})} = 0.378
$$
2.  We can derive this from the sigmoid formula
$$
\begin{align}
p_{i} (1+\exp(-6+0.05x_{1}+x_{2}))  & = \exp(-6+0.05x_{1}+x_{2})  \\
p_{i}  & = \exp(-6+0.05x_{1}+x_{2}) (1-p_{i}) \\
\frac{p_{i}}{1-p_{i}}  & = \exp(-6+0.05x_{1}+x_{2}) \\
\ln  \frac{p_{i}}{1-p_{i}}  &= -6 + 0.05x_{1}+x_{2} \\
 x_{1}  & = 6+ \ln  \frac{p_{i}}{1-p_{i}} - x_{2} = 5
\end{align}
$$

![[Deep-learning_interview_questions.pdf#page=36&rect=45,254,500,587&color=yellow|Deep-learning_interview_questions, p.36]]
![[Deep-learning_interview_questions.pdf#page=36&rect=45,71,500,249&color=yellow|Deep-learning_interview_questions, p.36]]
![[Deep-learning_interview_questions.pdf#page=37&rect=43,457,500,604&color=yellow|Deep-learning_interview_questions, p.37]]

1. First we interpret the values we got from the logistic regression.
	-  $\beta_{0}=-6.36$
	-  $\beta_{1}= -1.024$
	-  $\beta_{2}=0.119$
to solve this we do:
$$
\text{logits} = \ln (\text{odds}) \implies \text{odds} = \text{exp}(\beta_{0}+\beta_{1}x_{1}+\beta_{2}x_{2})
$$
proof:
$$
\begin{align}
\sigma (\vec{x})  & = \frac{e^{\beta X}}{1 + e^{\beta X}} \\
\text{odds} & = \frac{\frac{e^{\beta X}}{1 + e^{\beta X}}}{1 - \frac{e^{\beta X}}{1 + e^{\beta X}}} = \frac{e^{\beta X} }{1+e^{\beta X }- e^{\beta X}} = e^{\beta X}
\end{align}
$$

2. here we just need to substitute the values in the sigmoid

$$
p_{i} = \frac{\text{exp}(\beta_{0}+\beta_{1}x_{1}+\beta_{2}x_{2})}{1 + \text{exp}(\beta_{0}+\beta_{1}x_{1}+\beta_{2}x_{2})} = 0.989
$$
3. High coffee intake is associated with an increased probability of a second migraine because of the low p-value $0.305$, if we interpret it to mean that a p-value below 0.05 pct means that the relation is significant. and it's associated with an increased probability because of the sign of the coefficient being positive.
4. No, due to the high p-value (0.3818) we don't have meaningful statistical evidence of this claim.

