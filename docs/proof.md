### 📐 Proof of Theorem: Margin and Error Relationship

Let \( p(y = j \mid \pi) \) be the true relevance probability of relevance level \( j \).  
The Bayes-optimal classifier predicts:

\[
k = \arg\max_j \, p(y = j \mid \pi)
\]

The corresponding classification error is:

\[
\text{Error}(\pi) = 1 - p(y = k \mid \pi)
\]

Now define:

- \( s = \arg\max_{j \neq k} \, p(y = j \mid \pi) \) — the second most likely label  
- \( m = p(y = k \mid \pi) - p(y = s \mid \pi) \geq 0 \) — the **margin**

Maximizing the classification error is equivalent to minimizing \( p(y = k \mid \pi) \), under the constraint \( p(y = k \mid \pi) \geq p(y = s \mid \pi) \).  
This minimum is achieved when the margin is zero:

\[
p(y = k \mid \pi) = p(y = s \mid \pi)
\]

At this point, the classifier is maximally uncertain.  
Thus, minimizing \( m \) yields the greatest potential reduction in classification error.
