# coin-stats
Just my dive into simple statistics

### Goal
Understand, using coin tosses as an example, how these concepts work:
- Type I error (alpha)
- Type II error (beta)
- test power (1 - beta)
- p-value

### Plan
- **Formulate hypotheses**
  - H0: the coin is fair, p = 0.5
  - H1: the coin is biased, p != 0.5 (also consider a one-sided alternative separately)
- **Choose the statistic and the test**
  - observation: k heads out of n tosses
  - test: binomial test (optionally compare with the normal approximation)
- **Define the decision rule**
  - choose the significance level alpha
  - build the critical region under H0 (two-sided / one-sided)
- **Understand Type I/II errors and power**
  - alpha: the area of the critical region under the distribution assuming H0 is true
  - beta: the probability of failing to reject H0 when H1 is true
  - power: 1 - beta
- **Understand p-value**
  - interpretation: “how extreme the observation is assuming H0”
  - visually: the tail area(s) under the H0 distribution relative to the observed k
- **Create visualizations (in a notebook)**
  - overlay the distribution of k under H0 and under a chosen H1
  - highlight the alpha and beta regions
  - a separate plot for p-value (tail(s) under H0)
- **Experiments**
  - how alpha, beta, and power change with n and effect size (p in H1)
  - simulations vs analytics (cross-check results)

# installation

```bash
pip install uv
uv pip install -r requirements.txt
```

