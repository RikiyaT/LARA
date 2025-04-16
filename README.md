# LARA: LLM-Assisted Relevance Annotation

LARA is a flexible framework that leverages **large language models (LLMs)** for active learning and relevance annotation calibration. It supports both **binary relevance** and **graded relevance** annotation tasks, enabling efficient creation of high-quality test collections.

---

## 🚀 Quick Start Guide

Run the evaluation scripts as follows:

### 🔹 Binary Relevance (0/1 labels)

```bash
python src/test-al.py Ours Random Naive Pool MTF OursGroup[N]
```

### 🔸 Graded Relevance (0~k labels)

```bash
python src/test-al-graded.py Ours Random Naive Pool MTF OursGroup[N]
```

- `Ours`: Proposed LARA method.
- `OursGroup[N]`: LARA with N annotators grouped.

---

## 📂 Input File Format (CSV)

Provide the LLM prediction results in a `.csv` file.

### Binary Relevance Format:

```csv
topic_id,doc_id,annotation,prob_yes,prob_no
```

- `topic_id`: ID of the query/topic.
- `doc_id`: Document identifier.
- `annotation`: Predicted binary label (0 or 1).
- `prob_yes`: Confidence that the label is relevant (1).
- `prob_no`: Confidence that the label is not relevant (0).

### Graded Relevance Format:

```csv
topic_id,doc_id,annotation,prob_0,...,prob_k
```

- `annotation`: Predicted graded label (0–k).
- `prob_n`: Confidence for label `n`.

📌 **Example Implementations** (using [TREC-COVID from BEIR](https://huggingface.co/datasets/BeIR/trec-covid-qrels)):

- Binary relevance example: `src/example-covid-itachi.py`
- Graded relevance example: `src/example-covid-itachi-graded.py`

---

## 📝 Prompt Templates

We tested the following prompts in the paper:
- 📘 **Simple Prompt**
- 🔧 **Utility Prompt** based on [[Thomas et al., 2024]](https://arxiv.org/abs/2309.10621)
- 📙 **Rational Prompt** based on [[Upadhyay et al., 2024]](https://arxiv.org/abs/2411.08275)

{description} and {narrative} are given by the NIST assessors.

### 🔹 Binary Relevance (0/1)

These prompts produce **Yes (1)** or **No (0)** responses.

#### 📘 **Simple Prompt (Binary)**

> **Consider the following web page content:**  
> —BEGIN WEB PAGE CONTENT—  
> *{text}*  
> —END WEB PAGE CONTENT—  
>
> **Setting:**  
> A person has typed "*{query}*" into a search engine.  
> Intent: "*{description}*".  
>
> **Instruction:**  
> Answer if the web content is relevant to the query. *{narrative}*  
>
> Answer **yes** or **no**.  
>
> **Your answer:**

---

#### 🔧 **Utility Prompt (Binary)** [Thomas et al., 2024]

> Given a query and web page, provide a **Yes** or **No** answer indicating relevance.
>
> Imagine writing a report on the topic.  
> If the web page contains **vital information**, answer **Yes**; otherwise, answer **No**.
>
> **Query:** "*{query}*"  
> Intent: "*{description} {narrative}*"
>
> **Web Page:**  
> —BEGIN WEB PAGE CONTENT—  
> *{text}*  
> —END WEB PAGE CONTENT—
>
> **Instruction:**  
> Is the web content relevant to the query?
>
> Answer **yes** or **no**.  
>
> **Your answer:**

*Based on the utility framing by [Thomas et al. (2024)](https://arxiv.org/abs/2309.10621).*

---

#### 📙 **Rational Prompt (Binary)** [Upadhyay et al., 2024]

> You're an expert content judge. Use commonsense reasoning to determine relevance.
>
> - **Yes**: Passage dedicated to query, contains answer.  
> - **No**: Passage unrelated to query.
>
> Only return **Yes** or **No**.
>
> **Query:** "*{query}*"  
> Intent: "*{description} {narrative}*"
>
> **Passage:** "*{text}*"
>
> **Instruction:**  
> Is the web content relevant to the query?
>
> **Your answer:**

*Based on rational prompt style from [Upadhyay et al. (2024)](https://arxiv.org/abs/2411.08275).*

---

### 🔸 Graded Relevance (example with k=2)

#### 📘 **Simple Prompt (Graded)**

> **Consider the following web page content:**  
> —BEGIN WEB PAGE CONTENT—  
> *{text}*  
> —END WEB PAGE CONTENT—  
>
> A person typed "*{query}*" into a search engine.  
> Intent: "*{description}*".
>
> **Instruction:**  
> Answer relevance of web content. *{narrative}*  
>
> Answer **2** (highly relevant), **1** (partially relevant), or **0** (not relevant).  
>
> **Your answer:**

---

#### 🔧 **Utility Prompt (Graded)** [Thomas et al., 2024]

> Given a query and web page, provide a relevance score:
>
> - **2**: Highly relevant, very helpful.
> - **1**: Relevant, partially useful.
> - **0**: Not relevant.
>
> Imagine writing a report.  
> If vital information is present, score **2**; if partly useful, **1**; else **0**.
>
> **Query:** "*{query}*"  
> Intent: "*{description} {narrative}*"
>
> **Web Page:**  
> —BEGIN WEB PAGE CONTENT—  
> *{text}*  
> —END WEB PAGE CONTENT—
>
> **Instructions:**  
> Provide a relevance score (no explanation).  
>
> **Relevance Score:** {

*Based on the utility approach from [Thomas et al. (2024)](https://arxiv.org/abs/2309.10621).*

---

#### 📙 **Rational Prompt (Graded)** [Upadhyay et al., 2024]

> You're an expert judge. Use commonsense reasoning to determine relevance:
>
> - **2**: Highly relevant
> - **1**: Somewhat relevant
> - **0**: Not relevant
>
> Provide only the categorical value, no explanation.
>
> **Query:** "*{query}*"  
> Intent: "*{description} {narrative}*"
>
> **Passage:** "*{text}*"
>
> **Instruction:**  
> Is the web content relevant to the query?
>
> **Relevance Score:** {

*Based on rational prompts in [Upadhyay et al. (2024)](https://arxiv.org/abs/2411.08275).*

---

## 📚 References

If you use or refer to LARA, please cite the following papers associated with the prompt templates:

- Rational prompts:  
  > Shivani Upadhyay et al. (2024). "A Large-Scale Study of Relevance Assessments with Large Language Models: An Initial Look." arXiv: [2411.08275](https://arxiv.org/abs/2411.08275).

- Utility prompts:  
  > Paul Thomas et al. (2024). "Large language models can accurately predict searcher preferences." arXiv: [2309.10621](https://arxiv.org/abs/2309.10621).

- TREC-COVID dataset used in the example (from BEIR):  
  > [https://huggingface.co/datasets/BeIR/trec-covid-qrels](https://huggingface.co/datasets/BeIR/trec-covid-qrels)
