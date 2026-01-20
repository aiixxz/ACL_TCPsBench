

# TCPsBench

**TCPsBench** is a benchmark for evaluating multimodal large language models (MLLMs) on **traditional Chinese painting appreciation**, with a focus on culturally grounded interpretation rather than surface-level visual recognition.

---



## Benchmark Overview

We introduce **TCPsBench**, a benchmark grounded in the classical *Six Principles of Painting*, which provide a systematic framework for evaluating both formal techniques and spiritual expression in traditional Chinese painting.

### Key Statistics

* **Paintings:** 1,679 traditional Chinese paintings
* **Time span:** Wei–Jin period to Qing dynasty (220–1912 AD)
* **Sources:** 6 open-access museum archives
* **Questions:** 10,281 high-quality evaluation questions

### Capability Dimensions

TCPsBench evaluates MLLMs across **three core capability dimensions**:

1. **Basic Recognition**
   Understanding objects, elements, and visual components in paintings.
2. **Semantic Understanding**
   Interpreting artistic concepts, symbolism, and stylistic meaning.
3. **Transfer Reasoning**
   Applying artistic knowledge to novel contexts and reasoning across concepts.

These dimensions are instantiated into **eight evaluation task categories**, covering both low-level perception and high-level artistic interpretation.

### Data Construction

The benchmark is built through a systematic pipeline that includes:

* Data collection from authoritative museum sources
* Expert-guided annotation
* Question generation across multiple formats
* Multi-stage validation to ensure quality and consistency

---

## Evaluation

Using TCPsBench, we evaluate **10 mainstream MLLMs**, including both closed-source and open-source models.

### Key Findings

* Overall performance on traditional Chinese painting appreciation remains limited.
* Models perform particularly poorly on **professional-level tasks**, such as:

  * composition analysis
  * painting technique recognition
* **Closed-source models consistently outperform open-source models** across most tasks.
* Models perform more reliably on **constrained question formats** (e.g., multiple-choice).
* **Open-ended questions** that require integrated visual perception, artistic knowledge, and natural language generation remain highly challenging.


---

## Repository Structure
All QA pairs are stored under the Data/ directory, where each evaluation task corresponds to a single *.jsonl file. The Eval/ directory contains the evaluation code used to assess model performance on TCPsBench.
```text
TCPsBench/
├── Data/
│   ├── T1.jsonl
│   ├── T2.jsonl
│   ├── ...
├── Eval/
│   ├── scripts/
└── README.md




