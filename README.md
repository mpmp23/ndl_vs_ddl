# NDL vs DDL: Legal Formalization and Verification

An exploratory comparison of **Neo-Davidsonian event semantics (NeoD)** and a **restricted DDL-style deontic representation** for formalizing legal provisions and answering legal inference questions with Z3.

This repository contains the implementation and benchmark for a study prepared for **LexFlow 2027**. It is a research prototype, not a legal advice or compliance system.

## Research Question

How do DDL-style rules and Neo-Davidsonian event representations compare in their ability to support automated legal inference, and where do their formalization and verification pipelines fail?

The study uses **§ 8.2.1 (complaint management)** of the Australian Telecommunications Consumer Protections Code.

The benchmark consists of **20 questions**, with gold labels of:

- Entailment (7)
- Contradiction (7)
- NotMentioned (6)

Each question identifies the relevant legal subsections and includes a factual situation and a hypothesis.

## Methodology

The experiment consists of five stages:

1. **Legal encoding:** Formalize 22 legal snippets in either DDL-style rules or Neo-Davidsonian event semantics.
2. **Query translation:** Use the same query-translation model (`openai/gpt-6-astra` through OpenRouter) to translate each situation and hypothesis using the vocabulary of the saved encoding. Unrepresentable hypotheses may be marked `unsupported`; predicates are not silently harmonized or invented.
3. **Compilation:** Compile saved rules and translated queries into Z3 using `compile_ddl.py` or `compile_neod.py`.
4. **Verification:** Perform consistency, entailment, and contradiction checks with Z3 to classify each hypothesis.
5. **Evaluation:** Compare classifications against the gold labels, measuring accuracy, precision, recall, F1, classification coverage, compilation success, runtime, and token usage.

### Formalization Frameworks

**DDL-style representation**

Uses conditionals over predicates and deontic operators for obligations `[O]`, permissions `[P]`, and prohibitions `[F]`.

**Neo-Davidsonian representation**

Uses explicit event variables, predicates, thematic roles, quantification, and deontic operators to represent legally relevant actions and conditions.

Both frameworks use a restricted normative semantics. Prohibition is interpreted as an obligation not to perform an action (`[F]A = [O]¬A`). The implementation does not include defeasible superiority relations or rule priorities.

The NeoD compiler supports a restricted event and predicate syntax and reifies deontic action descriptions into modal schemas.

Consequently, the experiment compares **specific formalization and verification pipelines**, not the theoretical expressiveness of either formalism in general.

### Model Configurations

The experiment evaluates nine model slots across both frameworks.

- Six historical DDL XML baselines attributed to Horner et al.
- Three newly generated DDL model outputs.
- Nine newly generated Neo-Davidsonian model outputs.

All verification questions use GPT-6 Astra as the query translator.

**Model-version caveat:** The historical Gemini 3 Pro DDL output is compared against a newly generated Gemini 3.1 Pro NeoD output. This is not an exact same-version comparison.

Model configurations are documented in `models.yaml`.

## Benchmark and Evaluation

The benchmark contains:

- 20 legal inference questions
- 9 model slots
- 2 formalization frameworks
- 360 verification cases

Each case produces either an Entailment, Contradiction, or NotMentioned classification, or a separate failure status such as `unsupported`, `compile_error`, or `inconsistent`.

### Experimental Results

| Metric | DDL | NeoD |
|---|---:|---:|
| Correct predictions | 98/180 | 48/180 |
| End-to-end accuracy | 54.4% | 26.7% |
| Successfully classified cases | 102/180 | 102/180 |
| Classification coverage | 56.7% | 56.7% |
| Accuracy among classified cases | 96.1% | 47.1% |
| End-to-end Macro-F1 | 0.695 | 0.350 |
| Unsupported cases | 72 | 39 |
| Compilation errors | 4 | 36 |
| Inconsistent outcomes | 2 | 3 |

End-to-end accuracy counts unsupported and failed cases as unsuccessful attempts. Accuracy among classified cases considers only cases receiving a valid E/C/N classification.

End-to-end Macro-F1 includes unclassified cases as false negatives for their gold classes.

These are descriptive findings from a small exploratory benchmark. They do not establish that one formalism is intrinsically superior.

## Repository Structure

| File or directory | Purpose |
|---|---|
| `snippets.xml` | Legal text divided into formalization snippets |
| `scenarios.csv` | Benchmark scenarios |
| `queries.xml` | Legal inference questions |
| `gold_labels.csv` | Gold classifications and rationales |
| `models.yaml` | Model configurations |
| `prompts/encode/` | DDL and NeoD encoding prompts |
| `prompts/verify/` | Framework-specific query translation prompts |
| `run_encode.py` | Formalization generation |
| `run_verify.py` | Verification pipeline |
| `translate_query.py` | Natural-language query translation |
| `compile_ddl.py` | DDL-to-Z3 compiler |
| `compile_neod.py` | NeoD-to-Z3 compiler |
| `verify.py` | Shared Z3 reasoning engine |
| `evaluate.py` | Evaluation and metrics |
| `baselines/horner_ddl/` | Historical DDL baseline XML files |
| `runs/evaluation/` | Final evaluation results |

Raw generated encodings (`outputs/encodings/`), verification traces (`runs/verification/`), and archived attempts (`old_verification_outputs/`) are excluded from the initial Git commit pending review.

As a result, a fresh clone includes the code, benchmark, historical baselines, and evaluation summaries, but cannot fully reproduce the 360-case evaluation without the excluded raw artifacts.

## Installation

The project was developed using Python 3.12.7.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Model requests use the OpenRouter API.

Create a local `.env` file containing:

```dotenv
OPENROUTER_API_KEY=your_key_here
```

Never commit an API key to the repository.

## Running the Experiment

### 1. Inspect available formalizations

```bash
python run_verify.py --all-models --list-encodings
```

### 2. Dry-run a verification case

```bash
python run_verify.py \
  --queries queries.xml \
  --query-id Q001 \
  --model gpt52 \
  --framework ddl \
  --dry-run
```

Neither command makes an API request.

### 3. Run one verification case

```bash
python run_verify.py \
  --queries queries.xml \
  --query-id Q001 \
  --model gpt52 \
  --framework ddl
```

This makes an OpenRouter API call and saves a new verification attempt. Model requests may incur charges.

### 4. Evaluate saved results

When the raw `runs/verification/` artifacts are available locally:

```bash
python evaluate.py
```

This script does not call OpenRouter or generate new verification attempts. It reads saved records and writes evaluation summaries to `runs/evaluation/`, replacing existing summary files.

For a separate evaluation directory:

```bash
python evaluate.py --out /tmp/ndl_vs_ddl_evaluation
```

## Limitations and Reproducibility

The experiment has several important limitations:

- The benchmark contains only 20 questions.
- The DDL and NeoD compilers support restricted formal languages rather than complete legal or deontic reasoning systems.
- Some legal representations exceed the supported compiler grammar.
- Business-day and temporal semantics are simplified.
- The Gemini DDL and NeoD outputs use different model versions.
- Unclassified cases are retained as failures for end-to-end evaluation.
- The current initial repository excludes raw encoding-generation and verification artifacts pending review.

Original model outputs and verification artifacts are preserved locally without silent predicate reconciliation.

The study's findings should therefore be interpreted as properties of the implemented pipelines.

## Attribution

The historical DDL baseline artifacts are attributed to Horner et al. Full bibliographic references, provenance details, and applicable redistribution terms should be documented before public release.

The legal source material is drawn from the Australian Telecommunications Consumer Protections Code.

## Project Status

**Exploratory experiment completed.**

Quantitative evaluation is complete. Qualitative failure analysis, expanded methodological documentation, and release of reviewed research artifacts are ongoing.

Prepared for LexFlow 2027.