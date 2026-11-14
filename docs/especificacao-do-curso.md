# Course Project Specification

*Large Language Models*

## 1. Overview

The course project is a **group-based investigation** of a problem involving large language models. Projects are **intentionally open-ended**: a group may investigate, implement, compare, reproduce, extend, or evaluate a method or system related to topics covered in the course.

The goal is not to build the largest possible system, but to conduct a **well-scoped technical investigation** with a **clear objective and evidence** that supports the conclusions. Successful projects typically include the following components:

- A clear problem, research question, hypothesis, or technical objective;
- A meaningful implementation, system, or experimental component;
- An evaluation aligned with the project question;
- Appropriate baselines or points of comparison, when applicable;
- Analysis of results, failures, limitations, and threats to validity;
- An explicit connection to concepts discussed in the course.

## 2. Groups

Projects should use the **same groups formed for the seminar presentations**. The seminar activities and the project are separate course components, but the group composition should remain the same. Any proposed change in group composition must be communicated to the instructors.

## 3. Possible Project Formats

Groups may choose from a variety of project formats, including:

- **Experimental study:** evaluate a model, method, hypothesis, design choice, or inference strategy.
- **Reproduction and extension:** reproduce a reported result and extend it with a new dataset, model, baseline, ablation, or evaluation.
- **System or prototype:** build and systematically evaluate an LLM-based system, such as a retrieval-augmented pipeline or tool-using agent.
- **Dataset or evaluation study:** construct or adapt an evaluation setting and analyze effectiveness, robustness, efficiency, reliability, or failure modes.
- **Other formats are welcome** when the technical question, methodology, and evaluation are clearly defined.

Existing models, datasets, codebases, and APIs may be used when appropriate, provided that the group clearly documents what was reused and what was developed. Simply calling an existing model through an API, without a substantive technical or experimental contribution, is not sufficient, e.g., prompting a model on an existing benchmark and reporting accuracy versus the same, with a controlled ablation over prompt structure and a categorized error analysis.

## 4. Project Proposal Showcase (Sep 23)

All groups will **present their proposals** on the same day using a [shared Google Slides deck](https://docs.google.com/presentation/d/1Z_8SrKqXEEoPzU25bJw8OHz5bP3ZXpa0AT21fhbL6BE/edit?usp=sharing). Showcase proposal presentations should follow this structure:

- At most 3 slides (in addition to a group identification slide) / 5 minutes per group;
- The presentation should cover the problem and motivation, proposed approach or investigation, and planned evaluation.

**Groups may revise their plans** after the showcase based on feasibility and feedback.

## 5. Final Project Report (Nov 13)

Each group must submit a **final report** describing the outcomes of their project. The report should follow this structure:

- **At most 2 pages** in [ACM sigconf](https://www.acm.org/publications/proceedings-template) double-column format;
- References are excluded from the page limit;
- Figures and tables count toward the 2-page limit.

A **suggested report structure** is: problem and motivation; approach or system design; experimental setup; results; discussion, limitations, threats to validity, and conclusions. Groups should prioritize concise evidence and analysis over broad background material.

## 6. Final Project Presentations (Nov 16–25)

Groups will present in the same order as their group IDs, distributed across **four presentation dates** (Nov 16, 18, 23, 25), with three presentations per date. Presentations should follow this format:

- 20 minutes for the presentation;
- 10 minutes for questions and discussion;
- The presentation should focus on what was actually implemented or investigated, the evidence obtained, and the conclusions supported by the results.

## 7. Assessment

Projects will be assessed primarily on the **quality of the investigation** rather than on whether the final result is positive. Negative or inconclusive results may receive a strong grade when the study is well designed, carefully executed, and thoughtfully analyzed.

| Criterion | Weight | What we will look for |
|---|---|---|
| Problem definition | 15% | Clear objective, motivation, and appropriate scope, and most importantly, creativity and innovativeness. |
| Technical quality | 25% | Sound implementation, methodology, or system design. |
| Evaluation | 25% | Meaningful comparisons, measurements, or experiments aligned with the question. |
| Analysis | 20% | Interpretation of results, failure cases, limitations, and justified conclusions. |
| Communication | 15% | Clear, concise report and presentation; sufficient information for others to understand the work. |

## 8. Individual Contribution

The project will primarily receive a group grade. The final report should include a **brief contribution statement** describing each member's role. The instructors may request confidential peer feedback and may adjust individual grades when contributions are substantially uneven.

## 9. Use of AI Assistance

Large language models and AI coding assistants may be used in this project. This course studies these systems, and prohibiting their use would be inconsistent with its aims. The following conditions apply.

**Permitted without disclosure:**

> autocomplete, syntax help, debugging assistance, error-message interpretation, library lookup, and language editing of text the group has already written.

**Permitted with disclosure:**

> generation of non-trivial code (functions, modules, evaluation harnesses, plotting and analysis scripts); generation of prose subsequently revised by the group; assistance in designing experiments, selecting baselines, or interpreting results.

**Not permitted:**

> submitting generated text or code that no group member can explain; reporting results the group has not verified by execution; using an assistant to produce the analysis or conclusions that the report presents as the group's own.

**Disclosure format.**

> The report must include a brief AI-assistance statement (excluded from the page limit) naming the tools used, the components they contributed to, and the extent of that contribution. Two to four sentences are sufficient. Example: "GPT-5 and Claude Code were used for the data-loading and plotting code in src/eval/, and for language editing throughout. Experimental design, the evaluation harness in src/harness.py, and all analysis are the group's own."

<span style="color:red">**Groups are responsible for everything they submit, whether or not it was generated. During the final presentation, any member may be asked to explain any part of the code or report.**</span>

## 10. Code and Supporting Material Submission

Each group submits a single Git repository, hosted on GitHub or GitLab. Submit the repository URL together with the report, along with a commit hash identifying the state of the code at submission. Private repositories are acceptable; grant read access to the instructors' accounts before the deadline.

The repository must contain:

- README.md with a project summary, setup instructions, and the exact commands to reproduce the main results;
- a dependency specification (requirements.txt, environment.yml, pyproject.toml, or equivalent) with pinned versions;
- source code and configuration files;
- saved outputs, logs, or intermediate results sufficient to substantiate the reported numbers where full reproduction is costly.

**Size.** Keep the repository under 100 MB. Do not commit datasets, model weights, or large artifacts. Provide a download script or documented access instructions instead. Use Git LFS or an external host (institutional storage, Zenodo, Hugging Face) for anything larger, and document the location in the README.

**Credentials.** Do not commit API keys or other secrets. Document the required environment variables in the README and load them at runtime.

**Restricted data.** Groups working with data that cannot be redistributed should include access instructions and a small synthetic or public sample sufficient to exercise the pipeline.

## 11. Tentative Timeline

| Date | Milestone |
|---|---|
| September 23 | Project Proposal Showcase |
| October 23 | Two Pages Middle Report |
| November 13 | Final report and supporting material due |
| November 16, 18, 23, 25 | Final project presentations, in group-ID order |