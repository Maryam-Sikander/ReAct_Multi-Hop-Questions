# ReAct Multi-Hop Reasoning with Explicit Judgment

This project investigates whether adding an explicit **judgment mechanism** to the ReAct framework can improve multi-hop question answering.

The project is based on the ReAct framework, which interleaves reasoning with actions and observations to solve knowledge-intensive tasks. The main experimental objective is to compare standard ReAct against a modified ReAct framework that introduces an additional judgment step for evaluating retrieved evidence and the current reasoning trajectory.

The experiments are conducted on the **HotpotQA** multi-hop question answering benchmark using locally hosted language models.

---

## Overview

The project evaluates two reasoning approaches:

- **Standard ReAct**: The model alternates between Thought, Action, and Observation steps until it produces a final answer.
- **ReAct + Judgment**: An additional language model evaluates the retrieved evidence and reasoning trajectory before the agent continues.

The primary research question is:

> Does adding an explicit judgment step to ReAct improve multi-hop question answering compared with standard ReAct?
