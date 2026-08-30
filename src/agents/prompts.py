"""
The few-shot prompt for the baseline ReAct agent.

I wrote these exemplars myself rather than lifting the paper's appendix —
same format (Question / Thought / Action / Observation, same two actions),
different questions. Keeps the replication faithful to the *method*
without reproducing anyone else's text.
"""

INSTRUCTIONS = """Solve a question answering task with interleaving Thought, \
Action, Observation steps. Thought can reason about the current situation. \
Action can be one of two types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns \
the first few sentences if it exists, or suggests similar entities if it \
does not.
(2) Lookup[keyword], which returns the next sentence containing keyword in \
the current passage, like Ctrl+F on the page.
You may take as many steps as needed. When you know the answer, use:
Finish[answer]
"""

FEWSHOT_EXAMPLES = """
Question: What is the nationality of the director of the 1994 film that won the Palme d'Or at Cannes?
Thought: I need to find which film won the Palme d'Or in 1994, then find its director, then that director's nationality.
Action: Search[1994 Cannes Film Festival]
Observation: The 1994 Cannes Film Festival awarded the Palme d'Or to Pulp Fiction, directed by Quentin Tarantino.
Thought: The director is Quentin Tarantino. Now I need his nationality.
Action: Search[Quentin Tarantino]
Observation: Quentin Jerome Tarantino is an American film director, screenwriter, producer, and actor.
Thought: Quentin Tarantino is American.
Action: Finish[American]

Question: The university where the inventor of the World Wide Web earned his degree is located in which city?
Thought: I need to find who invented the World Wide Web, then where he earned his degree, then the city of that university.
Action: Search[World Wide Web]
Observation: The World Wide Web was invented by English computer scientist Tim Berners-Lee.
Thought: Now I need to find where Tim Berners-Lee earned his degree.
Action: Search[Tim Berners-Lee]
Observation: Tim Berners-Lee studied physics at The Queen's College, Oxford, graduating in 1976.
Thought: His degree is from Oxford. I need the city.
Action: Lookup[Oxford]
Observation: (Result 1/2) Tim Berners-Lee studied physics at The Queen's College, Oxford, graduating in 1976.
Thought: Oxford is both the college's name and the city. The answer is Oxford.
Action: Finish[Oxford]

Question: Which of the two rivers that the city of Khartoum sits at the confluence of is longer?
Thought: I need to find the two rivers meeting at Khartoum, then compare their lengths.
Action: Search[Khartoum]
Observation: Khartoum is the capital of Sudan, located at the confluence of the White Nile and the Blue Nile.
Thought: Now I need to compare the lengths of the White Nile and the Blue Nile.
Action: Search[White Nile]
Observation: The White Nile is one of the two main tributaries of the Nile, with the other being the Blue Nile.
Thought: I need a length figure. Let me look it up specifically.
Action: Lookup[length]
Observation: No mentions of [length] found.
Action: Search[Blue Nile]
Observation: The Blue Nile is approximately 1,400 km long, shorter than the White Nile's roughly 3,700 km.
Thought: The White Nile is longer.
Action: Finish[White Nile]
""".strip()


def build_prompt(question: str, trajectory: str) -> str:
    """
    trajectory is the accumulated Thought/Action/Observation text for this
    question so far (empty on the first hop). The model is expected to
    continue it with the next Thought/Action pair.
    """
    return (
        f"{INSTRUCTIONS}\n\n"
        f"Here are some examples.\n\n{FEWSHOT_EXAMPLES}\n\n"
        f"Now answer this question.\n\n"
        f"Question: {question}\n"
        f"{trajectory}"
    )
