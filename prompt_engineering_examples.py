"""
Prompt engineering examples for classroom teaching.

This file demonstrates multiple prompt techniques using the same
Hugging Face model so students can compare the prompt patterns directly.
"""

from llm_utils import ask_llm, load_llm


def print_example(title: str, prompt: str, answer: str) -> None:
    """Pretty-print one prompt engineering example."""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    print("\nPROMPT:\n")
    print(prompt)
    print("\nMODEL OUTPUT:\n")
    print(answer)


def run_prompt_engineering_demo() -> None:
    """
    Run a sequence of prompt engineering examples.

    Each example uses a different prompting technique so students can see
    how better prompts usually lead to better outputs.
    """
    generator = load_llm()

    # 1. Zero-shot prompting:
    # We ask the model to perform a task without giving any examples.
    zero_shot_prompt = """
    Explain photosynthesis for a 10-year-old in exactly 2 short sentences.
    Use the words sunlight, water, and food.
    Do not add a title or introduction.
    """.strip()
    zero_shot_answer = ask_llm(generator, zero_shot_prompt, max_new_tokens=60)
    print_example("1. Zero-shot Prompting", zero_shot_prompt, zero_shot_answer)

    # 2. One-shot prompting:
    # We provide exactly one example so the model learns the pattern.
    one_shot_prompt = """
    Rewrite each input as a polite email request.
    Return only the rewritten sentence.

    Example:
    Input: Send me the report today.
    Output: Could you please send me the report today?

    Input: Give me the meeting notes.
    Output:
    """.strip()
    one_shot_answer = ask_llm(generator, one_shot_prompt, max_new_tokens=40)
    print_example("2. One-shot Prompting", one_shot_prompt, one_shot_answer)

    # 3. Few-shot prompting:
    # We provide multiple examples so the model understands the pattern
    # more clearly than one-shot prompting.
    few_shot_prompt = """
    Classify the sentiment as Positive, Negative, or Neutral.
    Answer with one word only.

    Review: I loved the product and the delivery was fast.
    Sentiment: Positive

    Review: The item broke on the first day. Very disappointing.
    Sentiment: Negative

    Review: The package arrived yesterday. I have not used it yet.
    Sentiment: Neutral

    Review: The laptop works smoothly and the battery life is excellent.
    Sentiment:
    """.strip()
    few_shot_answer = ask_llm(generator, few_shot_prompt)
    print_example("3. Few-shot Prompting", few_shot_prompt, few_shot_answer)

    # 4. Role prompting:
    # Use a narrow tone-rewrite task so the role change is easy to see.
    role_prompt = """
    You are a polite customer support agent.
    Rewrite the sentence in a calm, helpful tone.
    Return only one sentence.

    Sentence: Your password is wrong. Try again.
    """.strip()
    role_answer = ask_llm(generator, role_prompt, max_new_tokens=40)
    print_example("4. Role Prompting", role_prompt, role_answer)

    # 5. Step-by-step prompting:
    # Demonstrate a stepwise answer with a simple everyday procedure.
    cot_prompt = """
    Explain the process in exactly 3 numbered steps.
    Keep each step short and clear.
    Return only the 3 steps and nothing else.

    Task: How do you make a cup of tea?
    """.strip()
    cot_answer = ask_llm(generator, cot_prompt, max_new_tokens=50)
    print_example("5. Step-by-step Prompting", cot_prompt, cot_answer)

    # 6. Structured output prompting:
    # We explicitly ask the model for a fixed format.
    structured_prompt = """
    Fill in this exact format using short phrases only:
    Definition:
    Real-world use:
    One challenge:

    Topic: Machine learning
    """.strip()
    structured_answer = ask_llm(generator, structured_prompt, max_new_tokens=60)
    print_example("6. Structured Output Prompting", structured_prompt, structured_answer)
