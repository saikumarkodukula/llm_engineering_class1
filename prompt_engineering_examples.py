"""
Prompt engineering examples for classroom teaching.

This file demonstrates multiple prompt techniques using the same
model backend so students can compare the prompt patterns directly.
"""

from llm_utils import ask_llm, load_llm


def print_example(title: str, prompt: str, answer: str) -> None:
    """
    Print one prompt engineering example in a classroom-friendly format.

    Parameters:
    - title: The title shown above the example.
    - prompt: The exact prompt that was sent to the model.
    - answer: The model output returned for that prompt.

    Returns:
    - None.
    """
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

    # 1. Zero-shot prompting asks for a direct answer with no examples.
    zero_shot_prompt = """
    Explain calorie deficit for a 12-year-old in exactly 2 short sentences.
    Use the words food, energy, and body.
    Do not add a title or introduction.
    """.strip()
    zero_shot_answer = ask_llm(generator, zero_shot_prompt, max_new_tokens=60)
    print_example("1. Zero-shot Prompting", zero_shot_prompt, zero_shot_answer)

    # 2. One-shot prompting provides one example of the target pattern.
    one_shot_prompt = """
    Rewrite each input as supportive health coaching advice.
    Return only the rewritten sentence.

    Example:
    Input: Stop eating junk food.
    Output: Try replacing some processed snacks with filling foods such as fruit, yogurt, or nuts.

    Input: Skip dinner if you want to lose weight faster.
    Output:
    """.strip()
    one_shot_answer = ask_llm(generator, one_shot_prompt, max_new_tokens=70)
    print_example("2. One-shot Prompting", one_shot_prompt, one_shot_answer)

    # 3. Few-shot prompting gives multiple labeled examples.
    few_shot_prompt = """
    Classify the statement as Helpful, Risky, or Neutral for weight-loss guidance.
    Answer with one word only.

    Statement: Walk regularly and build habits you can maintain.
    Label: Helpful

    Statement: Use an extreme crash diet without medical advice.
    Label: Risky

    Statement: This guide discusses obesity treatment pathways.
    Label: Neutral

    Statement: Combine a balanced diet with physical activity for gradual progress.
    Label:
    """.strip()
    few_shot_answer = ask_llm(generator, few_shot_prompt)
    print_example("3. Few-shot Prompting", few_shot_prompt, few_shot_answer)

    # 4. Role prompting changes the model's tone and framing.
    role_prompt = """
    You are a registered dietitian teaching a beginner.
    Rewrite the sentence in a calm, practical tone.
    Return only one sentence.

    Sentence: You failed your diet because you ate dessert.
    """.strip()
    role_answer = ask_llm(generator, role_prompt, max_new_tokens=60)
    print_example("4. Role Prompting", role_prompt, role_answer)

    # 5. Step-by-step prompting requests a compact ordered process.
    cot_prompt = """
    Explain the process in exactly 3 numbered steps.
    Keep each step short and clear.
    Return only the 3 steps and nothing else.

    Task: How do you start a simple weekly weight-loss routine?
    """.strip()
    cot_answer = ask_llm(generator, cot_prompt, max_new_tokens=70)
    print_example("5. Step-by-step Prompting", cot_prompt, cot_answer)

    # 6. Structured prompting asks for a fixed output shape.
    structured_prompt = """
    Fill in this exact format using short phrases only:
    Definition:
    Real-world use:
    One challenge:

    Topic: Multicomponent weight management
    """.strip()
    structured_answer = ask_llm(generator, structured_prompt, max_new_tokens=60)
    print_example("6. Structured Output Prompting", structured_prompt, structured_answer)
