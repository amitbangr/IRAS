# controller.py

import time


state = {
    "is_speaking": False,
    "cooldown": False
}

def clean_answer(answer, question):
    if not answer:
        return answer

    answer_lower = answer.lower()
    question_lower = question.lower()

    # remove full question if repeated
    if question_lower in answer_lower:
        answer = answer_lower.replace(question_lower, "").strip()

    # remove tail words
    q_words = question_lower.split()
    tail = " ".join(q_words[-6:])

    if tail in answer.lower():
        answer = answer.lower().replace(tail, "").strip()

    return answer

def run_interview(generate_questions, speak, listen, evaluate):
    """
    Controls the interview flow.
    Keeps things simple: ask → listen → evaluate
    """

    questions = generate_questions()

    print("\nStarting Interview...\n")

    results = []

    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")

        # 🔊 SPEAK
        state["is_speaking"] = True
        speak(question)
        state["is_speaking"] = False

        # 🧠 COOLDOWN (important to avoid TTS echo)
        state["cooldown"] = True
        time.sleep(1.5)
        state["cooldown"] = False

        # Capture answer
        print("Listening...")
        answer = listen()
        answer = clean_answer(answer, question)

        if not answer:
            answer = ""

        print(f"Answer: {answer}")

        # Evaluate answer
        evaluation = evaluate(question, answer)

        print(f"Evaluation:\n{evaluation}")

        results.append({
            "question": question,
            "answer": answer,
            "evaluation": evaluation
        })

    print("\nInterview Completed.\n")

    return results
