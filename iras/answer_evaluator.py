from iras.llm_loader import get_llm

# Get shared LLM instance
llm = get_llm()


def evaluate_answer(question: str, answer: str, role: str):
    """
    Evaluate a candidate's interview answer using the LLM.

    Args:
        question (str): Interview question asked
        answer (str): Candidate's response
        role (str): Target job role

    Returns:
        str: Evaluation including score and feedback
    """

    prompt = f"""
You are an expert hiring panel consisting of a Senior Engineer, Hiring Manager, and Technical Lead.

ROLE:
{role}

QUESTION:
{question}

CANDIDATE ANSWER:
{answer}

First determine whether the answer is actually evaluable.

An answer is NOT evaluable if:
- It is random words.
- It is speech-to-text garbage.
- It is unrelated to the question.
- It contains only fillers such as 'yes', 'no', 'okay', 'hmm', 'maybe'.
- It is too vague to judge knowledge.
- It does not attempt to answer the question.

Your task is to evaluate the answer exactly like a real interview.

Evaluation Criteria:

1. Relevance (0-10)
- Did the candidate directly answer the question?
- Avoid giving credit for unrelated information.

2. Technical Accuracy (0-10)
- Are the concepts technically correct?
- Penalize incorrect statements and hallucinations.

3. Depth & Expertise (0-10)
- Does the answer show real understanding?
- Reward reasoning, trade-offs, architecture thinking, and technical depth.

4. Communication (0-10)
- Is the answer structured, concise, and professional?

5. Practical Application (0-10)
- Does the candidate connect theory to projects, work experience, systems, or real-world situations?


Overall Score Calculation Guidance:
- First evaluate Relevance, Technical Accuracy, Depth, Communication, and Practical Application.
- Then determine overall_score from those category scores.
- overall_score must be consistent with the category scores.
- Strong performance in Relevance and Technical Accuracy should significantly influence overall_score.

Semantic Evaluation Rules:
- Multiple correct answers may exist.
- Do NOT compare against a predefined answer.
- Evaluate semantic meaning, not keyword matching.
- Reward technically correct answers even if wording differs.
- If the answer demonstrates understanding and correctly answers the question, score it accordingly.

Scoring Calibration:
- Excellent answer with strong technical accuracy and depth: 8-10.
- Good answer with correct explanation and reasonable detail: 7-8.
- Average answer with basic correctness but limited depth: 5-6.
- Weak answer with partial understanding: 3-4.
- Incorrect, irrelevant, or nonsensical answer: 0-2.
- Use the full 0-10 range.
- Do not cluster scores around 5-6.
- A detailed, technically correct answer should not be capped at 6.

Strength Rules:
- Mention only genuine strengths.
- If no clear strengths exist, write:
  - None identified

Weakness Rules:
- Be specific.
- Mention missing concepts, missing examples, shallow explanation, incorrect information, or poor communication when applicable.

Improvement Rules:
- Give actionable interview advice.
- Focus on exactly what would improve the score.

Return ONLY valid JSON.

Format:
{{
  "evaluable": true,
  "reason_if_not_evaluable": "",
  "overall_score": 0,
  "relevance": 0,
  "technical_accuracy": 0,
  "depth": 0,
  "communication": 0,
  "practical_application": 0,
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "suggested_improvement": "specific improvement",
  "hiring_signal": "Strong Hire | Hire | Borderline | Not Recommended"
}}

Rules:
- If the answer is not evaluable, set:
  "evaluable": false
- Provide a clear explanation in:
  "reason_if_not_evaluable"
- For non-evaluable answers, all scores should be 0.
- Examples of non-evaluable answers: 'very very cure', 'yes', 'okay', 'testing microphone', random words, or unrelated text.
- Every score must be an integer from 0 to 10.
- Never return NA, N/A, null, or empty values.
- If information is insufficient, assign a low score instead.
- Return JSON only.

Return valid JSON only.
"""

    try:
        evaluation = llm.generate(
            prompt,
            max_tokens=350,
            temperature=0.1,
        )

        if not evaluation or not evaluation.strip():
            return '{"evaluable": false, "reason_if_not_evaluable": "Empty model response", "overall_score": 0, "relevance": 0, "technical_accuracy": 0, "depth": 0, "communication": 0, "practical_application": 0, "strengths": [], "weaknesses": ["Empty model response"], "suggested_improvement": "Retry evaluation.", "hiring_signal": "Not Recommended"}'

        import re
        import json

        response_text = evaluation.strip()

        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

        if json_match:
            candidate_json = json_match.group(0)

            try:
                json.loads(candidate_json)
                return candidate_json
            except Exception:
                pass

        score_match = re.search(r'score\s*:?\s*(\d+)(?:/10)?', response_text, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 0

        fallback = {
            "evaluable": score > 0,
            "reason_if_not_evaluable": "Model did not return JSON.",
            "overall_score": score,
            "relevance": score,
            "technical_accuracy": score,
            "depth": score,
            "communication": score,
            "practical_application": score,
            "strengths": ["Parsed from legacy evaluator output"],
            "weaknesses": ["Evaluator did not return JSON"],
            "suggested_improvement": "Update evaluator prompt or model.",
            "hiring_signal": "Borderline" if score >= 5 else "Not Recommended"
        }

        return json.dumps(fallback)

    except Exception as e:
        print(f"❌ Error evaluating answer: {e}")
        return '{"evaluable": false, "reason_if_not_evaluable": "Evaluator exception", "overall_score": 0, "relevance": 0, "technical_accuracy": 0, "depth": 0, "communication": 0, "practical_application": 0, "strengths": [], "weaknesses": ["Evaluator exception"], "suggested_improvement": "Check logs.", "hiring_signal": "Not Recommended"}'


if __name__ == "__main__":
    question = "Explain the concept of object-oriented programming."
    answer = "OOP is a programming paradigm that uses classes and objects."
    role = "Software Engineer"

    print("\nEvaluation Result:\n")
    print(evaluate_answer(question, answer, role))
