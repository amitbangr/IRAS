from iras.llm_loader import get_llm
import re

# Get shared LLM instance
llm = get_llm()


def _extract_complete_questions(raw_text: str, num_questions: int):
    """Extract only complete questions ending with '?' from LLM output."""
    if not raw_text or not raw_text.strip():
        return []

    text = raw_text.replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text).strip()

    matches = re.findall(r"(?:^|\n)\s*(?:\d+[\.)]\s*)?(.+?\?)", text, flags=re.MULTILINE | re.DOTALL)

    cleaned = []
    for match in matches:
        question = " ".join(match.split())
        if question.endswith("?"):
            cleaned.append(question)
        if len(cleaned) >= num_questions:
            break

    return cleaned


def generate_interview_questions(skills: str, role: str, num_questions: int = 5, difficulty: str = "medium"):
    """
    Generate interview questions based on candidate skills and target role.

    Args:
        skills (str): Skills extracted from the resume
        role (str): Target job role
        num_questions (int): Number of questions to generate
        difficulty (str): Difficulty level of questions (easy, medium, hard)

    Returns:
        str: Generated interview questions
    """
    role_lower = role.lower().strip()

    difficulty = difficulty.lower().strip()

    difficulty_rules = {
        "easy": "Ask beginner-friendly questions focused on fundamentals, definitions, practical usage, and basic problem-solving. Avoid system design, advanced optimization, and complex theory.",
        "medium": "Ask industry-level questions requiring practical experience, implementation knowledge, and moderate problem-solving.",
        "hard": "Ask advanced questions involving architecture, optimization, tradeoffs, debugging, scalability, and deeper technical reasoning."
    }

    difficulty_instruction = difficulty_rules.get(
        difficulty,
        difficulty_rules["medium"]
    )

    role_topics = {
        "software engineer": (
            "data structures, algorithms, object-oriented programming, "
            "system design, APIs, databases, debugging, scalability, "
            "software development, backend systems"
        ),

        "full stack": (
            "React, Node.js, APIs, authentication, databases, "
            "deployment, frontend and backend integration"
        ),

        "frontend": (
            "React, JavaScript, CSS, UI design, state management, "
            "frontend performance"
        ),

        "backend": (
            "REST APIs, databases, authentication, caching, scalability, "
            "backend architecture"
        ),

        "data scientist": (
            "machine learning, feature engineering, model evaluation, "
            "Python, SQL, statistics, data analysis"
        ),

        "ai/ml": (
            "LLMs, transformers, RAG, deep learning, vector databases, "
            "AI deployment"
        ),

        "devops": (
            "CI/CD, Docker, Kubernetes, cloud deployment, monitoring, "
            "infrastructure automation"
        ),

        "cloud": (
            "AWS, Azure, GCP, cloud architecture, scalability, "
            "cloud security, deployment"
        ),

        "cybersecurity": (
            "network security, authentication, penetration testing, "
            "security vulnerabilities, encryption"
        )
    }

    selected_topics = "software engineering and problem solving"

    for key, topics in role_topics.items():
        if key in role_lower:
            selected_topics = topics
            break

    prompt = f"""
You are a professional technical interviewer.

Generate exactly {num_questions} real-time interview questions for this role:
{role}

Candidate Skills:
{skills}

Focus Topics:
{selected_topics}

Difficulty Level:
{difficulty}

Difficulty Instructions:
{difficulty_instruction}

Requirements:
- Questions must strongly match the target role.
- Keep questions practical, conversational, and medium difficulty.
- Avoid long case-study style questions.
- Avoid research-level or overly theoretical questions.
- Questions should sound like real company interviews.
- Keep each question under 25 words when possible.
- Every question must end with '?'
- Return only a numbered list.
- Do not include explanations, headings, labels, markdown, or extra text.
- Avoid repeating common interview questions.
- Generate diverse questions across different concepts.
- Do not always ask about overfitting.
- Avoid repeating the same topic multiple times.
- Follow the requested difficulty level strictly.
- For easy: ask fundamentals and resume-based questions.
- For medium: ask implementation and practical scenario questions.
- For hard: ask architecture, optimization, scalability, and advanced troubleshooting questions.

Examples:
- How do you optimize React application performance?
- How do you handle authentication in backend systems?
- How do you evaluate machine learning models?
- Describe a challenging bug you solved recently?
- How do you design scalable APIs?
"""

    try:
        questions = llm.generate(
            prompt,
            max_tokens=500,
            temperature=0.7,
        )

        parsed_questions = _extract_complete_questions(questions, num_questions)

        if not parsed_questions:
            return ""

        formatted_questions = [f"{i}. {q}" for i, q in enumerate(parsed_questions, start=1)]
        return "\n".join(formatted_questions)

    except Exception as e:
        print(f"❌ Error generating interview questions: {e}")
        return ""


if __name__ == "__main__":
    skills = "Python, Machine Learning, SQL, Data Analysis"
    role = "Data Scientist"

    print("\nGenerated Interview Questions:\n")
    print(generate_interview_questions(skills, role, num_questions=5, difficulty="easy"))
