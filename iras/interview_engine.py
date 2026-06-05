from iras.question_generator import generate_interview_questions
from iras.answer_evaluator import evaluate_answer
from iras.text_to_speech import TextToSpeech
from iras.speech_to_text import SpeechToText
import concurrent.futures
import re
import threading
import time


class InterviewEngine:
    """
    Core IRAS engine that manages the full AI interview process.

    Flow:
    AI generates question → speaks question → candidate answers via microphone
    → speech converted to text → LLM evaluates answer → next question.
    """

    def __init__(self, skills: str, role: str, num_questions: int = 5):
        self.skills = skills
        self.role = role
        self.num_questions = num_questions

        self.questions = []
        self.answers = []
        self.evaluations = []
        self.stop_interview = False
        self.max_question_echo_words = 8
        self.answer_record_seconds = 12
        self.single_question_mode = False
        self.tts_lock = threading.Lock()
        self.is_tts_speaking = False
        self.tts_echo_window_seconds = 1.2
        self.last_tts_end_time = 0.0
        self.tts_question_active = False
        self.current_tts_question_text = ""
        self.tts_started_at = 0.0
        self.tts_finished_event = threading.Event()
        self.tts_finished_event.set()
        self.last_question_spoken = ""
        self.last_question_printed = ""

        self.tts_echo_prefix_match_limit = 12
        self.tts_min_valid_answer_words = 3
        self.skip_tts_playback = False
        self.wait_for_tts_completion = True
        self.min_answer_word_count = 2
        self.min_answer_char_count = 8
        self.tts_words_per_second = 2.6
        self.tts_lead_buffer_seconds = 0.2
        self.tts_tail_buffer_seconds = 0.45
        self.tts_max_wait_seconds = 20.0
        self.post_tts_silence_seconds = 2.0  # increased buffer for real speaker drain
        self.enable_stt_echo_guard = True
        self.tts_guard_phrase = "[INTERVIEWER_TTS_ACTIVE]"

        # Voice modules
        self.tts = TextToSpeech()
        self.stt = SpeechToText()

        self.stt_supports_ignore_input = False
        record_audio_method = getattr(self.stt, "record_audio", None)
        if callable(record_audio_method):
            try:
                import inspect
                self.stt_supports_ignore_input = "ignore_input_while" in inspect.signature(record_audio_method).parameters
            except Exception:
                self.stt_supports_ignore_input = False

    def _fallback_questions(self):
        return [
            "Tell me about yourself.",
            f"Explain your experience related to {self.role}.",
            "What are your strongest technical skills?",
            "Describe a challenging problem you solved.",
            "Why should we hire you for this role?"
        ]

    def _should_ignore_mic_input(self):
        # block mic while TTS is active
        if self.wait_for_tts_completion and (self.is_tts_speaking or self.tts_question_active):
            return True

        # block mic shortly after TTS ends (speaker bleed)
        if time.time() - self.last_tts_end_time < (self.tts_echo_window_seconds + self.post_tts_silence_seconds):
            return True

        return False

    def _estimate_tts_duration(self, text: str) -> float:
        words = re.findall(r"\w+", text or "")
        if not words:
            return 1.0

        estimated = len(words) / max(self.tts_words_per_second, 0.1)
        estimated += self.tts_lead_buffer_seconds + self.tts_tail_buffer_seconds
        return max(1.0, min(self.tts_max_wait_seconds, estimated))

    def _listen_for_stop(self):
        """Listen for hotkey input to stop the interview."""
        print("\n⌨️ Press 's' and hit Enter at any time to stop the interview after the current question.\n")
        while not self.stop_interview:
            try:
                user_input = input().strip().lower()
                if user_input == "s":
                    self.stop_interview = True
                    print("\n🛑 Stop signal received. Interview will end after the current question.\n")
                    break
            except EOFError:
                break
            except Exception:
                break

    def _parse_questions(self, questions_text: str):
        parsed_questions = []
        current_question = ""

        for line in questions_text.split("\n"):
            clean_line = line.strip()
            if not clean_line:
                continue

            if re.match(r"^\d+[\.)-]?\s+", clean_line) or re.match(r"^[-*]\s+", clean_line):
                if current_question:
                    parsed_questions.append(current_question.strip())
                clean_line = re.sub(r"^\d+[\.)-]?\s*", "", clean_line)
                clean_line = re.sub(r"^[-*]\s*", "", clean_line)
                current_question = clean_line
            else:
                if current_question:
                    current_question += " " + clean_line
                else:
                    current_question = clean_line

        if current_question:
            parsed_questions.append(current_question.strip())

        return parsed_questions

    def generate_questions(self):
        """Generate interview questions using the LLM."""

        print("\n🧠 Generating interview questions using LLM...\n")

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                filtered_skills = filter_skills_by_role(
                    self.skills,
                    self.role
                )

                print(f"[IRAS ENGINE] Target Role: {self.role}")
                print(f"[IRAS ENGINE] Original Skills: {self.skills}")
                print(f"[IRAS ENGINE] Filtered Skills: {filtered_skills}")

                future = executor.submit(
                    generate_interview_questions,
                    filtered_skills,
                    self.role,
                    self.num_questions
                )
                questions_text = future.result(timeout=20)
        except Exception:
            print("⚠️ Question generation took too long. Using fallback questions.")
            questions_text = None

        if not questions_text:
            print("⚠️ LLM returned empty response. Using fallback questions.")
            self.questions = self._fallback_questions()
            return self.questions

        self.questions = self._parse_questions(questions_text)[:self.num_questions]

        if not self.questions:
            print("⚠️ No questions parsed from LLM output. Using fallback set.")
            self.questions = self._fallback_questions()

        print("✅ Questions generated:\n")
        for i, q in enumerate(self.questions, start=1):
            print(f"{i}. {q}")

        return self.questions

    def _ask_and_capture_answer(self, question: str):
        self.last_question_spoken = question

        print(f"\n🔊 AI Interviewer: {question}")
        spoken_question = question.strip()
        expected_tts_duration = self._estimate_tts_duration(spoken_question)

        if self.skip_tts_playback:
            print("⏭️ TTS playback skipped to avoid speaker-to-mic echo.")
            self.tts_started_at = time.time()
            self.tts_finished_event.set()
            self.last_tts_end_time = time.time()
            self.tts_question_active = False
            self.is_tts_speaking = False
            self.current_tts_question_text = ""
            print("✅ Interviewer finished speaking.")
            time.sleep(self.post_tts_silence_seconds)
        else:
            # 🔥 STRICT BLOCKING TTS (no threading, no overlap)
            with self.tts_lock:
                self.tts_started_at = time.time()
                self.is_tts_speaking = True
                self.tts_question_active = True
                self.current_tts_question_text = question

                print(f"⏱️ Estimated TTS duration: {expected_tts_duration:.2f}s")

                try:
                    self.tts.speak(question)  # BLOCKING CALL
                finally:
                    self.last_tts_end_time = time.time()
                    self.is_tts_speaking = False
                    self.tts_question_active = False
                    self.current_tts_question_text = ""

            print("✅ Interviewer finished speaking.")

            # HARD GUARANTEED BUFFER AFTER SPEAK
            time.sleep(self.post_tts_silence_seconds)

        print("🎤 Recording candidate answer...")
        # hard wait to ensure NO speaker audio is captured
        print("⏳ Waiting for complete silence before recording...")
        time.sleep(self.post_tts_silence_seconds)
        print(f"🎤 Recording for up to {self.answer_record_seconds} seconds using the STT module recorder...")

        if self.wait_for_tts_completion:
            while self._should_ignore_mic_input():
                time.sleep(0.05)
        else:
            dynamic_pause_until = self.last_tts_end_time + min(expected_tts_duration * 0.15, 0.8)
            while self._should_ignore_mic_input() or time.time() < dynamic_pause_until:
                time.sleep(0.05)

        if self.enable_stt_echo_guard and hasattr(self.stt, "set_echo_guard"):
            try:
                self.stt.set_echo_guard(
                    active=True,
                    question_text=spoken_question,
                    estimated_tts_duration=expected_tts_duration,
                    tts_end_time=self.last_tts_end_time,
                    guard_phrase=self.tts_guard_phrase,
                )
            except Exception:
                pass

        # final safety buffer before mic starts
        print("🔇 Final silence buffer...")
        time.sleep(0.8)
        if self.stt_supports_ignore_input:
            audio_path = self.stt.record_audio(
                duration=self.answer_record_seconds,
                ignore_input_while=self._should_ignore_mic_input,
            )
        else:
            audio_path = self.stt.record_audio(duration=self.answer_record_seconds)

        if self.enable_stt_echo_guard and hasattr(self.stt, "set_echo_guard"):
            try:
                self.stt.set_echo_guard(active=False)
            except Exception:
                pass

        if not audio_path:
            print("⚠️ Recording failed or stopped early.")
            return "No valid answer provided by candidate."

        print("🧠 Transcribing answer...")
        if self.enable_stt_echo_guard and hasattr(self.stt, "set_echo_guard"):
            try:
                self.stt.set_echo_guard(
                    active=True,
                    question_text=spoken_question,
                    estimated_tts_duration=expected_tts_duration,
                    tts_end_time=self.last_tts_end_time,
                    guard_phrase=self.tts_guard_phrase,
                )
            except Exception:
                pass

        answer_text = self.stt.transcribe(audio_path)

        if self.enable_stt_echo_guard and hasattr(self.stt, "set_echo_guard"):
            try:
                self.stt.set_echo_guard(active=False)
            except Exception:
                pass

        cleaned_answer = (answer_text or "").strip()
        # 🔇 Remove internal system/log phrases accidentally captured by STT
        noise_phrases = [
            "waiting for complete silence before recording",
            "recording for up to",
            "final silence buffer",
            "recording for",
            "speak clearly and keep the mic close",
        ]

        for phrase in noise_phrases:
            cleaned_answer = re.sub(
                re.escape(phrase),
                " ",
                cleaned_answer,
                flags=re.IGNORECASE
            ).strip()
        if self.tts_guard_phrase in cleaned_answer:
            cleaned_answer = cleaned_answer.replace(self.tts_guard_phrase, " ").strip()
        active_question_for_echo = (self.last_question_spoken or spoken_question).lower().strip()

        if not cleaned_answer:
            print("⚠️ No valid candidate answer detected.")
            return "No valid answer provided by candidate."

        normalized_question_words = re.findall(r"\w+", active_question_for_echo)
        normalized_answer_words = re.findall(r"\w+", cleaned_answer.lower())

        if normalized_question_words and normalized_answer_words:
            longest_prefix = 0
            max_prefix = min(
                len(normalized_question_words),
                len(normalized_answer_words),
                self.tts_echo_prefix_match_limit,
            )

            for i in range(max_prefix):
                if normalized_question_words[i] == normalized_answer_words[i]:
                    longest_prefix += 1
                else:
                    break

            # 🔥 stronger removal: if even 2+ words match, cut aggressively
            if longest_prefix >= 2:
                answer_word_spans = list(re.finditer(r"\w+", cleaned_answer))
                if longest_prefix < len(answer_word_spans):
                    cut_index = answer_word_spans[longest_prefix].start()
                    cleaned_answer = cleaned_answer[cut_index:].lstrip(" .,!?:;\"'\n\t")
                    print(f"⚠️ Aggressively removed {longest_prefix} echoed words from transcript.")
                else:
                    cleaned_answer = ""

        # 🔥 final hard filter: remove leading fragment if it still resembles question
        question_prefix_text = " ".join(normalized_question_words[:8])
        if cleaned_answer.lower().startswith(question_prefix_text[:20]):
            print("⚠️ Detected leftover question fragment at start. Removing it completely.")
            cleaned_answer = ""

        # 🔥 ultra-strong fix: remove first sentence if it overlaps heavily with question
        sentences = re.split(r'[.!?]+', cleaned_answer)
        if sentences:
            first_sentence = sentences[0].strip().lower()
            question_words = set(normalized_question_words)
            first_words = re.findall(r"\w+", first_sentence)

            if first_words:
                overlap = sum(1 for w in first_words if w in question_words)
                overlap_ratio = overlap / max(1, len(first_words))

                # if first sentence is mostly echo → remove it entirely
                if overlap_ratio >= 0.4:
                    print("⚠️ Removing echoed first sentence completely.")
                    cleaned_answer = ". ".join(sentences[1:]).strip()

        cleaned_answer = cleaned_answer.strip()
        if not cleaned_answer:
            print("⚠️ No valid candidate answer detected after echo filtering.")
            return "No valid answer provided by candidate."

        if len(cleaned_answer) < self.min_answer_char_count:
            print("⚠️ Transcript too short after echo filtering. Ignoring it.")
            return "No valid answer provided by candidate."

        remaining_answer_words = re.findall(r"\w+", cleaned_answer.lower())
        if normalized_question_words and remaining_answer_words:
            question_word_set = set(normalized_question_words)
            overlap_count = sum(1 for word in remaining_answer_words if word in question_word_set)
            overlap_ratio = overlap_count / max(1, len(remaining_answer_words))

            if (
                len(remaining_answer_words) <= max(self.tts_min_valid_answer_words, len(normalized_question_words) // 2)
                and overlap_ratio >= 0.6
            ):
                print("⚠️ Remaining transcript still looks like interviewer echo. Ignoring it.")
                return "No valid answer provided by candidate."

            if len(remaining_answer_words) < self.min_answer_word_count:
                print("⚠️ Transcript has too few words after echo filtering. Ignoring it.")
                return "No valid answer provided by candidate."

        if normalized_question_words and remaining_answer_words:
            question_prefix_text = " ".join(normalized_question_words[: self.tts_echo_prefix_match_limit])
            remaining_answer_text = " ".join(remaining_answer_words)

            if question_prefix_text and question_prefix_text in remaining_answer_text:
                print("⚠️ Transcript still contains a long interviewer-question fragment. Ignoring it.")
                return "No valid answer provided by candidate."

        return cleaned_answer

    def conduct_interview(self):
        """Run the full AI voice interview until user stops it."""

        print("\nStarting AI Interview...\n")

        if not self.questions:
            self.generate_questions()

        stop_listener = threading.Thread(target=self._listen_for_stop, daemon=True)
        stop_listener.start()

        question_index = 0

        while not self.stop_interview:
            if question_index < len(self.questions):
                question = self.questions[question_index]
            else:
                print("\n🧠 Generating next interview question dynamically...\n")
                filtered_skills = filter_skills_by_role(
                    self.skills,
                    self.role
                )

                generated = generate_interview_questions(
                    filtered_skills,
                    self.role,
                    num_questions=1
                )
                if not generated or not generated.strip():
                    generated = f"1. Can you explain more about your experience related to {self.role}?"

                parsed_questions = self._parse_questions(generated)
                if parsed_questions:
                    question = parsed_questions[0]
                else:
                    question = f"Can you explain more about your experience related to {self.role}?"

                self.questions.append(question)

            print(f"\nQuestion {question_index + 1}: {question}\n")
            self.last_question_printed = question
            answer_text = self._ask_and_capture_answer(question)
            print(f"Candidate Answer: {answer_text}\n")

            print("🧠 Evaluating answer using LLM...")

            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        evaluate_answer,
                        question,
                        answer_text,
                        self.role
                    )
                    evaluation = future.result(timeout=20)
            except Exception as e:
                print(f"⚠️ Evaluation failed: {e}")
                evaluation = evaluate_answer(
                    question,
                    answer_text,
                    self.role
                )

            if not evaluation or evaluation == [] or (isinstance(evaluation, str) and not evaluation.strip()):
                raise ValueError("LLM returned empty evaluation response")

            print("Evaluation:")
            print(evaluation)

            self.answers.append(answer_text)
            self.evaluations.append(evaluation)

            question_index += 1
            if self.single_question_mode:
                self.stop_interview = True
                break

        return self.generate_final_report()

    def generate_final_report(self):
        """Compile final interview results."""

        report = {
            "role": self.role,
            "skills": self.skills,
            "questions": self.questions,
            "answers": self.answers,
            "evaluations": self.evaluations
        }

        return report


# --- LLM ONLY FUNCTIONS (used by controller) ---

def filter_skills_by_role(skills, role):
    """Filter resume skills so unrelated domains do not contaminate question generation."""

    if not skills:
        return ""

    role_lower = str(role).lower().strip()

    role_skill_map = {
        "full stack": [
            "react", "next", "node", "express", "javascript",
            "typescript", "mongodb", "sql", "api", "frontend",
            "backend", "authentication", "css", "html"
        ],

        "frontend": [
            "react", "javascript", "typescript", "css",
            "html", "redux", "frontend", "next"
        ],

        "backend": [
            "node", "express", "api", "sql", "mongodb",
            "authentication", "backend", "database", "server"
        ],

        "data scientist": [
            "python", "machine learning", "pandas", "numpy",
            "statistics", "sql", "data analysis", "tensorflow",
            "scikit", "feature engineering"
        ],

        "ai/ml": [
            "llm", "transformers", "deep learning", "pytorch",
            "tensorflow", "rag", "vector database", "ml"
        ]
    }

    allowed_keywords = []

    for key, values in role_skill_map.items():
        if key in role_lower:
            allowed_keywords = values
            break

    if not allowed_keywords:
        return skills

    split_skills = [s.strip() for s in str(skills).split(",")]

    filtered = []

    for skill in split_skills:

        lower_skill = skill.lower()

        if any(keyword in lower_skill for keyword in allowed_keywords):
            filtered.append(skill)

    return ", ".join(filtered)

def generate_questions(
    skills="Python, Machine Learning, Data Analysis",
    role="Data Scientist",
    num_questions=5,
    difficulty="medium"
):
    try:
        filtered_skills = filter_skills_by_role(skills, role)

        print(f"[IRAS ROLE FILTER] Target Role: {role}")
        print(f"[IRAS ROLE FILTER] Original Skills: {skills}")
        print(f"[IRAS ROLE FILTER] Filtered Skills: {filtered_skills}")

        questions_text = generate_interview_questions(
            filtered_skills,
            role,
            num_questions,
            difficulty=difficulty
        )
    except Exception:
        return [
            "Tell me about yourself.",
            f"Explain your experience related to {role}.",
            "What are your strongest technical skills?",
            "Describe a challenging problem you solved.",
            "Why should we hire you?"
        ]

    if not questions_text:
        return []

    questions = []
    for line in questions_text.split("\n"):
        line = line.strip()
        if line:
            line = re.sub(r"^\d+[\.)-]?\s*", "", line)
            questions.append(line)

    return questions[:num_questions]


def evaluate(question, answer, role="Data Scientist"):
    return evaluate_answer(question, answer, role)
