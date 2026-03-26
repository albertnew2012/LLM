# instruction_data.py
# Small instruction dataset with a few paraphrases per concept.

BASE_INSTRUCTIONS = [
    {
        "questions": [
            "What is youth?",
            "What does youth mean?",
            "Can you explain youth?",
            "Is youth just a young age?",
        ],
        "answer": "No. Youth is not just a young age. It is a state of mind, a time of energy, curiosity, and boundless possibility that can be carried forward.",
    },
    {
        "questions": [
            "What is wisdom?",
            "What does wisdom mean?",
            "Can you explain wisdom?",
            "How would you describe wisdom?",
        ],
        "answer": "Wisdom is the ability to make good judgments based on knowledge and experience. It grows through reflection, humility, and the lessons that time teaches us.",
    },
    {
        "questions": [
            "What is courage?",
            "What does courage mean?",
            "Can you explain courage?",
            "How would you describe courage?",
        ],
        "answer": "Courage is the strength to face fear, pain, or uncertainty without retreating. It is not the absence of fear, but the decision to act despite it.",
    },
    {
        "questions": [
            "What is friendship?",
            "What does friendship mean?",
            "Can you explain friendship?",
            "How would you describe friendship?",
        ],
        "answer": "Friendship is a bond built on trust, shared experiences, and mutual care. A true friend remains present through both joy and hardship.",
    },
    {
        "questions": [
            "What is happiness?",
            "What does happiness mean?",
            "Can you explain happiness?",
            "How would you describe happiness?",
        ],
        "answer": "Happiness is a sense of contentment and joy that grows from within. It is nourished by purpose, connection, gratitude, and peace of mind.",
    },
    {
        "questions": [
            "What is time?",
            "What does time mean?",
            "Can you explain time?",
            "How would you describe time?",
        ],
        "answer": "Time is the continuous progression of existence from past through present into future. It is finite, which is why it is one of our most precious resources.",
    },
    {
        "questions": [
            "What is knowledge?",
            "What does knowledge mean?",
            "Can you explain knowledge?",
            "How would you describe knowledge?",
        ],
        "answer": "Knowledge is the understanding we gain through study, experience, and inquiry. It helps us solve problems and make sense of the world.",
    },
    {
        "questions": [
            "What is love?",
            "What does love mean?",
            "Can you explain love?",
            "How would you describe love?",
        ],
        "answer": "Love is a deep affection and care for another. It takes many forms, including love between friends, family members, and partners, and it is one of the strongest human experiences.",
    },
    {
        "questions": [
            "What is failure?",
            "What does failure mean?",
            "Can you explain failure?",
            "How would you describe failure?",
        ],
        "answer": "Failure is the result of an attempt that does not succeed. It is not only an ending, but also a teacher that shows where growth and improvement are needed.",
    },
    {
        "questions": [
            "What is success?",
            "What does success mean?",
            "Can you explain success?",
            "How would you describe success?",
        ],
        "answer": "Success is the achievement of a goal or purpose. It is defined differently by each person and is most meaningful when it aligns with personal values.",
    },
]


INSTRUCTION_DATA = [
    {"question": question, "answer": item["answer"]}
    for item in BASE_INSTRUCTIONS
    for question in item["questions"]
]


def format_sample(q, a):
    """Format a Q&A pair into a single training string."""
    return f"Question: {q}\nAnswer: {a}"
