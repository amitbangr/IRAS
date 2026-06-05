import os

# Create IRAS directory structure for interview system

def create_iras_structure(base_path):
    print("\n📦 Creating IRAS directory structure...\n")

    iras_path = os.path.join(base_path, "iras")

    files = [
        "llm_loader.py",
        "interview_engine.py",
        "question_generator.py",
        "answer_evaluator.py",
        "speech_to_text.py",
        "text_to_speech.py",
        "__init__.py"
    ]

    os.makedirs(iras_path, exist_ok=True)

    for f in files:
        file_path = os.path.join(iras_path, f)

        if not os.path.exists(file_path):
            with open(file_path, "w") as file:
                file.write("# IRAS module file\n")
            print(f"✅ Created: {file_path}")
        else:
            print(f"⚠ Already exists: {file_path}")

# Print project directory tree

def print_tree(start_path, max_depth=3):
    for root, dirs, files in os.walk(start_path):
        depth = root.replace(start_path, "").count(os.sep)
        if depth > max_depth:
            continue

        indent = "│   " * depth
        print(f"{indent}📂 {os.path.basename(root) if os.path.basename(root) else root}")

        subindent = "│   " * (depth + 1)
        for f in files:
            print(f"{subindent}📄 {f}")


# Search for possible model files

def find_models(start_path):
    print("\n🔎 Searching for model files...\n")

    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file.endswith((".gguf", ".bin", ".pt", ".pth", ".safetensors")):
                print(f"🧠 Model found: {os.path.join(root, file)}")


if __name__ == "__main__":
    project_path = os.getcwd()

    print("\n📁 PROJECT STRUCTURE\n")
    print_tree(project_path)

    find_models(project_path)

    create_iras_structure(project_path)

    print("\n✅ Scan complete. IRAS structure ready.\n")