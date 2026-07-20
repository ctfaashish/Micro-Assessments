from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "/home/ap2194.UNT/micro/data/flan_t5_micro_grouped"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

prompt = """You are an expert instructional designer specializing in computer science and cybersecurity education.

Task:
Create 3-5 focused sub-questions strictly based on the same concept and topic as the original question.

Rules:
- Do not deviate from the topic.
- Each sub-question must test only one concept or one skill.
- Keep the wording clear and specific.
- Do not add unrelated context.
- Output only the sub-questions as a numbered list.

Original question:
Explain paging and Segmentation."""

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    num_beams=4,
    early_stopping=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))