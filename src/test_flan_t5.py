from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "/home/ap2194.UNT/micro/data/flan_t5_micro"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

prompt = """You are an expert instructional designer specializing in computer science and cybersecurity education.

Task:
Create 3 focused sub-questions strictly based on the same concept and topic as the original question.

Rules:
- Do not deviate from the topic.
- Each sub-question must test only one concept or one skill.
- Keep the wording clear and specific.
- Do not add unrelated context.
- Output only the sub-questions as a numbered list.

Original question:
Compare paging and Segmentation."""

inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    num_beams=4,
    early_stopping=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))