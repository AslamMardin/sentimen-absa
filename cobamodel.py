from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "cahya/t5-small-paraphrase-indonesia"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
