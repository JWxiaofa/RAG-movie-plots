from transformers import T5Tokenizer, T5ForConditionalGeneration
from extract_info import get_movie_titles, get_movie_plots
import torch
from transformers import pipeline

def get_dolly_output(input_text: str) -> str:
    '''
    using Dolly LLM
    :param input_text: prompt/query text
    :return: LLM response
    '''
    generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    title = get_movie_titles(input_text, limit=1)
    plot = get_movie_plots(input_text, limit=1)

    query = input_text + f"using the following information: movie " \
                         f"name:" \
                         f" {title[0]}; " \
                         f"movie plots: {plot[0]}."
    res = generate_text(query)
    return res[0]["generated_text"]

def get_llm_output(input_text: str) -> str:
    '''
    using T5 LLM
    :param input_text: user input
    :return: LLM response
    '''
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    # retrieve from database first
    title = get_movie_titles(input_text, limit=1)
    plot = get_movie_plots(input_text, limit=1)

    # use original user input and retrieved information as prompt(query)
    query = input_text + f" use the following information to answer and SUMMARIZE THE MOVIE PLOT: " \
                         f"movie name: {title[0]}, movie plots: {plot[0]};"

    input_ids = tokenizer(query, return_tensors="pt").input_ids

    outputs = model.generate(input_ids, max_new_tokens=100)
    output = tokenizer.decode(outputs[0]).rstrip("</s>").lstrip("<pad>").strip()

    return output

if __name__ == '__main__':
    text = "what movie are about moon?"
    plot = get_movie_plots(text, limit=1)[0]
    print(plot)
    generate_result = get_llm_output(text)
    print(generate_result)
