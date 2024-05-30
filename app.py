import gradio as gr
from llm import get_llm_output
from extract_info import get_movie_titles, get_movie_plots

def get_three_relevant_movies(query: str):
    '''
    Get 3 relevant movies from database
    :param query: user input text
    :return: 3 strs
    '''
    titles = get_movie_titles(query, limit=3)
    plots = get_movie_plots(query, limit=3)

    result1 = f"Movie Name: {titles[0]}\n" + f"Plot: {plots[0]}"
    result2 = f"Movie Name: {titles[1]}\n" + f"Plot: {plots[1]}"
    result3 = f"Movie Name: {titles[2]}\n" + f"Plot: {plots[2]}"

    return result1, result2, result3

def handle_button_click(query: str):
    '''
    Return relevant information for butter click request
    :param query: user input text
    :return: all information needed for button click request
    '''
    answer = get_llm_output(query)
    movie1, movie2, movie3 = get_three_relevant_movies(query)
    return answer, movie1, movie2, movie3


with gr.Blocks(theme=gr.themes.Monochrome(), title="Movie Question Answering") as demo:
    gr.Markdown('''
    # Movie Question Answering 🎬
    # Ask about movie plots! 🍿''')
    textbox = gr.Textbox(label="Question:")
    with gr.Row():
        button = gr.Button("Submit", variant="primary")
    with gr.Column():
        output = gr.Textbox(lines=1, max_lines=10, label="Answer:")
        with gr.Row():
            movie_output_1 = gr.Textbox(lines=1, max_lines=10, label="Movie 1:")
            movie_output_2 = gr.Textbox(lines=1, max_lines=10, label="Movie 2:")
            movie_output_3 = gr.Textbox(lines=1, max_lines=10, label="Movie 3:")

    button.click(
        fn=handle_button_click,
        inputs=textbox,
        outputs=[output, movie_output_1, movie_output_2, movie_output_3]
    )


demo.launch()