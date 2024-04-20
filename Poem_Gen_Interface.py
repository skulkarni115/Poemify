# needs streamlit version 1.24.0 in order to run the image uploading system - recent versions of the package have issues
#need command streamlit run Poem_Gen_Interface.py to run
from transformers import AutoProcessor, AutoModelForCausalLM
import streamlit
import torch
from PIL import Image
import numpy as np
import os

streamlit.set_page_config(
    page_title="Poemify 📝",  
    page_icon=":pencil2:",
    layout="wide",  
    initial_sidebar_state="auto",  
)

#load processor
processor = AutoProcessor.from_pretrained("microsoft/git-base")

#load in model
model =  AutoModelForCausalLM.from_pretrained("microsoft/git-base")

# Load the saved model and run on cpu
model.load_state_dict(torch.load(r'finetuned_model_v1', map_location=torch.device('cpu')))

# enable eval mode
model.eval()


#function for formatting poem

def format_poem(poem, max_line_length):
    """
    Function for the final formatting of generated poetry text.
    Adds in newline characters as needed
    
    Args:
    poem(str): poetry text - all as one line
    max_line_length(int): max length for a line in the poem (based on characters)
    
    Returns:
    lines: list of the poem lines
    """
    lines = []
    current_line = ""
    
    for word in poem.split():
        
        # Check if adding the next word exceeds the max line length
        if len(current_line) + len(word) + 1 <= max_line_length:
            #add word to the line
            current_line += word + " "
            
        else:
            #append line to the list and reset current line var
            lines.append(current_line.strip())
            current_line = word + " "
    
    # Add the last line to list
    if current_line:
        lines.append(current_line.strip())
    
    #join with new line chars
    #formatted_poem = "\n".join(lines)
    return lines


#function for generating poem

def create_poem(image):
    """
    Function for generating a poem based on an image
    
    Args:
    Image: Image file opened with PIL library
    
    Returns:
    poem_final(str): generated poem
    """
    
    #run the image through the processor
    inputs = processor(images=image, return_tensors="pt")
    
    #grab pixel values
    pixel_values = inputs.pixel_values
    
    #grab logits
    logits = model.generate(pixel_values=pixel_values, max_length=50)
    
    #decode and get poem
    poem = processor.batch_decode(logits, skip_special_tokens=True)[0]
    
    poem_final = format_poem(poem,25)
    
    return poem_final


def main():
    """
    Main loop for streamlit application.
    """
    
    streamlit.title("Poemify: An Image to Poetry Generation Application")
    streamlit.write("Created by Surabhi Kulkarni and Hrisha Yagnik for the Large Language Models class at Northeastern .")

    # Upload image
    image = streamlit.file_uploader("Upload Image and then Click Poemify to get a poem", type=["jpg", "jpeg", "png"])

    if image is not None:
        # Display uploaded image
        img = Image.open(image)
        streamlit.image(img, caption='Uploaded Image', use_column_width=True)

        # Generate poem
        if streamlit.button('Poemify'):
            #use the create poem function
            poem = create_poem(img)
            streamlit.write("### Poem:")
            for line in poem:
                streamlit.write(line)

if __name__ == "__main__":
    main()