import streamlit as st
from PIL import Image
import torch
from model import model, processor

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def get_storemsg(data):
    output_msg = ''
    for msg in data:
        if msg['role']=='storemsg':
            output_msg += msg['content']
    return output_msg


st.title("Image Analyser")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]


# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        raw_image = Image.open(uploaded_file)

for msg in st.session_state.messages:
    if msg['role'] != 'storemsg':
        st.chat_message(msg["role"]).write(msg["content"])
# prompt_updated = ''



if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "storemsg","content":f"{prompt}\nASSISTANT:"})
    input_prompt = f"USER: <image>\n{get_storemsg(st.session_state.messages)}"
    inputs = processor(input_prompt, raw_image, return_tensors='pt').to(0, torch.float16)
    # Generate conversation response
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    conversation_result = processor.decode(output[0][2:], skip_special_tokens=True)
    reformed_conver = conversation_result.split('ASSISTANT:')[-1]
    st.session_state.messages.append({"role": "storemsg","content":reformed_conver+'\nUSER:\n'})

    st.session_state.messages.append({"role": "assistant", "content": f"{reformed_conver}"})
    st.chat_message("assistant").write(reformed_conver)

    final_out = get_storemsg(st.session_state.messages)
    print(input_prompt)
    

# print(st.session_state.messages)
    # Rerun the script to update the chat history
    # st.experimental_rerun()
