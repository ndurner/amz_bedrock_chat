import gradio as gr
import json
import os
import boto3

from settings_mgr import generate_download_settings_js, generate_upload_settings_js
from llm import LLM, log_to_console
from botocore.config import Config

dump_controls = False

def undo(history):
    history.pop()
    return history

def dump(history):
    return str(history)

def load_settings():  
    # Dummy Python function, actual loading is done in JS  
    pass  

def save_settings(acc, sec, prompt, temp):  
    # Dummy Python function, actual saving is done in JS  
    pass  

def process_values_js():
    return """
    () => {
        return ["access_key", "secret_key", "token"];
    }
    """

def bot(message, history, aws_access, aws_secret, aws_token, system_prompt, temperature, max_tokens, model: str, region):
    try:
        llm = LLM.create_llm(model)
        messages = llm.generate_body(message, history)

        config = Config(
            read_timeout = 600,
            connect_timeout = 30,
            retries = {
                'max_attempts': 10,
                'mode': 'adaptive'
            }
        )

        sess = boto3.Session(
            aws_access_key_id = aws_access,
            aws_secret_access_key = aws_secret,
            aws_session_token = aws_token,
            region_name = region)
        br = sess.client(service_name="bedrock-runtime", config = config)

        response = br.converse_stream(
            modelId = model,
            messages = messages,
            system = [{"text": system_prompt}],
            inferenceConfig = {
                "temperature": temperature,
                "maxTokens": max_tokens,
            }
        )
        response_stream = response.get('stream')
        
        partial_response = ""
        for chunk in llm.read_response(response_stream):
            partial_response += chunk
            yield partial_response

    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")

def import_history(history, file):
    with open(file.name, mode="rb") as f:
        content = f.read()

        if isinstance(content, bytes):
            content = content.decode('utf-8', 'replace')
        else:
            content = str(content)

    # Deserialize the JSON content
    import_data = json.loads(content)

    # Check if 'history' key exists for backward compatibility
    if 'history' in import_data:
        history = import_data['history']
        system_prompt.value = import_data.get('system_prompt', '')  # Set default if not present
    else:
        # Assume it's an old format with only history data
        history = import_data

    return history, system_prompt.value  # Return system prompt value to be set in the UI

with gr.Blocks() as demo:
    gr.Markdown("# Amazon™️ Bedrock™️ Chat™️ (Nils' Version™️) feat. Mistral™️ AI & Anthropic™️ Claude™️")

    with gr.Accordion("Startup"):
        gr.Markdown("""Use of this interface permitted under the terms and conditions of the 
                    [MIT license](https://github.com/ndurner/amz_bedrock_chat/blob/main/LICENSE).
                    Third party terms and conditions apply, particularly
                    those of the LLM vendor (AWS) and hosting provider (Hugging Face). This software and the AI models may make mistakes, so verify all outputs.""")
        
        aws_access = gr.Textbox(label="AWS Access Key", elem_id="aws_access")
        aws_secret = gr.Textbox(label="AWS Secret Key", elem_id="aws_secret")
        aws_token = gr.Textbox(label="AWS Session Token", elem_id="aws_token")
        model = gr.Dropdown(label="Model", value="anthropic.claude-3-5-sonnet-20240620-v1:0", allow_custom_value=True, elem_id="model",
                            choices=["anthropic.claude-3-5-sonnet-20240620-v1:0", "anthropic.claude-3-opus-20240229-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-v2:1", "anthropic.claude-v2",
                                     "mistral.mistral-7b-instruct-v0:2", "mistral.mixtral-8x7b-instruct-v0:1", "mistral.mistral-large-2407-v1:0"])
        system_prompt = gr.TextArea("You are a helpful yet diligent AI assistant. Answer faithfully and factually correct. Respond with 'I do not know' if uncertain.", label="System Prompt", lines=3, max_lines=250, elem_id="system_prompt")  
        region = gr.Dropdown(label="Region", value="us-west-2", allow_custom_value=True, elem_id="region",
                            choices=["eu-central-1", "eu-west-3", "us-east-1", "us-west-1", "us-west-2"])
        temp = gr.Slider(0, 1, label="Temperature", elem_id="temp", value=1)
        max_tokens = gr.Slider(1, 8192, label="Max. Tokens", elem_id="max_tokens", value=4096)
        save_button = gr.Button("Save Settings")  
        load_button = gr.Button("Load Settings")  
        dl_settings_button = gr.Button("Download Settings")
        ul_settings_button = gr.Button("Upload Settings")

        load_button.click(load_settings, js="""  
            () => {  
                let elems = ['#aws_access textarea', '#aws_secret textarea', '#aws_token textarea', '#system_prompt textarea', '#temp input', '#max_tokens input', '#model', '#region'];
                elems.forEach(elem => {
                    let item = document.querySelector(elem);
                    let event = new InputEvent('input', { bubbles: true });
                    item.value = localStorage.getItem(elem.split(" ")[0].slice(1)) || '';
                    item.dispatchEvent(event);
                });
            }  
        """)

        save_button.click(save_settings, [aws_access, aws_secret, aws_token, system_prompt, temp, max_tokens, model, region], js="""  
            (acc, sec, tok, system_prompt, temp, ntok, model, region) => {  
                localStorage.setItem('aws_access', acc);  
                localStorage.setItem('aws_secret', sec);  
                localStorage.setItem('aws_token', tok);  
                localStorage.setItem('system_prompt', system_prompt);
                localStorage.setItem('temp', document.querySelector('#temp input').value);  
                localStorage.setItem('max_tokens', document.querySelector('#max_tokens input').value);  
                localStorage.setItem('model', model);  
                localStorage.setItem('region', region);  
            }  
        """) 

        control_ids = [('aws_access', '#aws_access textarea'),
                       ('aws_secret', '#aws_secret textarea'),
                       ('aws_token', '#aws_token textarea'),
                       ('system_prompt', '#system_prompt textarea'),
                       ('temp', '#temp input'),
                       ('max_tokens', '#max_tokens input'),
                       ('model', '#model'),
                       ('region', '#region')]
        controls = [aws_access, aws_secret, aws_token, system_prompt, temp, max_tokens, model, region]

        dl_settings_button.click(None, controls, js=generate_download_settings_js("amz_chat_settings.bin", control_ids))
        ul_settings_button.click(None, None, None, js=generate_upload_settings_js(control_ids))

    chat = gr.ChatInterface(fn=bot, multimodal=True, additional_inputs=controls, retry_btn = None, autofocus = False)
    chat.textbox.file_count = "multiple"
    chatbot = chat.chatbot
    chatbot.show_copy_button = True
    chatbot.height = 350

    if dump_controls:
        with gr.Row():
            dmp_btn = gr.Button("Dump")
            txt_dmp = gr.Textbox("Dump")
            dmp_btn.click(dump, inputs=[chatbot], outputs=[txt_dmp])

    with gr.Accordion("Import/Export", open = False):
        import_button = gr.UploadButton("History Import")
        export_button = gr.Button("History Export")
        export_button.click(lambda: None, [chatbot, system_prompt], js="""
            (chat_history, system_prompt) => {
                const export_data = {
                    history: chat_history,
                    system_prompt: system_prompt
                };
                const history_json = JSON.stringify(export_data);
                const blob = new Blob([history_json], {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'chat_history.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
            """)
        dl_button = gr.Button("File download")
        dl_button.click(lambda: None, [chatbot], js="""
            (chat_history) => {
                // Attempt to extract content enclosed in backticks with an optional filename
                const contentRegex = /```(\\S*\\.(\\S+))?\\n?([\\s\\S]*?)```/;
                const match = contentRegex.exec(chat_history[chat_history.length - 1][1]);
                if (match && match[3]) {
                    // Extract the content and the file extension
                    const content = match[3];
                    const fileExtension = match[2] || 'txt'; // Default to .txt if extension is not found
                    const filename = match[1] || `download.${fileExtension}`;
                    // Create a Blob from the content
                    const blob = new Blob([content], {type: `text/${fileExtension}`});
                    // Create a download link for the Blob
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    // If the filename from the chat history doesn't have an extension, append the default
                    a.download = filename.includes('.') ? filename : `${filename}.${fileExtension}`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                } else {
                    // Inform the user if the content is malformed or missing
                    alert('Sorry, the file content could not be found or is in an unrecognized format.');
                }
            }
        """)
        import_button.upload(import_history, inputs=[chatbot, import_button], outputs=[chatbot, system_prompt])

demo.queue().launch()