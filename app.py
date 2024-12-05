import gradio as gr
import json
import io
import boto3
import base64
from PIL import Image

from settings_mgr import generate_download_settings_js, generate_upload_settings_js
from llm import LLM, log_to_console
from code_exec import eval_restricted_script
from chat_export import import_history, get_export_js
from botocore.config import Config

dump_controls = False

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

def bot(message, history, aws_access, aws_secret, aws_token, system_prompt, temperature, max_tokens, model: str, region, python_use):
    try:
        llm = LLM.create_llm(model)
        messages = llm.generate_body(message, history)
        sys_prompt = [{"text": system_prompt}] if system_prompt else []

        config = Config(
            read_timeout = 600,
            connect_timeout = 30,
            retries = {'max_attempts': 10, 'mode': 'adaptive'}
        )

        tool_config = {
            "tools": [{
                "toolSpec": {
                    "name": "eval_python",
                    "description": "Evaluate a simple script written in a conservative, restricted subset of Python."
                                "Note: Augmented assignments, in-place operations (e.g., +=, -=), lambdas (e.g. list comprehensions) are not supported. "
                                "Use regular assignments and operations instead. Only 'import math' is allowed. "
                                "Returns: unquoted results without HTML encoding.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "script": {
                                    "type": "string",
                                    "description": "The Python script that will run in a RestrictedPython context. "
                                                "Avoid using augmented assignments or in-place operations (+=, -=, etc.), as well as lambdas (e.g. list comprehensions). "
                                                "Use regular assignments and operations instead. Only 'import math' is allowed."
                                }
                            },
                            "required": ["script"]
                        }
                    }
                }
            }]
        } if python_use else None

        sess = boto3.Session(
            aws_access_key_id = aws_access,
            aws_secret_access_key = aws_secret,
            aws_session_token = aws_token,
            region_name = region)
        br = sess.client(service_name="bedrock-runtime", config = config)

        whole_response = ""
        while True:
            response = br.converse_stream(
                modelId = model,
                messages = messages,
                system = sys_prompt,
                inferenceConfig = {
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                },
                **({'toolConfig': tool_config} if python_use else {})
            )

            for stop_reason, message in llm.read_response(response.get('stream')):
                if isinstance(message, str):
                    whole_response += message
                    yield whole_response

                if stop_reason:
                    if stop_reason == "tool_use":
                        messages.append(message)

                        for content in message['content']:
                            if 'toolUse' in content:
                                tool = content['toolUse']

                                if tool['name'] == 'eval_python':
                                    tool_result = {}
                                    try:
                                        tool_script = tool["input"]["script"]

                                        whole_response += f"\n``` script\n{tool_script}\n```\n"
                                        yield whole_response

                                        tool_result = eval_restricted_script(tool_script)
                                        tool_result_message = {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "toolResult": {
                                                        "toolUseId": tool['toolUseId'],
                                                        "content": [{"json": tool_result }]
                                                    }
                                                }
                                            ]
                                        }

                                        whole_response += f"\n``` result\n{tool_result if not tool_result['success'] else tool_result['prints']}\n```\n"
                                        yield whole_response
                                    except Exception as e:
                                        tool_result_message = {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "toolResult": {
                                                        "content": [{"text":  e.args[0]}],
                                                        "status": 'error'
                                                    }
                                                }
                                            ]
                                        }
                                        whole_response += f"\n``` error\n{e.args[0]}\n```\n"
                                        yield whole_response

                                    messages.append(tool_result_message)
                    else:
                        return

    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")

def import_history_guarded(aws_access, aws_secret, aws_token, region, history, file):
    # check credentials first
    try:
        sess = boto3.Session(
            aws_access_key_id = aws_access,
            aws_secret_access_key = aws_secret,
            aws_session_token = aws_token,
            region_name = region)
        br = sess.client(service_name="bedrock")
        br.list_foundation_models(byProvider="invalid")
    except Exception as e:
        raise gr.Error(f"Bedrock login error: {str(e)}")

    # actual import
    return import_history(history, file)

def export_history(h, s):
    pass

with gr.Blocks(delete_cache=(86400, 86400)) as demo:
    settings_state = gr.BrowserState({})
    
    gr.Markdown("# Amazon™️ Bedrock™️ Chat™️ (Nils' Version™️) feat. Mistral™️ AI & Anthropic™️ Claude™️")

    with gr.Accordion("Startup"):
        gr.Markdown("""Use of this interface permitted under the terms and conditions of the 
                    [MIT license](https://github.com/ndurner/amz_bedrock_chat/blob/main/LICENSE).
                    Third party terms and conditions apply, particularly
                    those of the LLM vendor (AWS) and hosting provider (Hugging Face). This software and the AI models may make mistakes, so verify all outputs.""")
        
        aws_access = gr.Textbox(label="AWS Access Key", elem_id="aws_access")
        aws_secret = gr.Textbox(label="AWS Secret Key", elem_id="aws_secret")
        aws_token = gr.Textbox(label="AWS Session Token", elem_id="aws_token")
        model = gr.Dropdown(label="Model", value="anthropic.claude-3-5-sonnet-20241022-v2:0", allow_custom_value=True, elem_id="model",
                            choices=["anthropic.claude-3-5-sonnet-20240620-v1:0", "anthropic.claude-3-opus-20240229-v1:0", "meta.llama3-1-405b-instruct-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-v2:1", "anthropic.claude-v2",
                                     "mistral.mistral-7b-instruct-v0:2", "mistral.mixtral-8x7b-instruct-v0:1", "mistral.mistral-large-2407-v1:0", "anthropic.claude-3-5-sonnet-20241022-v2:0", "us.amazon.nova-pro-v1:0", "us.amazon.nova-lite-v1:0", "us.amazon.nova-micro-v1:0"])
        system_prompt = gr.TextArea("You are a helpful yet diligent AI assistant. Answer faithfully and factually correct. Respond with 'I do not know' if uncertain.", label="System Prompt", lines=3, max_lines=250, elem_id="system_prompt")  
        region = gr.Dropdown(label="Region", value="us-west-2", allow_custom_value=True, elem_id="region",
                            choices=["eu-central-1", "eu-west-3", "us-east-1", "us-west-1", "us-west-2"])
        temp = gr.Slider(0, 1, label="Temperature", elem_id="temp", value=1)
        max_tokens = gr.Slider(1, 8192, label="Max. Tokens", elem_id="max_tokens", value=4096)
        python_use = gr.Checkbox(label="Python Use")
        save_button = gr.Button("Save Settings")  
        load_button = gr.Button("Load Settings")  
        dl_settings_button = gr.Button("Download Settings")
        ul_settings_button = gr.Button("Upload Settings")

        @demo.load(inputs=[settings_state], 
                outputs=[aws_access, aws_secret, aws_token, system_prompt, 
                        temp, max_tokens, model, region, python_use])
        def load_from_browser_storage(saved_values):
            if not saved_values:
                return (aws_access.value, aws_secret.value, aws_token.value, 
                        system_prompt.value, temp.value, max_tokens.value,
                        model.value, region.value, python_use.value)
            return (saved_values.get('aws_access', aws_access.value), 
                    saved_values.get('aws_secret', aws_secret.value), 
                    saved_values.get('aws_token', aws_token.value), 
                    saved_values.get('system_prompt', system_prompt.value),
                    saved_values.get('temp', temp.value), 
                    saved_values.get('max_tokens', max_tokens.value),
                    saved_values.get('model', model.value), 
                    saved_values.get('region', region.value),
                    saved_values.get('python_use', python_use.value))

        @gr.on(
            [aws_access.change, aws_secret.change, aws_token.change, 
             system_prompt.change, temp.change, max_tokens.change,
             model.change, region.change, python_use.change],
            inputs=[aws_access, aws_secret, aws_token, system_prompt, 
                    temp, max_tokens, model, region, python_use],
            outputs=[settings_state]
        )
        def save_to_browser_storage(acc, sec, tok, prompt, temperature, 
                                tokens, mdl, reg, py_use):
            return {
                'aws_access': acc,
                'aws_secret': sec,
                'aws_token': tok,
                'system_prompt': prompt,
                'temp': temperature,
                'max_tokens': tokens,
                'model': mdl,
                'region': reg,
                'python_use': py_use
            }

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
        controls = [aws_access, aws_secret, aws_token, system_prompt, temp, max_tokens, model, region, python_use]

        dl_settings_button.click(None, controls, js=generate_download_settings_js("amz_chat_settings.bin", control_ids))
        ul_settings_button.click(None, None, None, js=generate_upload_settings_js(control_ids))

    chat = gr.ChatInterface(fn=bot, multimodal=True, additional_inputs=controls, autofocus = False, type = "messages")
    chat.textbox.file_count = "multiple"
    chatbot = chat.chatbot
    chatbot.show_copy_button = True
    chatbot.height = 450

    if dump_controls:
        with gr.Row():
            dmp_btn = gr.Button("Dump")
            txt_dmp = gr.Textbox("Dump")
            dmp_btn.click(dump, inputs=[chatbot], outputs=[txt_dmp])

    with gr.Accordion("Import/Export", open = False):
        import_button = gr.UploadButton("History Import")
        export_button = gr.Button("History Export")
        export_button.click(lambda: None, [chatbot, system_prompt], js=get_export_js())
        dl_button = gr.Button("File download")
        dl_button.click(lambda: None, [chatbot], js="""
            (chat_history) => {
                // Only define exception mappings
                const languageToExt = {
                    'python': 'py',
                    'javascript': 'js',
                    'typescript': 'ts',
                    'csharp': 'cs',
                    'ruby': 'rb',
                    'shell': 'sh',
                    'bash': 'sh',
                    'markdown': 'md',
                    'yaml': 'yml',
                    'rust': 'rs',
                    'golang': 'go',
                    'kotlin': 'kt'
                };

                const contentRegex = /```(?:([^\\n]+)?\\n)?([\\s\\S]*?)```/;
                const match = contentRegex.exec(chat_history[chat_history.length - 1][1]);
                
                if (match && match[2]) {
                    const specifier = match[1] ? match[1].trim() : '';
                    const content = match[2];
                    
                    let filename = 'download';
                    let fileExtension = 'txt'; // default

                    if (specifier) {
                        if (specifier.includes('.')) {
                            // If specifier contains a dot, treat it as a filename
                            const parts = specifier.split('.');
                            filename = parts[0];
                            fileExtension = parts[1];
                        } else {
                            // Use mapping if exists, otherwise use specifier itself
                            const langLower = specifier.toLowerCase();
                            fileExtension = languageToExt[langLower] || langLower;
                            filename = 'code';
                        }
                    }

                    const blob = new Blob([content], {type: 'text/plain'});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${filename}.${fileExtension}`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }
            }
        """)
        import_button.upload(import_history_guarded, 
                            inputs=[aws_access, aws_secret, aws_token, region, chatbot, import_button], 
                            outputs=[chatbot, system_prompt])
demo.queue(default_concurrency_limit = None).launch()