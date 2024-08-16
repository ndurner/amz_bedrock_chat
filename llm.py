from abc import ABC, abstractmethod
from typing import Type, TypeVar
import base64
import os
import json
from doc2json import process_docx
import fitz
from PIL import Image
import io

# constants
log_to_console = False

def process_pdf_img(pdf_fn: str):
    pdf = fitz.open(pdf_fn)
    message_parts = []

    for page in pdf.pages():
        # Create a transformation matrix for rendering at the calculated scale
        mat = fitz.Matrix(0.6, 0.6)
        
        # Render the page to a pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Encode image to base64
        base64_encoded = base64.b64encode(img_byte_arr).decode('utf-8')
        
        # Append the message part
        message_parts.append({
            "type": "text",
            "text": f"Page {page.number} of file '{pdf_fn}'"
        })
        message_parts.append({"type": "image", "source": {"type": "base64",
            "media_type": "image/png",
            "data": base64_encoded}})

    pdf.close()

    return message_parts

def encode_image(image_data):
    """Generates a prefix for image base64 data in the required format for the
    four known image formats: png, jpeg, gif, and webp.

    Args:
    image_data: The image data, encoded in base64.

    Returns:
    An object encoding the image
    """

    # Get the first few bytes of the image data.
    magic_number = image_data[:4]
  
    # Check the magic number to determine the image type.
    if magic_number.startswith(b'\x89PNG'):
        image_type = 'png'
    elif magic_number.startswith(b'\xFF\xD8'):
        image_type = 'jpeg'
    elif magic_number.startswith(b'GIF89a'):
        image_type = 'gif'
    elif magic_number.startswith(b'RIFF'):
        if image_data[8:12] == b'WEBP':
            image_type = 'webp'
        else:
            # Unknown image type.
            raise Exception("Unknown image type")
    else:
        # Unknown image type.
        raise Exception("Unknown image type")

    return {"type": "base64",
            "media_type": "image/" + image_type,
            "data": base64.b64encode(image_data).decode('utf-8')}

def encode_file(fn: str) -> list:
    user_msg_parts = []

    if fn.endswith(".docx"):
        user_msg_parts.append({"type": "text", "text": process_docx(fn)})
    elif fn.endswith(".pdf"):
        user_msg_parts.extend(process_pdf_img(fn))
    else:
        with open(fn, mode="rb") as f:
            content = f.read()

        isImage = False
        if isinstance(content, bytes):
            try:
                # try to add as image
                content = encode_image(content)
                isImage = True
            except:
                # not an image, try text
                content = content.decode('utf-8', 'replace')
        else:
            content = str(content)

        if isImage:
            user_msg_parts.append({"type": "image",
                                "source": content})
        else:
            fname = os.path.basename(fn)
            user_msg_parts.append({"type": "text", "text": f"``` {fname}\n{content}\n```"})

    return user_msg_parts

LLMClass = TypeVar('LLMClass', bound='LLM')
class LLM(ABC):
    @abstractmethod
    def generate_body(message, history, system_prompt, temperature, max_tokens):
        pass

    @abstractmethod
    def read_response(message, history, system_prompt, temperature, max_tokens):
        pass

    @staticmethod
    def create_llm(model: str) -> Type[LLMClass]:
        if model.startswith("anthropic.claude"):
            return Claude()
        elif model.startswith("mistral."):
            return Mistral()
        else:
            raise ValueError(f"Unsupported model: {model}")

class Claude(LLM):
    @staticmethod
    def generate_body(message, history, system_prompt, temperature, max_tokens):
        history_claude_format = []
        user_msg_parts = []
        for human, assi in history:
            if human:
                if type(human) is tuple:
                    user_msg_parts.extend(encode_file(human[0]))
                else:
                    user_msg_parts.append({"type": "text", "text": human})

            if assi is not None:
                if user_msg_parts:
                    history_claude_format.append({"role": "user", "content": user_msg_parts})
                    user_msg_parts = []

                history_claude_format.append({"role": "assistant", "content": assi})

        if message['text']:
            user_msg_parts.append({"type": "text", "text": message['text']})
        if message['files']:
            for file in message['files']:
                user_msg_parts.extend(encode_file(file['path']))
        history_claude_format.append({"role": "user", "content": user_msg_parts})
        user_msg_parts = []

        if log_to_console:
            print(f"br_prompt: {str(history_claude_format)}")

        body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "system": system_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": history_claude_format
            })
        
        return body

    @staticmethod
    def read_response(response_body) -> Type[str]:
        return response_body.get('content')[0].get('text')

class Mistral(LLM):
    @staticmethod
    def generate_body(message, history, system_prompt, temperature, max_tokens):
        prompt = "<s>"
        for human, assi in history:
            if human:
                if type(human) is tuple:
                    prompt += f"[INST] {encode_file(human[0])} [/INST]"
                else:
                    prompt += f"[INST] {human} [/INST]"

            if assi is not None:
                prompt += f"{assi}</s>"

        if message['text'] or message['files']:
            prompt += "[INST] "

            if message['text']:
                prompt += message['text']
            if message['files']:
                for file in message['files']:
                    prompt += f"{encode_file(file['path'])}\n"
            
            prompt += " [/INST]"

        if log_to_console:
            print(f"br_prompt: {str(prompt)}")

        body = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        return body

    @staticmethod
    def read_response(response_body) -> Type[str]:
        return response_body.get('outputs')[0].get('text')
