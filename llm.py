from abc import ABC, abstractmethod
from typing import Type, TypeVar
import base64
import json

# constants
image_embed_prefix = "🖼️🆙 "
log_to_console = False

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
                if human.startswith(image_embed_prefix):
                    with open(human.lstrip(image_embed_prefix), mode="rb") as f:
                        content = f.read()
                    user_msg_parts.append({"type": "image",
                                            "source": encode_image(content)})
                else:
                    user_msg_parts.append({"type": "text", "text": human})

            if assi:
                if user_msg_parts:
                    history_claude_format.append({"role": "user", "content": user_msg_parts})
                    user_msg_parts = []

                history_claude_format.append({"role": "assistant", "content": assi})

        if message:
            user_msg_parts.append({"type": "text", "text": human})
            
        if user_msg_parts:
            history_claude_format.append({"role": "user", "content": user_msg_parts})

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
            if prompt is not None:
                prompt += f"[INST] {human} [/INST]\n"
            if assi is not None:
                prompt += f"{assi}</s>\n"
        if message:
            prompt += f"[INST] {message} [/INST]"

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
