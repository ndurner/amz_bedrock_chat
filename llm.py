from abc import ABC, abstractmethod
from typing import Type, TypeVar
import base64
import os
import json
from doc2json import process_docx
import fitz
from PIL import Image
import io
import boto3
from botocore.config import Config
import re

# constants
log_to_console = False
use_document_message_type = False  # AWS document message type usage

LLMClass = TypeVar('LLMClass', bound='LLM')

class LLM:
    @staticmethod
    def create_llm(model: str) -> Type[LLMClass]:
        return LLM()

    def generate_body(self, message, history):
        messages = []

        # AWS API requires strict user, assi, user, ... sequence
        lastTypeHuman = False

        for human, assi in history:
            if human:
                if lastTypeHuman:
                    last_msg = messages.pop()
                    user_msg_parts = last_msg["content"]
                else:
                    user_msg_parts = []

                if isinstance(human, tuple):
                    user_msg_parts.extend(self._process_file(human[0]))
                else:
                    user_msg_parts.extend([{"text": human}])

                messages.append({"role": "user", "content": user_msg_parts})
                lastTypeHuman = True
            if assi:
                messages.append({"role": "assistant", "content": [{"text": assi}]})
                lastTypeHuman = False
        
        user_msg_parts = []
        if message.text:
            user_msg_parts.append({"text": message.text})
        if message.files:
            for file in message.files:
                user_msg_parts.extend(self._process_file(file.path))
        
        if user_msg_parts:
            messages.append({"role": "user", "content": user_msg_parts})
        
        return messages

    def _process_file(self, file_path):
        if use_document_message_type and self._is_supported_document_type(file_path):
            return [self._create_document_message(file_path)]
        else:
            return self._encode_file(file_path)

    def _is_supported_document_type(self, file_path):
        supported_extensions = ['.pdf', '.csv', '.doc', '.docx', '.xls', '.xlsx', '.html', '.txt', '.md']
        return os.path.splitext(file_path)[1].lower() in supported_extensions

    def _create_document_message(self, file_path):
        with open(file_path, 'rb') as file:
            file_content = file.read()
        
        file_name = re.sub(r'[^a-zA-Z0-9\s\-\(\)\[\]]', '', os.path.basename(file_path))[:200].strip() or "unnamed_file"
        file_extension = os.path.splitext(file_path)[1][1:]  # Remove the dot

        return {
            "document": {
                "name": file_name,
                "format": file_extension,
                "source": {
                    "bytes": file_content
                }
            }
        }

    def _encode_file(self, fn: str) -> list:
        if fn.endswith(".docx"):
            return [{"text": process_docx(fn)}]
        elif fn.endswith(".pdf"):
            return self._process_pdf_img(fn)
        else:
            with open(fn, mode="rb") as f:
                content = f.read()

            if isinstance(content, bytes):
                try:
                    # try to add as image
                    image_data = self._encode_image(content)
                    return [{"image": image_data}]
                except:
                    # not an image, try text
                    content = content.decode('utf-8', 'replace')
            else:
                content = str(content)

            fname = os.path.basename(fn)
            return [{"text": f"``` {fname}\n{content}\n```"}]

    def _process_pdf_img(self, pdf_fn: str):
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
            
            # Append the message parts
            message_parts.append({"text": f"Page {page.number} of file '{pdf_fn}'"})
            message_parts.append({"image": {
                "format": "png",
                "source": {"bytes": img_byte_arr}
            }})

        pdf.close()

        return message_parts

    def _encode_image(self, image_data):
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

        return {
                "format": image_type,
                "source": {"bytes": image_data}
            }

    def read_response(self, response_stream):
        for event in response_stream:
            if 'contentBlockDelta' in event:
                yield event['contentBlockDelta']['delta']['text']
            if 'messageStop' in event:
                if log_to_console:
                    print(f"\nStop reason: {event['messageStop']['stopReason']}")
            if 'metadata' in event:
                metadata = event['metadata']
                if 'usage' in metadata and log_to_console:
                    print("\nToken usage:")
                    print(f"Input tokens: {metadata['usage']['inputTokens']}")
                    print(f"Output tokens: {metadata['usage']['outputTokens']}")
                    print(f"Total tokens: {metadata['usage']['totalTokens']}")