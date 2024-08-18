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
from PIL import Image
import io
import math

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
        try:
            # Open the image using Pillow
            img = Image.open(io.BytesIO(image_data))
            original_format = img.format.lower()
        except IOError:
            raise Exception("Unknown image type")
        
        # check if within the limits for Claude as per https://docs.anthropic.com/en/docs/build-with-claude/vision
        def calculate_tokens(width, height):
            return (width * height) / 750

        tokens = calculate_tokens(img.width, img.height)
        long_edge = max(img.width, img.height)

        # Check if the image already meets all requirements
        if long_edge <= 1568 and tokens <= 1600 and len(image_data) <= 5 * 1024 * 1024:
            return {
                "format": original_format,
                "source": {"bytes": image_data}
            }

        # If we need to modify the image, proceed with resizing and/or compression
        while long_edge > 1568 or tokens > 1600:
            if long_edge > 1568:
                scale_factor = max(1568 / long_edge, 0.9)
            else:
                scale_factor = max(math.sqrt(1600 / tokens), 0.9)
            
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            long_edge = max(new_width, new_height)
            tokens = calculate_tokens(new_width, new_height)

        # Try to save in original format first
        buffer = io.BytesIO()
        img.save(buffer, format=original_format, quality=95)
        image_data = buffer.getvalue()
        
        # If the image is still too large, switch to WebP and compress
        if len(image_data) > 5 * 1024 * 1024:
            format_to_use = "webp"
            quality = 95
            while len(image_data) > 5 * 1024 * 1024:
                quality = max(int(quality * 0.9), 20)
                buffer = io.BytesIO()
                img.save(buffer, format=format_to_use, quality=quality)
                image_data = buffer.getvalue()
                if quality == 20:
                    # If we've reached quality 20 and it's still too large, resize
                    scale_factor = 0.9
                    new_width = int(img.width * scale_factor)
                    new_height = int(img.height * scale_factor)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    quality = 95  # Reset quality for the resized image
        else:
            format_to_use = original_format

        return {
            "format": format_to_use,
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