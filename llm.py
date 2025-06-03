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
import gradio

# constants
log_to_console = False

LLMClass = TypeVar('LLMClass', bound='LLM')

class LLM:
    @staticmethod
    def create_llm(model: str) -> Type[LLMClass]:
        return LLM()

    def generate_body(self, message, history):
        messages = []

        # AWS API requires strict user, assi, user, ... sequence
        lastTypeHuman = False

        for msg in history:
            if msg['role'] == "user":
                if lastTypeHuman:
                    last_msg = messages.pop()
                    user_msg_parts = last_msg["content"]
                else:
                    user_msg_parts = []

                content = msg['content']
                if isinstance(content, gradio.File) or isinstance(content, gradio.Image):
                    user_msg_parts.extend(self._process_file(content.value['path']))
                elif isinstance(content, tuple):
                    user_msg_parts.extend(self._process_file(content[0]))
                else:
                    user_msg_parts.extend([{"type": "text", "text": content}])

                messages.append({"role": "user", "content": user_msg_parts})
                lastTypeHuman = True
            else:
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": msg['content']}]
                })
                lastTypeHuman = False
        
        if lastTypeHuman:
            last_msg = messages.pop()
            user_msg_parts = last_msg["content"]
        else:
            user_msg_parts = []
        
        if message:
            if message["text"]:
                user_msg_parts.append({"type": "text", "text": message["text"]})
            if message["files"]:
                for file in message["files"]:
                    user_msg_parts.extend(self._process_file(file))
        
        if user_msg_parts:
            messages.append({"role": "user", "content": user_msg_parts})
        
        return messages

    def _process_file(self, file_path):
        return self._encode_file(file_path)



    def _encode_file(self, fn: str) -> list:
        if fn.endswith(".docx"):
            return [{"type": "text", "text": process_docx(fn)}]
        elif fn.endswith(".pdf"):
            return self._process_pdf_img(fn)
        else:
            with open(fn, mode="rb") as f:
                content = f.read()

            if isinstance(content, bytes):
                try:
                    # try to add as image
                    image_data = self._encode_image(content)
                    return [{"type": "image", "source": image_data}]
                except:
                    # not an image, try text
                    content = content.decode('utf-8', 'replace')
            else:
                content = str(content)

            fname = os.path.basename(fn)
            return [{"type": "text", "text": f"``` {fname}\n{content}\n```"}]

    def _process_pdf_img(self, pdf_fn: str):
        pdf = fitz.open(pdf_fn)
        message_parts = []
        page_scales = {}  # Cache for similar page sizes

        def calculate_tokens(width, height):
            return (width * height) / 750

        for page in pdf.pages():
            page_rect = page.rect
            orig_width = page_rect.width
            orig_height = page_rect.height
            page_key = (orig_width, orig_height)

            # Use cached scale as starting point if available
            scale = page_scales.get(page_key, 1.0)
            
            while True:
                # Render with current scale
                mat = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Check actual rendered dimensions
                actual_tokens = calculate_tokens(pix.width, pix.height)
                actual_long_edge = max(pix.width, pix.height)
                
                if actual_long_edge <= 1568 and actual_tokens <= 1600:
                    # We found a good scale, cache it
                    if page_key not in page_scales:
                        page_scales[page_key] = scale
                    break
                
                # Calculate new scale factor based on both constraints
                if actual_long_edge > 1568:
                    scale_factor = min(1568 / actual_long_edge, 0.9)
                else:
                    scale_factor = min(math.sqrt(1600 / actual_tokens), 0.9)
                
                scale *= scale_factor

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Handle compression
            quality = 95
            while True:
                buffer = io.BytesIO()
                img.save(buffer, format="webp", quality=quality)
                img_bytes = buffer.getvalue()
                
                if len(img_bytes) <= 5 * 1024 * 1024 or quality <= 20:
                    break
                    
                quality = max(int(quality * 0.9), 20)

            message_parts.append({"type": "text", "text": f"Page {page.number + 1} of file '{pdf_fn}'"})
            message_parts.append({"type": "image", "source": self._encode_image(img_bytes)})

        pdf.close()
        return message_parts

    def _encode_image(self, image_data):
        try:
            # Open the image using Pillow
            img = Image.open(io.BytesIO(image_data))
            original_format = img.format.lower()
        except IOError:
            raise Exception("Unknown image type")
        
        # Ensure correct orientation based on EXIF
        try:
            exif = img._getexif()
            if exif:
                orientation = exif.get(274)  # 274 is the orientation tag
                if orientation:
                    # Rotate or flip based on EXIF orientation
                    if orientation == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation == 8:
                        img = img.rotate(90, expand=True)
        except:
            pass  # If EXIF processing fails, use image as-is

        # check if within the limits for Claude as per https://docs.anthropic.com/en/docs/build-with-claude/vision
        def calculate_tokens(width, height):
            return (width * height) / 750

        tokens = calculate_tokens(img.width, img.height)
        long_edge = max(img.width, img.height)
        format_ok = original_format in ["jpg", "jpeg", "png", "webp"]

        # Check if the image already meets all requirements
        if format_ok and (long_edge <= 1568 and tokens <= 1600 and len(image_data) <= 5 * 1024 * 1024):
            out_fmt = original_format
            return {
                "type": "base64",
                "media_type": f"image/{out_fmt}",
                "data": base64.b64encode(image_data).decode('utf-8')
            }

        # If we need to modify the image, proceed with resizing and/or compression
        orig_scale_factor = 1
        orig_img = img
        while long_edge > 1568 or tokens > 1600:
            if long_edge > 1568:
                scale_factor = min(1568 / long_edge, 0.9)
            else:
                scale_factor = min(math.sqrt(1600 / tokens), 0.9)

            scale_factor = orig_scale_factor * scale_factor
            orig_scale_factor = scale_factor
            
            new_width = int(orig_img.width * scale_factor)
            new_height = int(orig_img.height * scale_factor)
            
            img = orig_img.resize((new_width, new_height), Image.LANCZOS)
            
            long_edge = max(img.width, img.height)
            tokens = calculate_tokens(img.width, img.height)

        # Try to save in original format first
        buffer = io.BytesIO()
        out_fmt = "png" if original_format == "png" else "webp"
        img.save(buffer, format=out_fmt, quality=95 if out_fmt == "webp" else None)
        image_data = buffer.getvalue()
        
        # If the image is still too large, switch to WebP and compress
        if len(image_data) > 5 * 1024 * 1024:
            quality = 95
            while len(image_data) > 5 * 1024 * 1024:
                quality = max(int(quality * 0.9), 20)
                buffer = io.BytesIO()
                img.save(buffer, format="webp", quality=quality)
                image_data = buffer.getvalue()
                if quality == 20:
                    # If we've reached quality 20 and it's still too large, resize
                    scale_factor = 0.9
                    new_width = int(img.width * scale_factor)
                    new_height = int(img.height * scale_factor)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    quality = 95  # Reset quality for the resized image
        return {
            "type": "base64",
            "media_type": f"image/{out_fmt}",
            "data": base64.b64encode(image_data).decode('utf-8')
        }

    def read_response(self, response_stream):
        """
        Handles response stream that may contain both regular text and tool use requests.
        Yields tuples of (text, tool_request, stop_reason) where:
        - text: accumulated text response
        - tool_request: dict with tool use details if present, None otherwise
        - stop_reason: string indicating why stream stopped, None while streaming
        """
        message = {}
        content = []
        message['content'] = content
        tool_use = {}
        text = ''
        stop_reason = None

        for chunk in response_stream:
            if 'chunk' in chunk:
                chunk = json.loads(chunk['chunk']['bytes'])

            if 'messageStart' in chunk or chunk.get('type') == 'message_start':
                if 'messageStart' in chunk:
                    message['role'] = chunk['messageStart']['role']
                else:
                    message['role'] = chunk['message']['role']
            elif 'contentBlockStart' in chunk or chunk.get('type') == 'content_block_start':
                start = chunk.get('contentBlockStart', chunk.get('start'))
                tool = start.get('toolUse') or start.get('tool_use')
                tool_use['toolUseId'] = tool['toolUseId']
                tool_use['name'] = tool['name']
            elif 'contentBlockDelta' in chunk or chunk.get('type') == 'content_block_delta':
                delta = chunk.get('contentBlockDelta', chunk.get('delta'))['delta'] if 'contentBlockDelta' in chunk else chunk['delta']
                if 'toolUse' in delta:
                    if 'input' not in tool_use:
                        tool_use['input'] = ''
                    tool_use['input'] += delta['toolUse']['input']
                elif 'text' in delta:
                    text += delta['text']
                    yield None, delta['text']
            elif 'contentBlockStop' in chunk or chunk.get('type') == 'content_block_stop':
                if 'input' in tool_use:
                    tool_use['input'] = json.loads(tool_use['input'])
                    content.append({'toolUse': tool_use})
                    tool_use = {}
                else:
                    content.append({'text': text})
            elif 'messageStop' in chunk or chunk.get('type') == 'message_stop':
                stop_reason = chunk.get('messageStop', chunk).get('stopReason') or chunk.get('stop_reason')
                yield stop_reason, message
            elif 'metadata' in chunk and 'usage' in chunk['metadata'] and log_to_console:
                metadata = chunk['metadata']
                print("\nToken usage:")
                print(f"Input tokens: {metadata['usage']['inputTokens']}")
                print(f"Output tokens: {metadata['usage']['outputTokens']}")
                print(f"Total tokens: {metadata['usage']['totalTokens']}")
