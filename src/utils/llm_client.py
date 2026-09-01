import os
import time

from google import genai
from dotenv import load_dotenv

load_dotenv()


class LLMResponse:
    def __init__(self, text, prompt_tokens, completion_tokens):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class LLMClient:
    def __init__(self, model_id, temperature=0.0, max_output_tokens=512,
                 retry_attempts=4, retry_backoff_s=5):
        self.model_id = model_id
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.retry_attempts = retry_attempts
        self.retry_backoff_s = retry_backoff_s

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("set GEMINI_API_KEY in your .env")
        self.client = genai.Client(api_key=api_key)

    def complete(self, prompt, stop=None):
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        if stop:
            generation_config["stop_sequences"] = stop

        last_err = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                interaction = self.client.interactions.create(
                    model=self.model_id,
                    input=prompt,
                    generation_config=generation_config,
                )
                text = interaction.outputs[-1].text
                usage = interaction.usage
                return LLMResponse(
                    text=text,
                    prompt_tokens=getattr(usage, "total_input_tokens", 0),
                    completion_tokens=getattr(usage, "total_output_tokens", 0),
                )
            except Exception as e:
                last_err = e
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_backoff_s * attempt)
        raise RuntimeError(f"Gemini call failed after {self.retry_attempts} tries: {last_err}")