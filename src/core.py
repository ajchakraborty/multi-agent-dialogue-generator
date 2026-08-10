from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import openai
import os
import random
import sys

try:
    # So standalone entry points (e.g. `python -m eval.run`) pick up credentials
    # from .env too, not just simulate.py. Never overrides the real environment,
    # so `eval "$(sfproxy env)"` still wins.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# sfproxy routing
#
# All OpenAI traffic (3 sim agents + 3 LLM judges) goes through the sfproxy
# Cortex gateway rather than api.openai.com. Token comes from `sfproxy env`;
# run `eval "$(sfproxy env)"` before generating, or put SNOWFLAKE_PAT in .env.
# ---------------------------------------------------------------------------

SFPROXY_BASE_URL = os.getenv(
    "SFPROXY_BASE_URL", "https://sfproxy.us.cloud.uniphore.com/v1"
)

# Cortex namespaces OpenAI models. Keeps `--model gpt-5.1` working unchanged
# in every existing command, script and doc.
MODEL_ALIASES = {
    "gpt-4.1": "openai-gpt-4.1",
    "gpt-5": "openai-gpt-5",
    "gpt-5-chat": "openai-gpt-5-chat",
    "gpt-5-mini": "openai-gpt-5-mini",
    "gpt-5-nano": "openai-gpt-5-nano",
    "gpt-5.1": "openai-gpt-5.1",
    "gpt-5.2": "openai-gpt-5.2",
    "gpt-5.4": "openai-gpt-5.4",
}


def resolve_model(model: str) -> str:
    """Map a bare OpenAI model name to its Cortex name. Idempotent."""
    return MODEL_ALIASES.get(model, model)


def _sfproxy_user() -> str:
    """Identity for X-Sfproxy-User. Omitting it is a 403 identity_required."""
    user = os.getenv("SFPROXY_USER")
    if user:
        return user.strip()

    # `sfproxy env` exports this as "X-Sfproxy-User: a@b.com|X-Sfproxy-Version: ..."
    raw = os.getenv("ANTHROPIC_CUSTOM_HEADERS", "")
    for header in raw.split("|"):
        name, _, value = header.partition(":")
        if name.strip().lower() == "x-sfproxy-user" and value.strip():
            return value.strip()

    raise RuntimeError(
        "sfproxy identity not found. Run: eval \"$(sfproxy env)\"  "
        "(or set SFPROXY_USER=you@uniphore.com)"
    )


def _sfproxy_token(explicit: Optional[str] = None) -> str:
    token = explicit or os.getenv("SNOWFLAKE_PAT") or os.getenv("SNOWFLAKE_CORTEX_TOKEN")
    if not token:
        raise RuntimeError(
            "SNOWFLAKE_PAT not set. Run: eval \"$(sfproxy env)\"  "
            "(tokens expire; re-run it if a job starts returning 401)"
        )
    return token


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@dataclass
class Message:
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationContext:
    messages: List[Message]
    agent_config: Dict[str, Any]
    turn_number: int


def convert_messages_to_hf_format(messages: List[Message]) -> str:
    """
    Convert conversation messages to HuggingFace model input format.
    
    Includes everything in the simulated context:
    - User messages wrapped in <user> tags
    - Assistant messages as full structured output (think/plan/action)
    - Tool results wrapped in <obs> tags
    
    Args:
        messages: List of Message objects from conversation history
        
    Returns:
        String representation of dialogue history in HF format
    """
    blocks = []
    
    for msg in messages:
        if msg.role == MessageRole.USER:
            blocks.append(f"<user> {msg.content} </user>")
        elif msg.role == MessageRole.ASSISTANT:
            # Check if message already has structured format (<action> tag)
            # If not, wrap plain text in structured format
            content = msg.content.strip()
            if "<action" in content:
                # Already in structured format, include as-is
                blocks.append(content)
            else:
                # Plain text (e.g., system greeting) - wrap in action tag only
                blocks.append(f"<action type=\"say\">{content}</action>")
        elif msg.role == MessageRole.TOOL:
            blocks.append(f"<obs> {msg.content} </obs>")
    
    return "\n\n".join(blocks)


def build_hf_prompt(
    system_prompt: str,
    tools: Dict[str, Any],
    dialogue_history: str
) -> str:
    """
    Build complete HuggingFace prompt from components.
    
    Args:
        system_prompt: System instruction prompt text
        tools: Dictionary of tool definitions
        dialogue_history: Conversation history in HF format (from convert_messages_to_hf_format)
        
    Returns:
        Complete prompt string ready for model input
    """
    import json
    
    # Build system content with tools
    system_content = f"{system_prompt}\n\nAvailable Tools:\n{json.dumps(tools, indent=2)}"
    
    # Build full prompt (simple version - no explicit generation instruction)
    prompt = (
        f"<system>\n{system_content}\n</system>\n\n"
        f"{dialogue_history}\n\n"
    )
    return prompt


class HuggingFaceLLMClient:
    """LLM client for HuggingFace models (with optional LoRA support)."""
    
    def __init__(
        self,
        model_name: str,
        base_model: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        load_in_4bit: bool = True,
        **kwargs
    ):
        """
        Initialize HuggingFace model client.
        
        Args:
            model_name: HuggingFace model identifier or path (if using LoRA, this is the LoRA adapter)
            base_model: Base model name if using LoRA (e.g., "Qwen/Qwen2.5-7B-Instruct")
            tokenizer_name: Optional tokenizer source (defaults to model_name). 
                          Useful when adapter was trained with a different tokenizer (e.g., RL model uses SFT tokenizer)
            load_in_4bit: Whether to use 4-bit quantization
            **kwargs: Additional arguments passed to model/tokenizer loading
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            import torch
            # Check if bitsandbytes is available (required for 4-bit quantization)
            try:
                import bitsandbytes
                from transformers import BitsAndBytesConfig
                self._has_bitsandbytes = True
            except ImportError:
                self._has_bitsandbytes = False
        except ImportError as e:
            raise ImportError(
                "HuggingFace dependencies not installed. "
                "Install with: pip install transformers torch peft accelerate"
            ) from e
        
        self.model_name = model_name
        self.base_model = base_model
        self.device = kwargs.get('device', 'auto')
        
        # Load tokenizer: use tokenizer_name if provided, otherwise model_name
        # This allows RL models to use SFT model's tokenizer (which has expanded vocab)
        tokenizer_path = tokenizer_name if tokenizer_name else model_name
        print(f"[HF] Loading tokenizer from: {tokenizer_path}", file=sys.stderr, flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=kwargs.get('trust_remote_code', True)
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"[HF] Tokenizer loaded. Vocab size: {len(self.tokenizer)}", file=sys.stderr, flush=True)
        
        # Load model
        if load_in_4bit:
            if not self._has_bitsandbytes:
                raise ImportError(
                    "4-bit quantization requires bitsandbytes (CUDA only). "
                    "Install with: pip install bitsandbytes, or use --no-4bit flag"
                )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            quantization_config = bnb_config
        else:
            quantization_config = None
        
        if base_model:
            # Load base model + LoRA adapter
            print(f"[HF] Loading base model: {base_model}", file=sys.stderr, flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=quantization_config,
                device_map=self.device,
                trust_remote_code=kwargs.get('trust_remote_code', True),
            )
            # Resize embeddings to match tokenizer before loading LoRA
            model.resize_token_embeddings(len(self.tokenizer))
            print(f"[HF] Loading LoRA adapter: {model_name}", file=sys.stderr, flush=True)
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(
                model,
                model_name,
                device_map=self.device,
            )
        else:
            # Load model directly
            print(f"[HF] Loading model: {model_name}", file=sys.stderr, flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=self.device,
                trust_remote_code=kwargs.get('trust_remote_code', True),
            )
            self.model = model
        
        self.model.eval()
        print(f"[HF] Model loaded successfully. Device: {next(self.model.parameters()).device}", file=sys.stderr, flush=True)
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response using HuggingFace model.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Generation parameters (temperature, max_new_tokens, etc.)
        
        Returns:
            Generated response text
        """
        import torch
        
        # Extract system prompt and tools from messages
        system_parts = []
        conversation_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))
            else:
                conversation_messages.append(msg)
        
        # Combine system messages
        system_prompt = "\n\n".join(system_parts) if system_parts else ""
        
        # Get tools from kwargs (passed by SystemAgent) or fallback to extraction
        tools = kwargs.get('tools', {})
        
        # If tools not provided directly, try to extract from system prompt (legacy/fallback)
        if not tools and "Available Tools:" in system_prompt:
            import json
            import re
            # Extract tools JSON from system prompt
            tools_match = re.search(r'Available Tools:\s*\n(.*?)(?=\n\n|\Z)', system_prompt, re.DOTALL)
            if tools_match:
                try:
                    tools = json.loads(tools_match.group(1))
                except json.JSONDecodeError:
                    pass
        
        # Remove tools section from system prompt (we'll add it back in build_hf_prompt)
        if "Available Tools:" in system_prompt:
            system_prompt = system_prompt.split("Available Tools:")[0].strip()
        
        # Convert conversation messages to Message objects, then to HF format
        message_objects = []
        for msg in conversation_messages:
            role_str = msg.get("role", "")
            if role_str == "user":
                role = MessageRole.USER
            elif role_str == "assistant":
                role = MessageRole.ASSISTANT
            elif role_str == "tool":
                role = MessageRole.TOOL
            else:
                continue  # Skip unknown roles
            message_objects.append(Message(role, msg.get("content", "")))
        
        # Convert to HF format
        dialogue_history = convert_messages_to_hf_format(message_objects)
        
        # Build full prompt
        prompt = build_hf_prompt(system_prompt, tools, dialogue_history)
        # print(f"[HF] Input prompt (last 500 chars):\n...{prompt[-500:]}", file=sys.stderr, flush=True)
        
        # Generation parameters
        max_new_tokens = kwargs.get('max_new_tokens', 512)
        temperature = kwargs.get('temperature', 0.0)
        
        # Tokenize
        print(f"[HF] Tokenizing prompt...", file=sys.stderr, flush=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]
        print(f"[HF] Prompt length: {prompt_len} tokens. Generating (max_new_tokens={max_new_tokens})...", file=sys.stderr, flush=True)
        
        # Define stopping criteria correctly (checking only new tokens)
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class StringStoppingCriteria(StoppingCriteria):
            def __init__(self, tokenizer, stop_strings, prompt_length):
                self.tokenizer = tokenizer
                self.stop_strings = stop_strings
                self.prompt_length = prompt_length
                
            def __call__(self, input_ids, scores, **kwargs):
                # Decode only the NEWLY generated tokens
                # input_ids contains [prompt + new_tokens]
                new_tokens = input_ids[0][self.prompt_length:]
                
                # Optimization: Only decode if we have enough tokens to potentially match
                if len(new_tokens) == 0:
                    return False
                    
                # Decode all new tokens to ensure we catch split words
                # (For very long generations, we might optimize to decode only tail, 
                # but with max 512 tokens, decoding the whole new sequence is fast enough)
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
                
                return any(s in text for s in self.stop_strings)

        stop_strings = ["</action>", "<user>"]
        stopping_criteria = StoppingCriteriaList([
            StringStoppingCriteria(self.tokenizer, stop_strings, prompt_len)
        ])
        
        import time
        start_time = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
        gen_time = time.time() - start_time
        
        # Decode only the generated portion
        gen_ids = output_ids[0, prompt_len:]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        gen_len = len(gen_ids)
        print(f"[HF] Generated {gen_len} tokens in {gen_time:.2f}s ({gen_len/gen_time:.1f} tok/s).", file=sys.stderr, flush=True)
        # print(f"[HF] Raw response:\n{response}\n[HF] End raw response", file=sys.stderr, flush=True)
        
        # Truncate at first <user> tag to prevent hallucinations
        user_tag_idx = response.find("<user>")
        if user_tag_idx != -1:
             print(f"[HF] Truncating response at <user> tag (index {user_tag_idx})", file=sys.stderr, flush=True)
             response = response[:user_tag_idx].strip()
        
        # Truncate after first </action> tag to prevent multiple turn hallucinations
        action_end_marker = "</action>"
        action_end_idx = response.find(action_end_marker)
        if action_end_idx != -1:
             # Include the marker itself
             truncate_at = action_end_idx + len(action_end_marker)
             # Only truncate if there is content after it
             if truncate_at < len(response):
                 print(f"[HF] Truncating response after first action (length {truncate_at})", file=sys.stderr, flush=True)
                 response = response[:truncate_at].strip()
        
        print(f"[HF] Final Response:\n{response}", file=sys.stderr, flush=True)
        return response


class LLMClient:
    def __init__(self, model: str, api_key: str = None, **kwargs):
        # Keep the name the caller asked for (used for the reasoning-effort gate
        # and for log lines); send the Cortex-namespaced name to the API.
        self.requested_model = model
        self.model = resolve_model(model)

        # api_key is accepted for signature compatibility with existing callers
        # (they pass OPENAI_API_KEY) but is ignored: auth is the sfproxy PAT.
        self.client = openai.OpenAI(
            api_key=_sfproxy_token(),
            base_url=SFPROXY_BASE_URL,
            default_headers={"X-Sfproxy-User": _sfproxy_user()},
            **kwargs
        )
        
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make API call and return raw response content
        
        Migrated to Responses API:
        - Separates system messages (→ instructions) from conversation messages (→ input)
        - Uses stateless mode (store=false) for synthetic data generation
        - Disables tool calls (no tools parameter)
        """
        # Separate system messages from conversation messages
        instructions_parts = []
        conversation_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                # Collect all system messages into instructions
                instructions_parts.append(msg.get("content", ""))
            else:
                # Keep user/assistant messages for conversation
                conversation_messages.append({
                    "role": msg.get("role"),
                    "content": msg.get("content", "")
                })
        
        # Combine system messages into single instructions string
        instructions = "\n\n".join(instructions_parts) if instructions_parts else None
        
        # Add cache-busting variation to prevent prompt caching from causing identical outputs
        # This adds a small random variation that doesn't affect the meaning but prevents cache hits
        cache_buster = f"\n\n[Variation: {random.random():.10f}]"
        if instructions:
            instructions = instructions + cache_buster
        else:
            # If no instructions, add to the last user message to prevent caching
            if conversation_messages:
                last_msg = conversation_messages[-1]
                last_msg["content"] = last_msg.get("content", "") + cache_buster
        
        # Prepare input: use conversation messages if available
        # Note: Responses API input can be a list of messages or a string
        input_messages = conversation_messages if conversation_messages else []
        
        # Extract temperature and other kwargs (excluding tools if present)
        api_kwargs = {k: v for k, v in kwargs.items() if k != "tools"}
        
        # Build API call parameters
        api_params = {
            "model": self.model,
            "store": False,  # Stateless mode for synthetic data generation
            **api_kwargs
        }
        
        # Add input (required) - use messages list or empty list as fallback
        if input_messages:
            api_params["input"] = input_messages
        else:
            # Fallback: empty user message if no conversation messages
            api_params["input"] = [{"role": "user", "content": ""}]
        
        # Add instructions only if present (optional parameter)
        if instructions:
            api_params["instructions"] = instructions

        # Gate on the requested name so this keeps firing after the Cortex
        # rename (self.model is now "openai-gpt-5.1", not "gpt-5.1").
        if self.requested_model in ("gpt-5.1", "openai-gpt-5.1"):
            # Effort is the prime suspect for the low eval pass rate; override
            # with MADS_REASONING_EFFORT=medium to A/B it without a code edit.
            api_params["reasoning"] = {
                "effort": os.getenv("MADS_REASONING_EFFORT", "none")
            }
            api_params["text"] = {"verbosity": "low"}

        # Remove 'temperature' from api_params if model is gpt-5-mini
        if self.requested_model in ("gpt-5-mini", "openai-gpt-5-mini"):
            if "temperature" in api_params:
                del api_params["temperature"]
        
        try:
            # Use Responses API
            # Check if responses API is available
            if not hasattr(self.client, 'responses'):
                raise AttributeError("Responses API not available")
            
            # Debug: log temperature if present
            if 'temperature' in api_params:
                import logging
                logging.getLogger(__name__).debug(f"API call with temperature={api_params['temperature']}")
            
            print(f"[OpenAI] Sending request to {self.model}...", file=sys.stderr, flush=True)
            response = self.client.responses.create(**api_params)
            print(f"[OpenAI] Received response ({len(response.output_text)} chars)", file=sys.stderr, flush=True)
            
            # Extract response content - Use output_text convenience property
            # Then strip everything before the first <think> tag
            # This handles cases where the model outputs JSON or other text before the structured format
            
            if hasattr(response, 'output_text') and response.output_text is not None:
                text = response.output_text
                # Find the first occurrence of <think>
                marker = '<think>'
                idx = text.find(marker)
                if idx != -1:
                    # Strip everything before the marker
                    return text[idx:]
                # If marker not found, return as-is (shouldn't happen, but be safe)
                return text
            
            # Fallback: manually extract from output array if output_text not available
            if hasattr(response, 'output') and isinstance(response.output, list):
                texts = []
                for item in response.output:
                    if hasattr(item, 'content') and hasattr(item, 'role') and item.role == 'assistant':
                        for content_item in item.content:
                            if hasattr(content_item, 'text') and content_item.text:
                                texts.append(content_item.text)
                
                if texts:
                    combined_text = '\n'.join(texts)
                    # Strip everything before the first <think>
                    marker = '<think>'
                    idx = combined_text.find(marker)
                    if idx != -1:
                        return combined_text[idx:]
                    return combined_text
            
            # Final fallback: return empty string
            return ""
        except (AttributeError, Exception) as e:
            # Fallback to old chat completions API if responses API not available or fails
            # This allows graceful degradation if Responses API is not yet available
            print(
                f"[OpenAI] Responses API failed ({type(e).__name__}: {str(e)[:200]}); "
                f"retrying via chat.completions",
                file=sys.stderr, flush=True,
            )
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs
                )
                return response.choices[0].message.content
            except Exception as fallback_error:
                # If both APIs fail, raise the original Responses API error
                raise e from fallback_error
