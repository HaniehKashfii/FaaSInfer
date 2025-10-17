"""
Chat bot example using FaaSInfer.

Demonstrates multi-turn conversation with context management.
"""

import asyncio
from faasinfer import FaaSInfer, FaaSInferConfig, ModelConfig
from faasinfer.config import BatchingPolicy
from faasinfer.utils.logging import setup_logging


async def main():
    """Run chat bot."""
    setup_logging(log_level="INFO")
    
    # Configure system
    config = FaaSInferConfig()
    config.models = [
        ModelConfig(
            model_id="llama-2-13b-chat",
            model_name="meta-llama/Llama-2-13b-chat-hf",
            model_size_gb=26.0,
            num_parameters_b=13.0,
            tensor_parallel_size=2,
            max_context_length=2048,
            batching_policy=BatchingPolicy.VLLM_CONTINUOUS,
        ),
    ]
    
    # Initialize
    print("Initializing chat bot...")
    system = FaaSInfer(config)
    await system.initialize()
    print("Chat bot ready! Type 'quit' to exit.\n")
    
    # Conversation loop
    conversation_history = []
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Build prompt with history
        conversation_history.append(f"User: {user_input}")
        prompt = "\n".join(conversation_history) + "\nAssistant:"
        
        # Generate response
        print("Assistant: ", end="", flush=True)
        response_text = ""
        
        async for response in system.generate(
            prompt=prompt,
            model_id="llama-2-13b-chat",
            max_new_tokens=200,
            temperature=0.7,
        ):
            print(response.text[len(response_text):], end="", flush=True)
            response_text = response.text
            
            if response.finish_reason:
                break
        
        print("\n")
        conversation_history.append(f"Assistant: {response_text}")
        
        # Keep last 4 turns to manage context
        if len(conversation_history) > 8:
            conversation_history = conversation_history[-8:]
    
    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())