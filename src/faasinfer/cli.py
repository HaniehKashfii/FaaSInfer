"""
CLI interface for FaaSInfer.
"""

import asyncio
import click
import sys
import json
from pathlib import Path

from faasinfer import FaaSInfer, FaaSInferConfig, ModelConfig
from faasinfer.utils.logging import setup_logging
from faasinfer.config import BatchingPolicy


@click.group()
@click.option('--log-level', default='INFO', help='Logging level')
@click.pass_context
def cli(ctx, log_level):
    """FaaSInfer CLI - Low-Latency Serverless LLM Inference."""
    setup_logging(log_level=log_level)
    ctx.ensure_object(dict)


@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.option('--host', default='0.0.0.0', help='API host')
@click.option('--port', default=8000, help='API port')
def serve(config, host, port):
    """Start FaaSInfer API server."""
    click.echo("Starting FaaSInfer API server...")
    
    # Load configuration
    if config:
        with open(config, 'r') as f:
            config_dict = json.load(f)
        faasinfer_config = FaaSInferConfig.from_dict(config_dict)
    else:
        # Use default configuration with example models
        faasinfer_config = FaaSInferConfig()
        faasinfer_config.models = [
            ModelConfig(
                model_id="opt-6.7b",
                model_name="facebook/opt-6.7b",
                model_size_gb=13.0,
                num_parameters_b=6.7,
                batching_policy=BatchingPolicy.VLLM_CONTINUOUS,
            ),
        ]
    
    # Start API server
    from faasinfer.gateway.api import app
    import uvicorn
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


@cli.command()
@click.argument('prompt')
@click.option('--model', default=None, help='Model ID')
@click.option('--max-tokens', default=100, help='Maximum tokens to generate')
@click.option('--temperature', default=1.0, help='Sampling temperature')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
def generate(prompt, model, max_tokens, temperature, config):
    """Generate text from prompt."""
    
    async def run():
        # Load configuration
        if config:
            with open(config, 'r') as f:
                config_dict = json.load(f)
            faasinfer_config = FaaSInferConfig.from_dict(config_dict)
        else:
            faasinfer_config = FaaSInferConfig()
            faasinfer_config.models = [
                ModelConfig(
                    model_id="opt-6.7b",
                    model_name="facebook/opt-6.7b",
                    model_size_gb=13.0,
                    num_parameters_b=6.7,
                ),
            ]
        
        # Initialize system
        system = FaaSInfer(faasinfer_config)
        await system.initialize()
        
        # Generate
        click.echo(f"Prompt: {prompt}\n")
        click.echo("Response: ", nl=False)
        
        async for response in system.generate(
            prompt=prompt,
            model_id=model,
            max_new_tokens=max_tokens,
            temperature=temperature,
        ):
            click.echo(response.text, nl=False)
            sys.stdout.flush()
        
        click.echo("\n")
        
        # Shutdown
        await system.shutdown()
    
    asyncio.run(run())


@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Config file path')
def status(config):
    """Get system status."""
    
    async def run():
        # Load configuration
        if config:
            with open(config, 'r') as f:
                config_dict = json.load(f)
            faasinfer_config = FaaSInferConfig.from_dict(config_dict)
        else:
            faasinfer_config = FaaSInferConfig()
        
        # Initialize system
        system = FaaSInfer(faasinfer_config)
        await system.initialize()
        
        # Get status
        status_dict = await system.get_status()
        
        click.echo(json.dumps(status_dict, indent=2))
        
        # Shutdown
        await system.shutdown()
    
    asyncio.run(run())


@cli.command()
@click.argument('checkpoint_path')
@click.argument('model_id')
@click.option('--tensor-parallel', default=1, help='Tensor parallel size')
@click.option('--pipeline-parallel', default=1, help='Pipeline parallel size')
def convert(checkpoint_path, model_id, tensor_parallel, pipeline_parallel):
    """Convert checkpoint to optimized format."""
    from faasinfer.storage.converter import CheckpointConverter
    
    click.echo(f"Converting checkpoint for {model_id}...")
    
    converter = CheckpointConverter()
    result = converter.convert(
        checkpoint_path=checkpoint_path,
        model_id=model_id,
        tensor_parallel_size=tensor_parallel,
        pipeline_parallel_size=pipeline_parallel,
    )
    
    if result.success:
        click.echo(f"✓ Conversion successful")
        click.echo(f"  Output path: {result.optimized_path}")
        click.echo(f"  Total size: {result.total_size / 1e9:.2f} GB")
        click.echo(f"  Partitions: {result.num_partitions}")
        click.echo(f"  Checksum: {result.checksum[:16]}...")
    else:
        click.echo(f"✗ Conversion failed: {result.error}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('model_id')
def inspect(model_id):
    """Inspect converted checkpoint."""
    from faasinfer.storage.converter import CheckpointConverter
    
    converter = CheckpointConverter()
    metadata = converter.inspect(model_id)
    
    if metadata:
        click.echo(json.dumps(metadata, indent=2))
    else:
        click.echo(f"Model {model_id} not found", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()