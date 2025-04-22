1. RPU 平台识别的过程
2. 把 RPU 上的模型加载到 vllm 中

```bash
skip_tokenizer_init=False,
tokenizer_mode=auto, 
revision=None, 
override_neuron_config=None, 
tokenizer_revision=None, 
trust_remote_code=False, 
dtype=torch.bfloat16, 
max_seq_len=131072, 
download_dir=None, 
load_format=LoadFormat.AUTO, 
tensor_parallel_size=1, 
pipeline_parallel_size=1, 
disable_custom_all_reduce=False, 
quantization=None, 
enforce_eager=False, 
kv_cache_dtype=auto,  
device_config=cuda, 
decoding_config=DecodingConfig(guided_decoding_backend='auto', reasoning_backend=None), observability_config=ObservabilityConfig(show_hidden_metrics=False, otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), 
seed=None, 
served_model_name=/home/ubuntuhx/xirui.hao/Models/deepseek-1.5B/DeepSeek-R1-Distill-Qwen-1.5B, 
num_scheduler_steps=1, 
multi_step_stream_outputs=True, 
enable_prefix_caching=True, 
chunked_prefill_enabled=True, 
use_async_output_proc=True, 
disable_mm_preprocessor_cache=False, 
mm_processor_kwargs=None, 
pooler_config=None, compilation_config={"level":3,"custom_ops":["none"],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output"],"use_inductor":true,"compile_sizes":[],"use_cudagraph":true,"cudagraph_num_of_warmups":1
}
```
