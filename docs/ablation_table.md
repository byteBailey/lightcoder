# Ablation comparison (same 20-instance subset, Qwen3_5-27B)

```
配置	样本数	解决	解决率%	空patch	错误	平均调用	平均用时s	平均patch字符
baseline	20	8/20	40.0	1	0	34.4	302	1318
no_compression	20	8/20	40.0	0	0	28.4	239	1248
no_subagent	20	8/20	40.0	2	0	31.5	252	1061
prompt_minimal	20	8/20	40.0	2	0	29.6	222	1068
prompt_react	20	6/20	30.0	0	0	31.4	269	1648
```
