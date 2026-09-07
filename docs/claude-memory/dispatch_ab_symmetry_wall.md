---
name: dispatch-ab-symmetry-wall
description: 副本派发策略A/B实验的核心发现——对称部署+自由分发下,加权随机会把两副本(内存和SLO违规率)都均衡掉,导致memory_inverse/random/slo_weighted三策略goodput全程接近,无论低载还是过载。想区分策略需要打破对称性。
metadata:
  node_type: memory
  type: project
---

在 xLLM 双副本(Qwen3-8B#0/#1,对称部署)上做副本派发策略 A/B,用匿名化生产负载 trace + 自然分发(不硬指定副本),反复实测得到一个**反直觉但是数学必然**的核心结论:

## 核心发现:自由分发 → 副本对称 → 策略无从区分

三策略权重公式(`request_metric_aggregator.cpp:357-365`):
- memory_inverse: `raw = 1/used_pages`
- random: `raw = 1.0`
- slo_weighted: `raw = (1/used_pages) × 1/(1+α·viol_rate)`

**只要请求由派发器自由分发,加权随机就把两副本均衡掉**:
- 两副本 used_pages 差始终 <5%(实测 long_qps1: 524/22000;hetero: 1150/22000),权重恒 50/50(std<0.02)
- **连 SLO 违规率也被均衡**:过载(qps24,SLO达成率跌到23%)时两副本同步违规,`1/(1+α·viol)` 对两边惩罚相同 → 还是 50/50
- 结果:三策略 goodput 全程接近(qps18 膝点: slo10.87/mem10.56/random10.14,差<8%;qps24 过载: mem4.23≈slo4.30)

**关键机制**:slo_weighted 的 SLO 反馈项要生效,需要"副本间 viol **不对称**";但自由分发让两副本收到统计同质的流量 → viol 也对称 → 反馈项对两边等值 → 退化成 memory_inverse。同理 memory_inverse 的内存反比也因两副本内存对称而退化成轮询。

**低负载时三策略数学上等价**:viol=0 → slo_weighted 的 `1/(1+0)=1` → 完全等于 memory_inverse。

## 实验设计教训(踩过的弯路)

1. **不要为了"制造差异"去硬指定副本**(force-replica header)——那是人为不对称,不真实,结论无意义(用户否掉了)。
2. **不要纠结"均衡不均衡/hot-share是否0.5"**——均衡不是衡量标准,**goodput 才是**。memory_inverse 想均衡就让它均衡,直接比 goodput 谁高即可(用户点醒)。
3. **"不硬指定 + 真实trace + 想区分派发策略"三者内在矛盾**:自由分发必然均衡化。要区分策略,只能打破对称性,真实来源是:①会话亲和(prefix cache,同会话请求想去同副本)②副本异构(容量/能力不同)③真实背景负载不均。这些才是派发策略真正有价值的场景。
4. **膝点/QPS 选择**:qps 太低(long qps1 并发其实有4.17但仍对称)测不出;过载(qps24)也因同步违规测不出。不是负载高低问题,是**对称性**问题。

## 数据集与方法(供复现)

- trace: `/path/to/anonymized_trace.csv`(匿名化生产负载,约85万条,`time,model,input_tokens,output_tokens`)
- 恒定 λ 泊松到达(丢弃原始时间戳),ignore_eos+max_tokens=trace_output 复现 decode 长度
- 主指标 goodput(满足 TTFT<1s 且 TPOT<50ms 的成功吞吐),取自 server sample 事件
- harness: `bench/replica_dispatch_ab/`(trace_loader.py / run.py / analyze.py / launch_matrix.sh)
- 关联: [[build-and-run]] 的 PhyPagePool 脉冲坑(实验极不稳定的根源)、[[container-pid-isolation]]
