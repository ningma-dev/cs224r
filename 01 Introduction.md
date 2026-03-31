深度强化学习解决序列决策问题
序列性：与supervised learning不同，RL没有假设数据独立同分布(i.i.d.)。agent的行为会影响未来决策

术语
- 状态 / state / $s_t$：世界的完整描述
- 观测 / observation / $o_t$：agent在时刻 $t$ 看到的信息（世界的局部）
- 动作 / action / $a_t$：agent在时可 $t$ 做的决定
- 轨迹 / trajectory / $\tau$ ：状态动作对的序列 $(s_1​,a_1​,s_2​,a_2​,…,s_T​,a_T​)$
- 策略 / policy / $\pi$
- 奖励 / reward / $r(s, a)$

RL的目标是学习一个参数化策略 $\pi_{\theta}(a|s)$，以最大化预期回报的总和
- 轨迹的概率分布 $p_{\theta}(\tau)$ 
	$$p_{\theta}(s_1, a_1, ..., s_T, a_T) = p(s_1) \prod^T_{t=1} \pi_{\theta} (a_t | s_t) p(s_{t+1} | s_t, a_t)$$
	
- 目标函数 $J(\theta)$
	$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} [  \sum^T_{t=1}r(s_t, a_t)]$$