# ReinforcementLearningWithDemonstration

This project investigates how to learn a policy with a Reinforcement Learning Algorithm that has access to demonstrations. You can find the report that I have done [here](./ActiveLearningFromDemonstrations.pdf).

### Algorithms
Three algorithms are implemented in this repository:
- [Learning from Limited Demonstrations](http://ncfrn.mcgill.ca/members/pubs/kimnips13.pdf) from Kim et al.. In linear setting.
- [Deep Q-learning from Qemonstrations](https://arxiv.org/pdf/1704.03732.pdf) from Hester et al.. In tabular setting.
- [Reinforcement learning from imperfect demonstrations](https://arxiv.org/pdf/1802.05313.pdf) from Gao et al.. In neural network setting.

Please find the code in the [algorithms](./algorithms) folder.

### Environment
A GridWorl environment has been implemented from the package [RL-Berry](https://github.com/rlberry-py/rlberry).

Please find the code in the [simulators](./simulators) folder.

### Experiments
To play with the algorithms, you can use the [experiments](./experiments) folder where jupyter notebooks have been made to try the different algorithms.

<table style="width:100%; table-layout:fixed;">
	<tr>
		<td><img width="1000px" src="gifs/DQfD_demo.gif"></td>
	</tr>
	<tr>
		<td>Example of the DQfD notebook</td>
	</tr>
</table>
