#RL-ExpectedSarsa-MountainCar

## Member
| [Yuqian Cao (yuqian5:1558141)](https://github.com/yuqian5) | [Junyao Cui (junyao1:1527740)](https://github.com/lospub) |
| :---: |:---:|
|[![](https://avatars2.githubusercontent.com/u/28016308?s=400&v=4)]()|[![](https://avatars2.githubusercontent.com/u/46499400?s=400&v=4)]()|

## Code Explain
* agent.py
* * Contains the code for the Expected SARSA agent
* MontainCarEnv.py
* * Contains the code for the Mountain Car Environment
* q3.py
* * Produce the chart for Part 3
* q4.py 
* * Produce the chart for Part 4
* utility.py
* * Contain utility functions for the agent
* tiles3.py
* * Copy from Rich Sutton for tile coding

## Part 3
##### Agent parameter 
|ALPHA|EPSILON|NUM_TILE|NUM_TILING|
|---|---|---|---|
|0.1|0.0|8|8|

##### 50 Run average
[![](https://github.com/yuqian5/RL-ExpectedSarsa-MountainCar/blob/master/doc/0.1%5B16:44%5D.png)]()

##### Mean and STD. ERROR
|MEAN|STD. ERROR|
|653.3927|47.1082428602|

## Part 4
##### Agent parameter 
|ALPHA|EPSILON|NUM_TILE|NUM_TILING|
|---|---|---|---|
|0.5|0.1|4|32|

##### 50 Run average
[![](https://github.com/yuqian5/RL-ExpectedSarsa-MountainCar/blob/master/doc/0.5%5B16:49%5D.png)]()

##### Mean and STD. ERROR
|MEAN|STD. ERROR|
|242.4813|11.6725668508|

##### Graph Compare
###### Legend
|ALPHA|EPSILON|NUM_TILE|NUM_TILING|Line Colour|
|---|---|---|---|---|
|0.1|0.0|8|8|Blue|
|0.2|0.1|4|32|Green|
|0.5|0.1|4|32|Red|
[![](https://github.com/yuqian5/RL-ExpectedSarsa-MountainCar/blob/master/doc/compare%5B16:50%5D.png)]()

