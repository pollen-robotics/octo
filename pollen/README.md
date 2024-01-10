# TO TRY

## Binarize gripper motion
- In recordings, when I grasp an object, the angle does not go to zero because the object is in the way. But we want the policy to fully close the gripper

## Understand reward system ?

## keep images aspect ratio
- do not resize them to a 256x256 square

## Longer history window ?
- is 2 right now, maybe 5 ?

## Activate proprioception ?
- is this really disabled ? Check with the authors
- It is definitely **enabled**

## Check input normalization when training

## Train without proprio ? 
"Adding Proprioceptive Inputs: resulting policies seemed generally worse, potentially due to a strong correlation between states and future actions"


# Remarks
- removing input state normalization seems to help (?)