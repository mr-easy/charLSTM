# charLSTM
Motivation from Andrej Karparthy's blog [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
Trained using PyTorch. On 39 books of Mark Twain (from [fullbooks.com](http://www.fullbooks.com/)).

## Some generated text
* [He said this in a very low tone.] When he was asleep and said that he had not been less delivered on a sharp hand and said, "What hotel you going to stay? We can't be so suddlest-and it can't take pass and confessed.
* [he does not allow him to ]go to work and stay in the season. The most of them were the result of the law, the comportance of the season of milk are pleasantly resembles his present machinery, who can play now.
* [the begger came to him ]and said that the stranger was not a mere remark for a spectator and not a more pretty difficult one. It was a mere superantial instinct. We could not see that it was what he was before.
* [I told her not to] come and dart gradually drift behind it. I have seen nothing to compare with him. I could not have said anything of him, but the most of them was a student of the palace to the stage of St. Nicholas.

Don't try to make sense out of them.

## To generate text using the trained model
```
python generate.py --initial_letters "any initial text you want to give" --num_letters 200
```
Where you can give initial letters and the number of letters you want to generate.

## To train your own model on your own data
1. Set the model parameters in `model.py` file.
2. Put your data in the `data/` directory. You may need to change the data processing code.
3. Train the model using:
```
python train.py
```
4. The model will be saved in the `model/` directory. OVERWRITING the already saved model, you can change the model save path name at the end of `train.py` file. Also need to change in `generate.py` file.

The text can have 96 different characters, which include English lowercase and uppercase letters, punctuations from `string.punctuations`, white space character(`' '`) and newline character(`'\n'`). You can change it in the `utils.py` file.
