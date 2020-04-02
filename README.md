# charLSTM
Motivation from Andrej Karparthy's blog [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
Trained using pytorch. On 39 books of Mark Twain (from [fullbooks.com](http://www.fullbooks.com/)),

## To genereate text using the trained model:
```
python generate --initial_letters "any inital text you want to give" --num_letters 200
```
Where you can give inital letters and the number of letters you want to generate.

## To train your own model on your own data:
1. Set the model parameters in `model.py` file.
2. Put your data in `data/` directory. You may need to change the data processing code.
3. Train the model using:
```
python train.py
```
4. The model will be saved in the `model/` directory.

The text can have 96 different characters, which include english lowercase and uppercase letters, punctuations from `string.punctuations`, white space character(`' '`) and newline character(`'\n'`). You can change it in the `utils.py` file.
