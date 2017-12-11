# Tamil name generation with RNN

This is an experiment to generate names using deep neural networks. The dataset used is collected from the webiste [peyar.in](http://peyar.in). The DNN consists of a single RNN in seq2seq configuration, which takes a character, and a random seed to generate the next character. 

Overview of the process:-
- Take a name and its category (boy or girl)
- Convert it to tace16 encoding. It is possible to use unicode directly, but tace16 seemed more natural.
- Run the RNN over the name, letter by letter so as to make the network learn what letters can occur together. 
