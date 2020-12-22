import recognize as r


words = ["one", "two", "three", "four", "five"]
num_states = 3

reco = r.Recognizer(num_states)
# reco.training(words)

for word in words:
    print(f"\nRecognition of word {word}")
    reco.recognize_word(word)



