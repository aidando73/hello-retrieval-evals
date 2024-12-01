import datasets

# https://huggingface.co/datasets/hotpotqa/hotpot_qa?row=16
# https://arxiv.org/pdf/1606.05250
dataset = datasets.load_dataset("rajpurkar/squad")

print(dataset["train"][500])

row1 = {
    "id": "5733bed24776f41900661188",
    "title": "University_of_Notre_Dame",
    "context": "The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome). Its main seminary, Moreau Seminary, is located on the campus across St. Joseph lake from the Main Building. Old College, the oldest building on campus and located near the shore of St. Mary lake, houses undergraduate seminarians. Retired priests and brothers reside in Fatima House (a former retreat center), Holy Cross House, as well as Columba Hall near the Grotto. The university through the Moreau Seminary has ties to theologian Frederick Buechner. While not Catholic, Buechner has praised writers from Notre Dame and Moreau Seminary created a Buechner Prize for Preaching.",
    "question": "Where is the headquarters of the Congregation of the Holy Cross?",
    "answers": {"text": ["Rome"], "answer_start": [119]},
}

row2 = {
    "id": "56be99b53aeaaa14008c913f",
    "title": "Beyoncé",
    "context": "In 2011, documents obtained by WikiLeaks revealed that Beyoncé was one of many entertainers who performed for the family of Libyan ruler Muammar Gaddafi. Rolling Stone reported that the music industry was urging them to return the money they earned for the concerts; a spokesperson for Beyoncé later confirmed to The Huffington Post that she donated the money to the Clinton Bush Haiti Fund. Later that year she became the first solo female artist to headline the main Pyramid stage at the 2011 Glastonbury Festival in over twenty years, and was named the highest-paid performer in the world per minute.",
    "question": "Who did Beyonce donate the money to earned from her shows?",
    "answers": {"text": ["Clinton Bush Haiti Fund"], "answer_start": [367]},
}