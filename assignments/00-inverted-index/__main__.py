from collections import defaultdict

DOCUMENTS = dict(
    1="twinkle, twinkle, little star,",
    2="how I wonder what you are.",
    3="up above the world so high,",
    4="like a diamond in the sky.",
    5="when the blazing sun is gone,",
    6="when he nothing shines upon,",
    7="then you show your little light,",
    8="twinkle, twinkle, all the night.",
)

word_index = defaultdict(lambda: set())

def clean_data(data: str):
    # Strip away commas and periods from the data.
    data = data.replace(",", "")
    data = data.replace(".", "")

    # Clean away trailing whitespace.
    data = data.strip()

    return data

def add_document(document_id, data):
    data_cleaned = clean_data(data)

    for word in data_cleaned.split(" "):
        word_index[word].add(document_id)

def print_index():
    for word, occurences in word_index.items():
        print(f"[{len(occurences)}] {word} - {', '.join(occurences)}")

if __name__ == "__main__":
    for document_id, data in DOCUMENTS.items():
        add_document(document_id, data)
    print_index()
