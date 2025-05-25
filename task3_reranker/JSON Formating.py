import json

# Load the training data (update path as needed)
with open('training13b.json', 'r') as f:
    data = json.load(f)

# Let's check the data structure and parse relevant fields
questions = data["questions"]

# We will store question-snippet pairs here
question_snippet_pairs = []

for question in questions:
    question_text = question["body"]
    snippets = question["snippets"]
    
    for snippet in snippets:
        snippet_text = snippet["text"]
        snippet_document = snippet["document"]
        
        # Save question-snippet pair along with document info
        question_snippet_pairs.append({
            "question": question_text,
            "snippet": snippet_text,
            "document": snippet_document
        })

# Let's take a look at the first 3 pairs
print(question_snippet_pairs[:3])
