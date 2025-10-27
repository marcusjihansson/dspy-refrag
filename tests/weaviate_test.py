import json

import requests
import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.query import MetadataQuery

# Connect to local Weaviate instance
client = weaviate.connect_to_local()

try:
    # Create collection with Ollama embeddings and generative models
    # Ensure idempotency: remove existing collection if it exists
    try:
        if client.collections.exists("Question"):
            client.collections.delete("Question")
    except Exception:
        # If the exists/delete checks fail for any reason, proceed and let create() surface errors
        pass

    questions = client.collections.create(
        name="Question",
        vectorizer_config=Configure.Vectorizer.text2vec_ollama(
            api_endpoint="http://host.docker.internal:11434",
            model="nomic-embed-text:latest",
        ),
        generative_config=Configure.Generative.ollama(
            api_endpoint="http://host.docker.internal:11434",
            model="gemma3:1b",
        ),
    )

    print("Collection created successfully!")

    # Fetch sample data from GitHub
    resp = requests.get(
        "https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json"
    )
    data = json.loads(resp.text)

    # Batch import data
    print(f"Importing {len(data)} objects...")
    with questions.batch.dynamic() as batch:
        for d in data:
            batch.add_object(
                {
                    "answer": d["Answer"],
                    "question": d["Question"],
                    "category": d["Category"],
                }
            )
            if batch.number_errors > 10:
                print("Batch import stopped due to excessive errors.")
                break

    # Check for failed imports
    failed_objects = questions.batch.failed_objects
    if failed_objects:
        print(f"Number of failed imports: {len(failed_objects)}")
        print(f"First failed object: {failed_objects[0]}")
    else:
        print("All objects imported successfully!")

    # Perform semantic search
    print("\n--- Semantic Search Results ---")
    response = questions.query.near_text(
        query="biology", limit=2, return_metadata=MetadataQuery(distance=True)
    )

    for obj in response.objects:
        print(json.dumps(obj.properties, indent=2))
        print(f"Distance: {obj.metadata.distance}\n")

    # Use generative search (RAG) with gemma2
    print("\n--- Generative Search (RAG) ---")
    response = questions.generate.near_text(
        query="biology",
        limit=2,
        single_prompt="Explain this Jeopardy question and answer in simple terms: {question} - {answer}",
    )

    for obj in response.objects:
        print(f"Question: {obj.properties['question']}")
        print(f"Answer: {obj.properties['answer']}")
        print(f"Generated explanation: {obj.generated}")
        print("-" * 50)

except Exception as e:
    print(f"Error: {e}")

finally:
    # Clean up
    client.close()
    print("\nConnection closed.")
