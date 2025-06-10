from flask import Flask, render_template, request, redirect, url_for
import requests
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')


def flatten_categories(categories):
    """
    Categories can be nested lists; flatten to a list of strings.
    """
    flattened = []
    for c in categories:
        if isinstance(c, list):
            flattened.extend(flatten_categories(c))
        else:
            flattened.append(c)
    return flattened


def build_subject_query(subjects):
    quoted = [f'subject:"{sub.strip()}"' for sub in subjects if sub.strip()]
    query = " OR ".join(quoted)
    return query


def fetch_books_from_google(query, max_results=20):
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}"
    try:
        res = requests.get(url)
        res.raise_for_status()
        items = res.json().get('items', [])
    except Exception:
        items = []

    books = []
    for item in items:
        info = item.get('volumeInfo', {})
        title = info.get('title', 'Unknown Title')
        authors = ", ".join(info.get('authors', []))
        description = info.get('description', '') or ''
        image_url = info.get('imageLinks', {}).get('thumbnail', '')
        average_rating = info.get('averageRating', 0)
        ratings_count = info.get('ratingsCount', 0)
        categories = info.get('categories', [])

        if description.strip():
            books.append({
                'title': title,
                'authors': authors,
                'description': description,
                'image_url': image_url,
                'average_rating': average_rating,
                'ratings_count': ratings_count,
                'categories': categories
            })
    return books


def extract_keywords_tfidf(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    try:
        vectorizer.fit([text])
    except ValueError:
        return []
    return vectorizer.get_feature_names_out()


def recommend_books_dynamic(seed_book, top_k=5, similarity_threshold=0.1, min_ratings=0):
    categories = seed_book.get('categories', [])
    if categories:
        categories = flatten_categories(categories)
    else:
        combined_text = (seed_book.get('title', '') + " " + seed_book.get('description', '')).lower()
        categories = extract_keywords_tfidf(combined_text, top_n=7)

    title_keywords = seed_book.get('title', '').split()
    search_terms = list(set(categories + title_keywords))

    if not search_terms:
        return []

    subject_query = build_subject_query(search_terms)
    print(f"Subject query: {subject_query}")

    candidates = fetch_books_from_google(subject_query, max_results=40)
    print(f"Candidates with subjects: {len(candidates)}")

    if not candidates:
        fallback_query = ' '.join(search_terms)
        print(f"Fallback query: {fallback_query}")
        candidates = fetch_books_from_google(fallback_query, max_results=40)
        print(f"Candidates with fallback: {len(candidates)}")

    candidates = [b for b in candidates if b['title'].lower() != seed_book['title'].lower()]

    if not candidates:
        return []

    all_desc = [seed_book['description']] + [b['description'] for b in candidates]
    embeddings = model.encode(all_desc)
    seed_emb = embeddings[0].reshape(1, -1)
    candidate_embs = embeddings[1:]

    similarity_scores = cosine_similarity(seed_emb, candidate_embs).flatten()

    scored_books = []
    for i, book in enumerate(candidates):
        if book['ratings_count'] < min_ratings:
            continue
        popularity_score = math.log(1 + book['ratings_count']) * book['average_rating'] if book['average_rating'] else 0
        combined_score = similarity_scores[i] * (1 + popularity_score)

        if similarity_scores[i] >= similarity_threshold:
            scored_books.append((combined_score, book))

    scored_books.sort(key=lambda x: x[0], reverse=True)
    return [b for _, b in scored_books[:top_k]]



@app.route('/')
def index():
    return render_template('index.html')  # Only form, no results

@app.route('/search')
def search():
    query = request.args.get('query', '').strip()
    results = []
    if query:
        results = fetch_books_from_google(f'intitle:"{query}"', max_results=10)
    return render_template('search-results.html', results=results, query=query)


@app.route('/book/<title>')
def book_detail(title):
    title = title.replace("_", " ")
    books = fetch_books_from_google(f'intitle:"{title}"', max_results=1)
    if not books:
        return redirect(url_for('index'))

    seed_book = books[0]
    recommendations = recommend_books_dynamic(seed_book, top_k=5)

    return render_template('book.html', book=seed_book, recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
