import json
from bs4 import BeautifulSoup


def remove_html_tag(text):
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    return text


def load_courses():
    with open('suankho_courses.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def load_old_comments():
    with open('comments.txt', 'r', encoding='utf-8') as f:
        return [comment.strip() for comment in f.readlines()]


def load_suankho_comments():
    with open('suankho_courses.json', 'r', encoding='utf-8') as f:
        courses = json.load(f)
        comments = []
        for course in courses:
            for comment in course['comments']:
                text = remove_html_tag(comment['body'])
                comments.append(text)
        return comments


def save_reviews_by_lines(reviews, filename):
    with open(filename, 'w+', encoding='utf-8') as fw:
        json.dump(reviews, fw, ensure_ascii=False)


def start_processing():
    old_comments = set(load_old_comments())
    suankho_comments = set(load_suankho_comments())
    comments = set()
    comments.update(old_comments)
    comments.update(suankho_comments)

    print("Comments count: ", len(comments))


if __name__ == '__main__':
    start_processing()
