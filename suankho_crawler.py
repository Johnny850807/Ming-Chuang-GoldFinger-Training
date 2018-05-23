import urllib
import json
from urllib.request import urlopen
from datetime import datetime
from bs4 import BeautifulSoup


def create_url_opener():
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    return opener


def url_to_json(url):
    return json.loads(create_url_opener().open(url).read().decode('utf-8'))


def save_courses(courses):
    with open('suankho_courses.json', 'w+', encoding='utf-8') as f:
        raw = json.dumps(courses, ensure_ascii=False)
        f.write(raw)


def crawl():
    pages = range(165, 232+1)
    with open('suankho_courses.json', 'r', encoding='utf-8') as f:
        all_courses = json.load(f)
    for page in pages:
        try:
            all_courses.extend(crawl_courses_from_page(page))
            if page % 10 == 0:
                save_courses(all_courses)
                print("Courses saved.")
        except:
            print("Error occurs in page ", page)

    print("Courses got: ", len(all_courses), ", task done!")


def crawl_courses_from_page(page):
    link = 'https://suankho.com/courses?page=' + str(page)
    courses = []
    opener = create_url_opener()
    with opener.open(link) as o:
        html = o.read().decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')
    course_cards = soup.find_all('div', attrs={"class": "product-card product-list"})

    for course_card in course_cards:
        courses.append(crawl_course_from_card(course_card))

    print("Page " + str(page) + " courses in stock!")
    return courses


def crawl_course_from_card(course_card):
    course_link_elm = course_card.find('a', attrs={'class': 'product-thumb'})
    course_link = course_link_elm.get('href')
    comments_link = course_link + '/comments'
    comments = crawl_comments_under_comment_link(comments_link)

    # e.g.https://suankho.com/courses/573471361-%E5%9C%8B%E9%9A%9B%E9%87%91%E8%9E%8D%E5%B8%82%E5%A0%B4%E8%88%87%E5%8C%AF%E5%85%8C
    # first find the index of courses, then target the index the number at
    number_index = course_link.index('courses') + 8
    course_id = course_link[number_index:number_index + 9]

    teacher_name = course_link_elm.find('p').string
    subject_name = course_card.find('h3', attrs={'class': 'product-title'}).find('a').string
    return {'course_link': course_link,
            'comments_link': comments_link,
            'course_id': course_id,
            'teacher_name': teacher_name,
            'subject_name': subject_name,
            'comments': comments}


def crawl_comments_under_comment_link(comments_link):
    js = url_to_json(comments_link)
    total_pages = js['page']['total']
    comments = []
    for i in range(1, total_pages + 1):
        page_link = comments_link + '?page=' + str(i)
        js = url_to_json(page_link)
        comments.extend(js['comments'])
    return comments


if __name__ == '__main__':
    print("請呼叫crawl()函數")
