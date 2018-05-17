import urllib
import json
from urllib.request import urlopen
from datetime import datetime
from bs4 import BeautifulSoup
from multiprocessing import Pool



def parse_date(time):  # 2016-09-05 00:58:56
    return datetime.strptime(time, '%Y-%m-%d %H:%M:%S')


def url_to_json(url):
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    return json.loads(opener.open(url).read().decode('utf-8'))


def get_all_comments_from_time(comment_id, latest_comments_time):
    comments = set()
    latest = latest_comments_time
    while True:
        url = 'http://goldenfinger.mcu.mingtsay.tech/fetch/comment/' + comment_id + '/' + \
              latest.strftime('%Y-%m-%d_%H:%M:%S')
        dataset = url_to_json(url)[1]
        latest = min(parse_date(item['timestamp']) for item in dataset)
        comments.update([item['message'] for item in dataset])
        if len(dataset) < 5:
            break
    return list(comments)


def crawl_course(course_elm):
    fields = course_elm.find_all('h2', attrs={'class': 'box-header-description'})
    course_number = fields[0].string
    course_name = fields[1].string
    teacher_name = fields[2].string
    course_category = fields[3].string

    panel_response_elm = course_elm.find('div', attrs={'class': 'panel-resp'})
    comments = []

    if panel_response_elm:  # there is no response element only if the course has no comments
        course_comment_id = panel_response_elm.get('data-form-id')

        # we get the latest comment time so that the api can start scanning all comments from the latest.
        latest_comments_time = max([parse_date(datetime_elm.get('title')) for datetime_elm
                                    in course_elm.find_all('span', attrs={'class': 'comment-timestamp'})])
        comments = get_all_comments_from_time(course_comment_id, latest_comments_time)

    return {"course_number": course_number,
            "course_name": course_name,
            "teacher_name": teacher_name,
            "course_category": course_category,
            "comments": comments}


def crawl():
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    html = None

    with opener.open('http://goldenfinger.mcu.mingtsay.tech/') as o:
        html = o.read().decode('utf-8')

    soup = BeautifulSoup(html, 'html.parser')
    course_elms = soup.find_all('div', attrs={'class': 'row course-row'})
    courses = [crawl_course(course_elm) for course_elm in course_elms]
    comments = ['\n'.join(courses['comments']) for courses in courses]

    with open('course.json', 'w+', encoding='utf-8') as f:
        raw = json.dumps(courses, ensure_ascii=False)
        f.write(raw)

    with open('comments.txt', 'w+', encoding='utf-8') as f:
        raw = ''.join(comments)
        f.write(raw)
    print("Done!")


crawl()
