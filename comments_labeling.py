
comments = None
labeled_comments = None

with open('comments.txt', 'r', encoding='utf-8') as f:
    comments = [comment.strip() for comment in f.readlines()]

with open('labeled_comments.txt', 'r+', encoding='utf-8') as f:
    labeled_comments = [comment.split(' ')[0].strip() for comment in f.readlines()]

unlabaled_comments = [comment for comment in list(set(comments) - set(labeled_comments))
                      if len(comment) != 0]

idx = 0
for comment in unlabaled_comments:
    print(comment)
    num = input("請分類：(0) 好過 (1) 不推 (2) 收穫多 (3) 資訊提供 (4) 麻煩 (5) 提問 (6) 垃圾訊息: ")

    with open('labeled_comments.txt', 'a+', encoding='utf-8') as f:
        f.write(comment + " " + num + "\n")
    idx += 1
    print(str(len(unlabaled_comments)-idx) + " comments left.")
print(unlabaled_comments)



