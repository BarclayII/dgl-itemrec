import fileinput

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

for line in fileinput.input():
    book_id, content = line.split('\t')
    if isEnglish(content):
        print(line.rstrip())
