import os
import re

import unidecode

angle_brackets_enclosed_global = re.compile(r'<.*?>')
# default is to remove the following characters completely in all contexts and replace with spaces:
# parentheses, square brackets, braces, colon, semicolon, question mark, exclamation point,
# asterisk, caret, pipe, tilde, underscore, backslash
remove_chars_regex_global = re.compile(r'[\(\:\)\?\*;\!\[\]_\~\{\}\^\|\\]')
number_regex = re.compile(r'(?P<word>\b\w+?\b)#')

non_numeric_comma_pattern = re.compile(r',(?!\^)')
non_numeric_period_pattern = re.compile(r'\.(?!\^)')
non_numeric_dash_pattern = re.compile(r'\-(?!\^)')
numeric_foot_pattern = re.compile(r"(?P<meas>\^+)'")
numeric_inch_pattern = re.compile(r'(?P<meas>\^+)"')

multiple_commas_pattern = re.compile(r',(\s*,)+')

def unescape(data):
    return data.replace("&quot;", '"').replace("&quot", '"').replace("&amp;", "and").replace("&amp", "and")

def striphtml(data):
    try:
        return angle_brackets_enclosed_global.sub(' ', data)
    except TypeError:
        try:
            return angle_brackets_enclosed_global.sub(' ', str(data))
        except TypeError:
            return ' '

def analyze(text):
    return clean_text(text)

def clean_text(line, elim_regex=remove_chars_regex_global):
    '''
    :param line:
    :param elim_regex: a configurable parameter giving the list of all ascii characters which are to be replaced by space.
    :return:
    '''
    line = striphtml(line)
    # the following is NOT URL unescaping: it replaces the html character entities for double-quote and ampersand
    # with an actual double-quote or the word "and"
    line = unescape(line)
    line = line.replace("`", "'")
    line = line.replace("w/", "with ")
    line = line.replace("|", " , ")
    # unicode folding
    line = unidecode.unidecode(line)
    line = re.sub(elim_regex, r" ", line)
    # replace expressions such as "word#" "word #" with "word number"
    line = re.sub(number_regex, r'\g<word> number', line)
    # insert spaces around equals sign
    line = line.replace("=", " = ")

    # replace digits with a standard symbol caret, which was replaced with space earlier
    line = re.sub(r"[\d]", "^", line)
    # insert spaces around periods, commas unless they are surrounded by digits
    line = re.sub(non_numeric_comma_pattern, " , ", line)
    line = re.sub(non_numeric_period_pattern, " . ", line)
    # for a dash that is not followed by a digit, replace with " , ":
    line = re.sub(non_numeric_dash_pattern, " , ", line)
    # replace single quote immediately following digit with "foot" and double quote immediately
    # following digit with "inch"
    line = re.sub(numeric_foot_pattern, r'\g<meas> inches', line)
    line = re.sub(numeric_inch_pattern, r'\g<meas> feet', line)
    line = re.sub(r'[\'\"]', '', line)
    line = re.sub("\$", " dollars ", line)
    line = line.lower()
    line = re.sub(multiple_commas_pattern, ",", line)
    return line

def get_readme():
    parent = os.path.dirname(os.path.dirname(__file__))

    with open(os.path.join(parent, 'README.md')) as f:
        content = f.read()

    return content

if __name__ == '__main__':
    clean_text("3'")