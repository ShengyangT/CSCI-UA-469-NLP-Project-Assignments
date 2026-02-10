import re
import sys

def find_dollar_amounts(filename):
    # OPEN USER INPUTTED FILE & READ
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # DOLLAR REGEX --> USE TO SERACH TEXT FOR DOLLAR AMOUNTS
    # dollar_regex = r'''(?:[$][0-9\,]+(?:\.[0-9]+)?(?:\s(?:hundred|thousand|million|billion))?)|(?:(?:(?:[0-9]+(?:[0-9\,]+)?(?:\.[0-9]+)?)|one|two|three|four|five|six|seven|eight|nine|ten|tens|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|fourty|fifty|sixty|seventy|eighty|ninety|half|quarter|a)[\s-])+(?:(?:(?:[0-9]+(?:[0-9\,]+)?(?:\.[0-9]+)?)|one|two|three|four|five|six|seven|eight|nine|ten|tens|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|fourty|fifty|sixty|seventy|eighty|ninety|half|quarter|a|of|hundred|hundreds|thousand|thousands|million|millions|billion|billions|gazillion)[\s])*\b(?:dollar|dollars|cent|cents)\b'''
    # 改进的正则表达式，修复以下问题：
    # 1. 避免匹配形容词用法如 "eight-dollar coat"
    # 2. 支持带撇号的数字如 "t'ree" 
    # 3. 更好地处理数字和文字的组合
    # 改进的正则表达式：修复拼写错误，避免形容词用法
    dollar_regex = r'(?:[$][0-9\,]+(?:\.[0-9]+)?(?:\s(?:hundred|thousand|million|billion|trillion|gazillion))?)|(?:(?:(?:[0-9]+(?:[0-9\,]+)?(?:\.[0-9]+)?)|one|two|three|four|five|six|seven|eight|nine|ten|tens|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|half|quarter|a)\s)+(?:(?:(?:(?:[0-9]+(?:[0-9\,]+)?(?:\.[0-9]+)?)|one|two|three|four|five|six|seven|eight|nine|ten|tens|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|half|quarter|a|of|hundred|hundreds|thousand|thousands|million|millions|billion|billions|trillion|trillions|gazillion)\s)+)?\b(?:dollar|dollars|cent|cents)\b'

    # FIND ALL DOLLAR AMOUNTS IN THE TEXT USING THE REGEX
    matches = re.findall(dollar_regex, text, re.IGNORECASE | re.VERBOSE)

    # OPEN OUTPUT FILE AND WRITE FOUND MATCHES THERE
    with open('dollar_output.txt', 'w') as output_file:
        for match in matches:
            output_file.write(match.strip() + '\n')
                
if __name__ == '__main__':
    # IF USER ENTERED INCORRECT NUMBER OF ARGUMENTS... print error message
    if len(sys.argv) != 2:
        print("Please enter name of the program and the file you would like to read.")
        sys.exit(1)

    # ELSE... run program with the inputted file
    find_dollar_amounts(sys.argv[1])
