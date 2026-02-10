import re
import sys

def find_phone_numbers(filename):
    # OPEN USER INPUTTED FILE & READ
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()

    # PHONE NUMBER REGEX --> USE TO SERACH TEXT FOR PHONE NUMBERS
    #phone_format = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    #phone_format=r'(\+?[0-9]{1,3}[ -.,]?)?(\(?[0-9]{3}\)?[ -.,]?[0-9]{3}[ -.,]?[0-9]{4})|\b[0-9]{10}\b'
    #phone_regex = r'[^0-9\/]((?:\(?\d{3}\)?.?)\d{3}.?\d{4})[^0-9|/]])'
    phone_regex = r'[^0-9\/]((?:\(?\d{3}\)?.?)\d{3}.?\d{4})[^0-9\/]'


    # FIND ALL PHONE NUMBERS IN THE TEXT USING THE REGEX
    matches = re.findall(phone_regex, text)

    # OPEN OUTPUT FILE AND WRITE FOUND MATCHES THERE
    with open('telephone_output.txt', 'w') as output_file:
        for match in matches:
            output_file.write(str(match).strip() + '\n')
            #output_file.write(match.strip() + '\n')
            #output_file.write(match + '\n')

if __name__ == '__main__':
    # IF USER ENTERED INCORRECT NUMBER OF ARGUMENTS... print error message
    if len(sys.argv) != 2:
        print("Please enter name of the program and the file you would like to read.")
    # ELSE... run program with the inputted file
    else:
        find_phone_numbers(sys.argv[1])
