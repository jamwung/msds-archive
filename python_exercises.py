import numpy as np

def count_and_print(n):
    message = ''
    for i in range(1,n+1):
        if i%6 == 0:
            message += f'{i} foo\n'
        elif i%2 == 0:
            message += f'{i} fizz\n'
        elif i%3 == 0:
            message += f'{i} fuzz\n'
        else:
            message += f'{i}\n'
    message = message.strip('\n') 
    print(message)
    
    return message

def overlap_interval(start1, end1, start2, end2):
    higher_start = max(start1, start2)
    
    lower_end = min(end1, end2)

    if higher_start >= lower_end:
        print('No overlap!')
    else:
        print(f'{higher_start} {lower_end}')
        
def cosine(theta):
    n = 0
    maclaurin_term = 1
    maclaurin_sum = 0
    while abs(maclaurin_term) > 10**(-15):
        nfactorial = 1
        for i in range(1,2*n + 1):
            nfactorial = nfactorial*i
        maclaurin_term = ((-1)**n)*(theta**(2*n))/nfactorial
        maclaurin_sum += maclaurin_term
        n += 1
    return maclaurin_sum

def gcd(a,b):
    a, b = abs(a), abs(b)
    
    if a == 0 or b == 0:
        return max(a,b)
    elif max(a,b)%min(a,b) == 0:
        return min(a,b)
    else:
        gcdivisor = max(a,b)
        
        while a%gcdivisor != 0 or b%gcdivisor != 0:
            gcdivisor = gcdivisor - 1
        return gcdivisor

def biased_sum(*integers, base=2):
    sum = 0
    if integers == ():
        return None
    else:
        for i in integers:
            if i % base == 0:
                sum += 2 * i
            else:
                sum += i
        return sum

def last_in_sequence(digits):
    digits = ''.join(filter(str.isdigit, digits))
    n = 0
    last_digit = None
    
    for i in range(len(digits)):
        if int(digits[i]) == n:
            last_digit = n
            n += 1
            
            if n%10 == 0:
                n = 0
            else:
                pass
        else:
            pass

    return last_digit

def check_password(password): 
    if (len(password) >= 8 
        and any(x.isupper() for x in password) 
        and any(x.islower() for x in password) 
        and any(x.isdigit() for x in password)):
        return True
    else:
        return False

def is_palindrome(text):
    alpha_text = ''.join(filter(str.isalpha, text))
    lower_text = alpha_text.lower()
    mid_index = len(lower_text)//2
    
    if lower_text != '' and len(lower_text)%2 == 0:
        return lower_text[:mid_index] == lower_text[:mid_index-1:-1]
    elif len(lower_text)%2 == 1:
        return lower_text[:mid_index] == lower_text[:mid_index:-1]
    else:
        return False

def create_squares(num_stars):
    dash = 2 * ('+ ' + num_stars * '- ') + '+'

    star = '| ' + num_stars * '* '

    line = '| ' + num_stars * '  ' 

    square = dash + '\n'
    for i in range(num_stars):
        square += star + line + '|\n'
    square += dash + '\n'
    for i in range(num_stars):
        square += line + star + '|\n'
    square += dash + '\n'
    
    print(square)
    
    return square

def create_grid(num_squares, num_stars):
    square = ''
    border = ''
    fill1 = ''
    fill2 = ''

    border = num_squares * ('+ ' + num_stars *'- ') + '+\n'

    for j in range(num_squares):
        if j%2 != 0:
            fill1 += '| ' + num_stars * '  '
        else:
            fill1 += '| ' + num_stars * '* '
    fill1 += '|\n'

    for j in range(num_squares):
        if j%2 != 0:
            fill2 += '| ' + num_stars * '* '
        else:
            fill2 += '| ' + num_stars * '  '
    fill2 += '|\n'

    for k in range(num_squares):
        square += border
        if k%2 == 0:
            for l in range(num_stars):
                square += fill1
        else:
            for m in range(num_stars):
                square += fill2
    square += border
    
    print(square)
    
    return square

def chop(a_list):
    a_list[:] = a_list[1:-1]
    return None

def sum_multiples(a_list):
    total = 0
    for i in a_list:
        if i % 3 == 0 or i % 5 == 0:
            total += i
    return total

def rotate(numbers, k):
    k = k % len(numbers)
    rotated_num = numbers[-k:] + numbers[:-k]
    
    return rotated_num

def on_all(func, a_list):
    new_list = []
    for i in a_list:
        new_list.append(func(i))
    
    return(new_list)

def matrix_times_vector(mat, vec):
    
    new_mat = []
    for i in range(len(mat)):
        new_mat_value = 0
        for j in range(len(vec)):
            new_mat_value += mat[i][j] * vec[j]
        new_mat.append(new_mat_value)
    
    return new_mat

def coder(text, to_morse=True):
    alphanum_to_morse = {'A':'.-', 'B':'-...',
                         'C':'-.-.', 'D':'-..', 'E':'.',
                         'F':'..-.', 'G':'--.', 'H':'....',
                         'I':'..', 'J':'.---', 'K':'-.-',
                         'L':'.-..', 'M':'--', 'N':'-.',
                         'O':'---', 'P':'.--.', 'Q':'--.-',
                         'R':'.-.', 'S':'...', 'T':'-',
                         'U':'..-', 'V':'...-', 'W':'.--',
                         'X':'-..-', 'Y':'-.--', 'Z':'--..',
                         '1':'.----', '2':'..---', '3':'...--',
                         '4':'....-', '5':'.....', '6':'-....',
                         '7':'--...', '8':'---..', '9':'----.',
                         '0':'-----', ', ':'--..--', '.':'.-.-.-',
                         '?':'..--..', '/':'-..-.', '-':'-....-',
                         '(':'-.--.', ')':'-.--.-', ' ': ' ', '`':'`'}

    morse_to_alphanum = dict([(value, key)
                              for key, value in alphanum_to_morse.items()])

    translated = ''
    if to_morse:
        for i in text.upper():
            translated += alphanum_to_morse[i] + ' '
    else:
        text = text.replace('   ', ' ` ').replace(' ','=')
        text_as_list = text.split('=')
        
        index = 0
        while index < len(text_as_list):
            for j in text_as_list:
                text_as_list[index] = morse_to_alphanum[j]
                index += 1
                
        translated = ''.join(text_as_list).replace('`',' ')
    return translated.strip(' ')

def sort_by_key(items_with_keys, ascending=True):
    items_with_keys.sort(key=lambda x: (x[1], x[0]), reverse=not ascending)
    return items_with_keys

def count_words(text):
    
    text_to_list = text.lower().replace('\n', ' ').split(' ')
    
    text_to_list = [i for i in text_to_list if i] 
    
    word_count_dict = {}
    for j in text_to_list:
        word_count_dict[j] = text_to_list.count(j)
    return word_count_dict

def display_tree(a_dict, indent=''):
    display = ''

    for key, value in sorted(a_dict.items()):
        display += indent + str(key) + ':'
        if type(value) is dict:
            display += '\n' + display_tree(value, indent+'  ')
        else:
            display += ' ' + str(value) + '\n'

    return display

def get_nested_key_value(nested_dict, key):
    key_as_list = key.split('.')
    
    new_dict = nested_dict
    try:
        for i in key_as_list:
            new_dict = new_dict[i]
    except:
        return None
    else:
        return new_dict

def value_counts(a_list, out_path):
    a_dict = {}
    for element in set(a_list):
        a_dict[element] = a_list.count(element)
        
    sorted_list = sorted(a_dict.items(), key=lambda x: (-x[1], x[0]))
    
    with open(out_path, 'w') as file:
        for item, count in sorted_list:
            file.write(f'{item},{str(count)}\n')

def is_subset(sublist, superlist, strict=True):
    if strict:
        output = False
        fit_interval = len(superlist) - len(sublist) + 1
        for i in range(fit_interval):
            if superlist[i:i+len(sublist)] == sublist:
                output = True
                break
    else:
        output = True
        for element in set(sublist):
            if element not in superlist:
                output = False
                break
    return output

def count_words(input_file, output_file):
    with open(input_file, 'r') as i_file:
        text = i_file.read()
    text_as_list = text.lower().split()
    
    wc_dict = {}
    for word in set(text_as_list):
        wc_dict[word] = text_as_list.count(word)
    
    with open(output_file, 'wb') as o_file:
        pickle.dump(wc_dict, o_file)

class Person:
    def __init__(self, position=(0,0)):
        self.position = (position[0], position[1])
        self.infected = False
        
    def move(self, dx=0, dy=0):
        self.position = (self.position[0] + dx, self.position[1] + dy)
    
    def get_position(self):
        return self.position
    
    def is_infected(self):
        return self.infected
    
    def set_infected(self):
        self.infected = True
    
    def get_infected(self, other, threshold):
        if other.is_infected():
            distance = ((other.position[0] - self.position[0]) ** 2 + 
                       (other.position[1] - self.position[1]) ** 2) ** 0.5
            if distance < threshold:
                self.set_infected()

class QuarantinedPerson(Person):
    def move(self, dx=0, dy=0):
        pass

def file_lines(**filepaths):
    file_dict = {}
    for key, filepath in filepaths.items():
        try:
            with open(filepath, 'r') as file:
                text = file.read()
                file_dict.update({key:text.count('\n')})
        except:
            pass
    return file_dict

class TenDivError(ValueError):
    def __init__(self, exception):
        super().__init__(exception)
        
def ten_div(num, denom):
    try:
        if num < 0 or num > 10:
            raise TenDivError()
        else:
            return num/denom
    except Exception as e:
        raise TenDivError(f'Error encountered: {e}')

def most_frequent(filepath):
    with open(filepath, 'r') as file:
        text = file.read()
    text_as_list = text.lower().split()
    
    wc_dict = {}
    for word in set(text_as_list):
        wc_dict[word] = text_as_list.count(word)
    
    sort_as_list = sorted(wc_dict.items(), key=lambda x: (-x[1], x[0]))
    sort9_as_list = sort_as_list[:9]
    
    most_freq_words = [key for key, value in sort9_as_list]
    
    return most_freq_words