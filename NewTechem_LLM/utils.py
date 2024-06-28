import time
import re

def timeit(f):
    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func:{} took: {} sec'.format(f, (te - ts)))
        return result

    return timed



def filter_english(text):
    # Define a regular expression pattern to match non-English characters
    pattern = re.compile(r'[^a-zA-Z\s.,!?\'"-]')
    
    # Substitute all non-English characters with an empty string
    filtered_text = pattern.sub('', text)
    
    # Remove trailing punctuation from each word
    filtered_text = ' '.join(word.strip('.,!?\'"-') for word in filtered_text.split())
    
    return filtered_text.strip() + '.'
