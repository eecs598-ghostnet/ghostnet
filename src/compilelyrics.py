from english2phoneme import e2p
import IPython, glob, os, re, math
from num2words import num2words

def _conv_num(match):
    return re.sub(r'[^\w\n ]+',"", num2words(int(match.group())))

artist_paths = glob.glob('../data/lyrics_headers/*')
artist_paths += glob.glob('../data/lyrics/*')

for artist_path in artist_paths:
    print(artist_path)
    if (not os.path.exists(os.path.join(artist_path, 'lyrics.txt'))) and os.path.isdir(artist_path):
        headers = artist_path.split('/')[-2]
        artist = artist_path.split('/')[-1]

        output_file = open(os.path.join(artist_path, 'lyrics.txt'), "w+", encoding='utf-8')
        song_path = os.path.join(artist_path, '*/*.txt')
        songs = glob.glob(song_path)
        
        lyrics = ''
        for song in songs:
            lines = open(song, encoding='utf-8').\
                read().strip()
            lines = re.sub('\n{3,}', '\n\n', lines)
            lines = lines.split('\n')
            include_lines = True
            for line in lines:
                if len(line)>0 and line[0] == '[':
                    if (':' in line) or ('-' in line):
                        header_artist = re.split('[:-]',re.sub('[\[\]]+', ' ', line))[-1].strip()
                        include_lines = (artist in header_artist)
                    else:
                        include_lines = True
                    
                    continue
                
                if include_lines:
                    lyrics += line + '\n'
            lyrics += '\n\n'
        
        lyrics = re.sub(r'-'," ", lyrics)
        lyrics = re.sub(r'[^\w\n ]+',"", lyrics)
        lyrics = re.sub(r'\d+', _conv_num, lyrics)
        lyrics = re.sub('_+','',lyrics)
        lyrics = re.sub('\n{3,}', '\n\n\n', lyrics)
        lyrics = lyrics.lower()
        lyrics = lyrics.strip('\n')
        
        output_file.write(lyrics)

