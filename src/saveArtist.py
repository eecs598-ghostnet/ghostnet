from spotipy.oauth2 import SpotifyClientCredentials
import billboard, re, IPython, os, spotipy, lyricsgenius
from difflib import SequenceMatcher


genius = lyricsgenius.Genius("INSERT KEY", verbose=False, remove_section_headers=True, skip_non_songs=True)
genius_headers = lyricsgenius.Genius("INSERT KEY", verbose=False, remove_section_headers=False, skip_non_songs=True)
client_credentials_manager = SpotifyClientCredentials(client_id='INSERT KEY', client_secret='INSERT KEY')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

artists=["Ariana Grande"]
albums_seen=set()
songs_seen= set()

for artist in artists:
  search = sp.search(q=artist, type='album', limit=50)
  for album in search['albums']['items']:
    album_artist = album['artists'][0]['name']
    album_name = album['name'].split('(')[0]

    if (not album_artist==artist) or (album_name in albums_seen) or ("Live" in album_name):
      # print('%s in albums seen'%album_name)
      continue
    
    for remove_section_headers in [True,False]:
      headers_str = ('lyrics_headers','lyrics')[remove_section_headers]
      album_path = '../data/%s/%s/%s'%(headers_str, album_artist.replace(' ',''), re.sub(r"[/ \\]+", '', album_name))
      if not os.path.exists(album_path):
        os.makedirs(album_path)

    album_id = album['id'] 
    print('ALBUM: %s'%album_name)
    albums_seen.add(album_name)
    tracks = sp.album_tracks(album_id=album_id)
    for idx,t in enumerate(tracks['items']):
      track_name = t['name'].split('(')[0]

      if track_name in songs_seen or ("Remix" in track_name) or ("Radio" in track_name) or ("Edit" in track_name) or ("Mix" in track_name):
        continue
      
      songs_seen.add(track_name)
      genius_result = genius.search_song(track_name, album_artist)
      genius_headers_result = genius_headers.search_song(track_name, album_artist)

      try:
        if not(genius_result is None) and (SequenceMatcher(None,genius_result.title, track_name).ratio() > 0.7):
          pass
          song_path = '../data/%s/%s/%s/%s'%('lyrics', album_artist.replace(' ',''), re.sub(r'[/ \\]+', '', album_name), re.sub(r'[/ \\’]+', '', track_name))
          genius_result.save_lyrics(filename=song_path, extension='txt', verbose=False, overwrite=True)
        
        if not(genius_headers_result is None) and (SequenceMatcher(None,genius_headers_result.title, track_name).ratio() > 0.7):
          song_path = '../data/%s/%s/%s/%s'%('lyrics_headers', album_artist.replace(' ',''), re.sub(r'[/ \\]+', '', album_name), re.sub(r'[/ \\’]+', '', track_name))
          genius_headers_result.save_lyrics(filename=song_path, extension='txt', verbose=False, overwrite=True)
      except:
        IPython.embed()

